# 工单编号：人工智能CV-智能交通路口管理-交通拥堵检测任务
# 优化版：YOLO车辆检测 + ByteTrack跟踪 + 分车道速度/密度/占用率 + 连续帧拥堵报警

import argparse
import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
import time
import os
import sys


class TrafficCongestionDetector:
    def __init__(self, video_path, config):
        """
        初始化交通拥堵检测器
        :param video_path: 输入视频路径
        :param config: 配置参数
        """
        self.video_path = str(video_path)
        self.config = config

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件：{self.video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps is None or self.fps <= 1:
            self.fps = config.get("fallback_fps", 25)

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 加载 YOLO 模型
        model_path = config.get("model_path", "yolov8s.pt")
        if not os.path.exists(model_path):
            print(f"提示：模型文件 {model_path} 不存在，Ultralytics 会尝试自动下载。")
        self.model = YOLO(model_path)

        # YOLO 参数
        self.vehicle_classes = config.get("vehicle_classes", [2, 3, 5, 7])
        self.conf = config.get("conf", 0.35)
        self.iou = config.get("iou", 0.5)
        self.imgsz = config.get("imgsz", 640)
        self.tracker = config.get("tracker", "bytetrack.yaml")

        # 速度参数
        self.pixel_to_meter = config.get("pixel_to_meter", 0.05)
        self.speed_smooth_window = config.get("speed_smooth_window", 8)
        self.track_history_len = config.get("track_history_len", 30)
        self.stale_track_frames = config.get("stale_track_frames", int(self.fps * 2))

        # 拥堵阈值
        self.speed_threshold = config.get("speed_threshold", 20)
        self.density_threshold = config.get("density_threshold", 8)
        self.occupancy_threshold = config.get("occupancy_threshold", 60)
        self.min_count_for_speed = config.get("min_count_for_speed", 3)

        # 连续帧判断，避免闪烁误报
        self.consecutive_frames = config.get(
            "consecutive_frames",
            max(5, int(self.fps * 0.5))
        )
        self.warning_cooldown = config.get("warning_cooldown", 1.0)

        # 显示和保存
        self.show_window = config.get("show_window", True)
        self.save_output = config.get("save_output", True)
        self.draw_tracks = config.get("draw_tracks", True)
        self.draw_boxes = config.get("draw_boxes", True)

        # 车道配置
        self.lanes = self._prepare_lanes(config["lanes"])

        # 轨迹信息
        self.track_history = defaultdict(lambda: deque(maxlen=self.track_history_len))
        self.speed_history = defaultdict(lambda: deque(maxlen=self.speed_smooth_window))
        self.track_last_seen = {}
        self.track_last_lane = {}

        # 车道状态
        self.lane_states = {
            lane["id"]: {
                "consecutive": 0,
                "warning_active": False,
                "last_warning_time": 0,
                "warning_count": 0,
                "congested_frames": 0,
                "last_metrics": {}
            }
            for lane in self.lanes
        }

        # 统计信息
        self.frame_idx = 0
        self.processing_times = []
        self.total_alerts = 0

        print(f"初始化完成：{self.width}x{self.height}, FPS={self.fps:.2f}, 车道数={len(self.lanes)}")

    def _prepare_lanes(self, lanes):
        """
        预处理车道区域，计算车道多边形面积
        """
        prepared = []
        for lane in lanes:
            pts = np.array(lane["detection_area"], dtype=np.int32)
            area = abs(cv2.contourArea(pts))

            if area <= 0:
                raise ValueError(f"车道 {lane.get('id')} 的 detection_area 面积无效，请检查坐标。")

            item = dict(lane)
            item["polygon"] = pts
            item["area"] = area
            item["capacity"] = lane.get("capacity", self.density_threshold + 2)

            prepared.append(item)

        return prepared

    @staticmethod
    def _box_center(box):
        """
        获取检测框中心点
        """
        x1, y1, x2, y2 = box
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    @staticmethod
    def _box_area(box):
        """
        计算检测框面积
        """
        x1, y1, x2, y2 = box
        return max(0, x2 - x1) * max(0, y2 - y1)

    def _point_in_lane(self, point, lane):
        """
        判断点是否在车道区域内
        """
        return cv2.pointPolygonTest(lane["polygon"], point, False) >= 0

    def _get_lane_id_for_box(self, box):
        """
        根据车辆中心点判断车辆属于哪个车道
        """
        cx, cy = self._box_center(box)

        for lane in self.lanes:
            if self._point_in_lane((cx, cy), lane):
                return lane["id"]

        return None

    def _calculate_speed(self, track_id, current_pos):
        """
        根据上一帧中心点计算速度。
        注意：必须先算速度，再把当前点加入历史。
        """
        history = self.track_history[track_id]

        if len(history) == 0:
            return None

        prev_pos = history[-1]
        displacement_px = np.linalg.norm(np.array(current_pos) - np.array(prev_pos))

        displacement_m = displacement_px * self.pixel_to_meter
        speed = displacement_m * self.fps * 3.6

        # 过滤异常速度，防止跟踪ID跳变导致速度过大
        max_reasonable_speed = self.config.get("max_reasonable_speed", 140)
        if speed > max_reasonable_speed:
            return None

        self.speed_history[track_id].append(speed)

        # 平滑速度
        return float(np.mean(self.speed_history[track_id]))

    def _cleanup_stale_tracks(self):
        """
        清理长时间没有出现的轨迹，避免内存一直增长
        """
        stale_ids = [
            tid for tid, last_seen in self.track_last_seen.items()
            if self.frame_idx - last_seen > self.stale_track_frames
        ]

        for tid in stale_ids:
            self.track_history.pop(tid, None)
            self.speed_history.pop(tid, None)
            self.track_last_seen.pop(tid, None)
            self.track_last_lane.pop(tid, None)

    def draw_lane_markers(self, frame):
        """
        绘制车道检测区域
        """
        overlay = frame.copy()

        for lane in self.lanes:
            lane_id = lane["id"]
            is_congested = self.lane_states[lane_id]["warning_active"]

            color = (0, 0, 255) if is_congested else (0, 255, 0)

            cv2.polylines(frame, [lane["polygon"]], isClosed=True, color=color, thickness=2)
            cv2.fillPoly(overlay, [lane["polygon"]], color)

            cv2.putText(
                frame,
                f"Lane {lane_id}",
                lane["label_position"],
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        alpha = self.config.get("lane_fill_alpha", 0.10)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def _init_lane_metrics(self):
        """
        初始化每一帧的车道统计信息
        """
        metrics = {}

        for lane in self.lanes:
            metrics[lane["id"]] = {
                "count": 0,
                "box_area_sum": 0.0,
                "speeds": [],
                "track_ids": []
            }

        return metrics

    def _update_lane_state(self, lane, metrics):
        """
        根据车道统计信息更新拥堵状态
        """
        lane_id = lane["id"]

        count = metrics["count"]
        avg_speed = float(np.mean(metrics["speeds"])) if metrics["speeds"] else 0.0

        # 占用率：车辆检测框面积 / 车道区域面积
        # 这是近似算法，适合课程项目和演示。
        occupancy = min(100.0, metrics["box_area_sum"] / lane["area"] * 100)

        # 容量占用率：车辆数量 / 车道容量
        capacity = max(1, lane.get("capacity", 10))
        capacity_occupancy = min(100.0, count / capacity * 100)

        # 拥堵判断逻辑
        by_density = count >= self.density_threshold
        by_occupancy = occupancy >= self.occupancy_threshold or capacity_occupancy >= self.occupancy_threshold
        by_speed = count >= self.min_count_for_speed and avg_speed > 0 and avg_speed <= self.speed_threshold

        instant_congested = by_density or by_occupancy or by_speed

        state = self.lane_states[lane_id]

        # 连续帧判断，避免一帧误检就报警
        if instant_congested:
            state["consecutive"] += 1
        else:
            state["consecutive"] = max(0, state["consecutive"] - 1)

        warning_active = state["consecutive"] >= self.consecutive_frames
        state["warning_active"] = warning_active

        if warning_active:
            state["congested_frames"] += 1

        state["last_metrics"] = {
            "count": count,
            "avg_speed": avg_speed,
            "occupancy": occupancy,
            "capacity_occupancy": capacity_occupancy,
            "by_density": by_density,
            "by_occupancy": by_occupancy,
            "by_speed": by_speed,
            "instant_congested": instant_congested,
            "warning_active": warning_active
        }

    def _draw_vehicle(self, frame, box, track_id, lane_id, speed):
        """
        绘制车辆框、ID、速度、车道信息
        """
        x1, y1, x2, y2 = map(int, box)

        color = (0, 255, 0)
        if lane_id is not None and self.lane_states[lane_id]["warning_active"]:
            color = (0, 0, 255)

        if self.draw_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        speed_text = "--" if speed is None else f"{speed:.1f}"
        lane_text = "-" if lane_id is None else str(lane_id)

        label = f"ID:{track_id} L:{lane_text} V:{speed_text}km/h"

        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

        if self.draw_tracks and len(self.track_history[track_id]) >= 2:
            pts = np.array(self.track_history[track_id], dtype=np.int32)
            cv2.polylines(frame, [pts], isClosed=False, color=(0, 255, 255), thickness=2)

    def _draw_dashboard(self, frame):
        """
        绘制左上角统计面板
        """
        y = 30

        cv2.putText(
            frame,
            "Traffic Congestion Detection",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2
        )

        y += 35

        for lane in self.lanes:
            lane_id = lane["id"]
            state = self.lane_states[lane_id]
            m = state["last_metrics"]

            if not m:
                continue

            status = "CONGESTION" if state["warning_active"] else "NORMAL"
            color = (0, 0, 255) if state["warning_active"] else (0, 255, 0)

            text = (
                f"Lane {lane_id}: {status} | "
                f"Count:{m['count']} | "
                f"Speed:{m['avg_speed']:.1f}km/h | "
                f"Occ:{m['occupancy']:.1f}% | "
                f"Hold:{state['consecutive']}/{self.consecutive_frames}"
            )

            cv2.putText(
                frame,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                color,
                2
            )

            y += 28

        active_lanes = [
            str(lid)
            for lid, s in self.lane_states.items()
            if s["warning_active"]
        ]

        if active_lanes:
            alarm_text = "ALARM: TRAFFIC CONGESTION - Lane " + ",".join(active_lanes)

            cv2.putText(
                frame,
                alarm_text,
                (10, self.height - 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3
            )

    def _maybe_print_warning(self):
        """
        控制台报警，带冷却时间
        """
        now = time.time()

        for lane_id, state in self.lane_states.items():
            if not state["warning_active"]:
                continue

            if now - state["last_warning_time"] < self.warning_cooldown:
                continue

            m = state["last_metrics"]

            print(
                f"预警：车道 {lane_id} 发生拥堵 | "
                f"车辆数={m['count']} | "
                f"平均速度={m['avg_speed']:.1f}km/h | "
                f"占用率={m['occupancy']:.1f}% | "
                f"连续帧={state['consecutive']}"
            )

            state["last_warning_time"] = now
            state["warning_count"] += 1
            self.total_alerts += 1

    def detect_congestion(self, frame):
        """
        单帧检测：车辆检测、跟踪、分车道统计、拥堵判断
        """
        lane_metrics = self._init_lane_metrics()

        results = self.model.track(
            frame,
            persist=True,
            classes=self.vehicle_classes,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            tracker=self.tracker,
            verbose=False
        )

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()

            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().numpy()
            else:
                # 极少数情况下没有 track id，使用临时ID兜底
                track_ids = np.array([-(i + 1) for i in range(len(boxes))])

            for box, track_id in zip(boxes, track_ids):
                track_id = int(track_id)
                center = self._box_center(box)

                # 核心修复：先计算速度，再追加当前点
                speed = self._calculate_speed(track_id, center)

                self.track_history[track_id].append(center)
                self.track_last_seen[track_id] = self.frame_idx

                lane_id = self._get_lane_id_for_box(box)
                self.track_last_lane[track_id] = lane_id

                if lane_id is not None:
                    lane_metrics[lane_id]["count"] += 1
                    lane_metrics[lane_id]["box_area_sum"] += self._box_area(box)
                    lane_metrics[lane_id]["track_ids"].append(track_id)

                    if speed is not None and speed > 0:
                        lane_metrics[lane_id]["speeds"].append(speed)

                self._draw_vehicle(frame, box, track_id, lane_id, speed)

        # 更新每个车道拥堵状态
        for lane in self.lanes:
            self._update_lane_state(lane, lane_metrics[lane["id"]])

        # 控制台报警
        self._maybe_print_warning()

        # 绘制统计面板
        self._draw_dashboard(frame)

        # 清理失效轨迹
        self._cleanup_stale_tracks()

        return frame

    def run(self, output_path="output_congestion.mp4"):
        """
        主循环
        """
        out = None

        if self.save_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

            if not out.isOpened():
                print(f"警告：无法创建输出视频 {output_path}，将不会保存结果。")
                out = None

        print("开始处理视频，按 q 可退出。")

        while True:
            start_time = time.time()

            ret, frame = self.cap.read()
            if not ret:
                break

            # 画车道区域
            self.draw_lane_markers(frame)

            # 检测拥堵
            processed_frame = self.detect_congestion(frame)

            # FPS 显示
            elapsed = time.time() - start_time
            self.processing_times.append(elapsed)

            fps_display = 1.0 / elapsed if elapsed > 0 else 0

            cv2.putText(
                processed_frame,
                f"FPS: {fps_display:.1f}",
                (10, self.height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            if out is not None:
                out.write(processed_frame)

            if self.show_window:
                cv2.imshow("Traffic Congestion Detection", processed_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            self.frame_idx += 1

        self.cap.release()

        if out is not None:
            out.release()

        if self.show_window:
            cv2.destroyAllWindows()

        self.generate_report(output_path)
        print(f"处理完成，结果视频：{output_path}")

    def generate_report(self, output_path):
        """
        生成性能报告
        """
        if not self.processing_times:
            print("没有处理任何帧。")
            return

        total_time = sum(self.processing_times)
        avg_fps = len(self.processing_times) / total_time if total_time > 0 else 0

        avg_time = np.mean(self.processing_times) * 1000
        min_time = np.min(self.processing_times) * 1000
        max_time = np.max(self.processing_times) * 1000

        lane_lines = []

        for lane_id, state in self.lane_states.items():
            seconds = state["congested_frames"] / self.fps

            lane_lines.append(
                f"车道 {lane_id}: 报警次数={state['warning_count']}, "
                f"拥堵帧数={state['congested_frames']}, 约 {seconds:.2f} 秒"
            )

        report = f"""
===== 交通拥堵检测系统性能报告 =====
输入视频          : {self.video_path}
输出视频          : {output_path}
视频分辨率        : {self.width}x{self.height}
视频FPS           : {self.fps:.2f}
总处理帧数        : {len(self.processing_times)}
平均处理FPS       : {avg_fps:.2f}
单帧最短处理时间  : {min_time:.2f} ms
单帧最长处理时间  : {max_time:.2f} ms
单帧平均处理时间  : {avg_time:.2f} ms
报警总次数        : {self.total_alerts}
连续帧报警阈值    : {self.consecutive_frames} 帧，约 {self.consecutive_frames / self.fps:.2f} 秒
速度阈值          : {self.speed_threshold} km/h
车辆数阈值        : {self.density_threshold} 辆/车道
占用率阈值        : {self.occupancy_threshold}%

{chr(10).join(lane_lines)}
=====================================
"""

        print(report)

        with open("performance_report.txt", "w", encoding="utf-8") as f:
            f.write(report)


# ==================== 配置参数 ====================
# 重点：车道区域一定要根据你的视频重新改坐标
config = {
    "model_path": "yolov8s.pt",

    # COCO类别：car=2, motorcycle=3, bus=5, truck=7
    "vehicle_classes": [2, 3, 5, 7],

    # YOLO检测参数
    "conf": 0.35,
    "iou": 0.5,
    "imgsz": 640,
    "tracker": "bytetrack.yaml",

    # 像素到米的比例
    # 没有实际标定时，先用 0.03 ~ 0.08 调试
    "pixel_to_meter": 0.05,
    "max_reasonable_speed": 140,

    # 拥堵判断阈值
    "speed_threshold": 20,        # km/h，平均速度低于该值可能拥堵
    "density_threshold": 8,       # 单车道车辆数阈值
    "occupancy_threshold": 60,    # 占用率阈值
    "min_count_for_speed": 3,     # 至少3辆车才启用速度拥堵判断

    # 连续帧报警
    # 25FPS下，10帧约0.4秒，满足报警小于1秒
    "consecutive_frames": 10,
    "warning_cooldown": 1.0,

    # 显示与保存
    "show_window": True,
    "save_output": True,
    "draw_tracks": True,
    "draw_boxes": True,
    "lane_fill_alpha": 0.10,

    # 车道检测区域
    # 这里的坐标只是示例，你必须按自己的视频画面改
    "lanes": [
        {
            "id": 1,
            "detection_area": [(100, 500), (400, 500), (400, 700), (100, 700)],
            "label_position": (150, 470),
            "capacity": 10
        },
        {
            "id": 2,
            "detection_area": [(400, 500), (700, 500), (700, 700), (400, 700)],
            "label_position": (450, 470),
            "capacity": 10
        },
        {
            "id": 3,
            "detection_area": [(700, 500), (1000, 500), (1000, 700), (700, 700)],
            "label_position": (750, 470),
            "capacity": 10
        }
    ]
}


def parse_args():
    parser = argparse.ArgumentParser(description="交通拥堵检测系统")

    parser.add_argument(
        "--video",
        type=str,
        default=r"D:\python\visdrone2019\tools\jiaotonglukou.mp4",
        help="输入视频路径"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="output_congestion.mp4",
        help="输出视频路径"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="YOLO模型路径，例如 yolov8s.pt / yolov8n.pt / best.pt"
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="不弹出显示窗口，只保存视频"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.model:
        config["model_path"] = args.model

    if args.no_show:
        config["show_window"] = False

    try:
        detector = TrafficCongestionDetector(args.video, config)
        detector.run(output_path=args.output)

    except Exception as e:
        print(f"程序运行出错：{e}")
        sys.exit(1)