import math
import asyncio
from collections import deque
from dataclasses import dataclass

import cv2
import mediapipe as mp
from ultralytics import YOLO


@dataclass
class PassFrameStatus:
    chest: int
    overhead: int
    ground: int
    released_now: bool
    frame_idx: int
    head_y: int | None
    chest_top_y: int | None
    chest_bot_y: int | None
    ground_y: int | None
    ball_center_y: int | None
    dist_ball_to_wrist_px: float | None


class PassAnalyzer:

    def __init__(self):
        self.YOLO_CONFIDENCE = 0.3
        self.INFERENCE_WIDTH = 512
        self.GROUND_TOLERANCE_FRAC = 0.02

        self.NEAR_DISTANCE_FRACTION = 0.06
        self.FAR_DISTANCE_FRACTION = 0.12
        self.NEAR_MIN_PIXELS = 24
        self.FAR_MIN_PIXELS = 65
        self.NEAR_CONSECUTIVE_FRAMES = 1
        self.FAR_CONSECUTIVE_FRAMES = 1
        self.MIN_AWAY_SPEED_PER_FRAME = 0.5

        self.CHEST_MARGIN_PIXELS = 30
        self.HEAD_MARGIN_PIXELS = 5
        self.PASS_THRESHOLD_REPS = 4

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.LEFT_WRIST, self.RIGHT_WRIST = 15, 16
        self.LEFT_SHOULDER, self.RIGHT_SHOULDER = 11, 12
        self.FACE_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.LEFT_ANKLE, self.RIGHT_ANKLE = 27, 28
        self.LEFT_HEEL, self.RIGHT_HEEL = 29, 30
        self.LEFT_TOE, self.RIGHT_TOE = 31, 32

        self.yolo_model: YOLO | None = None
        self.model_yolo: YOLO | None = None
        self.callback = None

        self.frame_idx = 0

        self.chest_count = 0
        self.overhead_count = 0
        self.ground_count = 0

        self.CHEST_IDLE = 0
        self.CHEST_WAIT_RELEASE = 1
        self.CHEST_WAIT_RETURN = 2
        self.chest_state = self.CHEST_IDLE
        self.chest_cycle_is_legal = False

        self.OVERHEAD_IDLE = 0
        self.OVERHEAD_WAIT_RETURN = 1
        self.overhead_state = self.OVERHEAD_IDLE
        self.overhead_cycle_is_legal = False
        self.overhead_active = False

        self.last_labeled_type = None
        self.idle_frame_streak = 0
        self.IDLE_RESET_FRAMES = 10

        self._rel_near_streak = 0
        self._rel_far_streak = 0
        self._rel_seen_near = False
        self._rel_recent_distances = deque(maxlen=4)

    def reset_session(self):
        self.__init__()

    def set_callback(self, cb):
        self.callback = cb

    def set_model(self, yolo_model: YOLO):
        self.yolo_model = yolo_model

    def _async_emit(self, payload: dict):
        if not self.callback:
            return
        try:
            if asyncio.iscoroutinefunction(self.callback):
                asyncio.create_task(self.callback(payload))
            else:
                self.callback(payload)
        except Exception as e:
            print("[callback error]", e)

    def _release_detector_update(self, distance_px, wrists_visible, near_thr, far_thr):
        if (not wrists_visible) or (distance_px is None):
            return False

        is_near = distance_px < near_thr
        is_far = distance_px > far_thr

        self._rel_near_streak = self._rel_near_streak + 1 if is_near else 0
        self._rel_far_streak = self._rel_far_streak + 1 if is_far else 0

        prev = self._rel_recent_distances[-1] if self._rel_recent_distances else distance_px
        self._rel_recent_distances.append(distance_px)
        away_speed = distance_px - prev if prev is not None else 0

        if self._rel_near_streak >= self.NEAR_CONSECUTIVE_FRAMES:
            self._rel_seen_near = True

        released = (
            self._rel_seen_near and
            self._rel_far_streak >= self.FAR_CONSECUTIVE_FRAMES and
            away_speed >= self.MIN_AWAY_SPEED_PER_FRAME
        )

        if released:
            self._rel_near_streak = 0
            self._rel_far_streak = 0
            self._rel_seen_near = False
            self._rel_recent_distances.clear()
            return True

        return False

    def _pick_ball_box(self, yolo_results, names, infer_h, infer_w, conf_min=0.5):
        if not yolo_results:
            return None
        boxes = yolo_results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        for box in boxes:
            conf = float(box.conf.cpu().numpy())
            if conf < conf_min:
                continue

            cls_id = int(box.cls.cpu().numpy())

            if isinstance(names, dict):
                cname = str(names.get(cls_id, "")).lower()
            elif isinstance(names, list) and 0 <= cls_id < len(names):
                cname = str(names[cls_id]).lower()
            else:
                cname = ""

            if ("ball" not in cname) and ("basket" not in cname) and (cls_id != 0):
                continue

            x1, y1, x2, y2 = box.xyxy.cpu().numpy()[0].astype(int)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(infer_w - 1, x2)
            y2 = min(infer_h - 1, y2)
            return (x1, y1, x2, y2)

        return None

    def _estimate_ground_y(self, lm, H):
        if lm is None:
            return int(0.85 * H)
        idxs = [
            self.LEFT_ANKLE, self.RIGHT_ANKLE,
            self.LEFT_HEEL, self.RIGHT_HEEL,
            self.LEFT_TOE, self.RIGHT_TOE
        ]
        ys = []
        for i in idxs:
            p = lm[i]
            if getattr(p, 'visibility', 0) > 0.2 and 0 <= p.y <= 1:
                ys.append(p.y * H)
        return int(min(H - 1, max(ys))) if ys else int(0.85 * H)

    def _estimate_head_y(self, lm, H):
        if lm is None:
            return int(0.27 * H)
        ys = []
        for i in self.FACE_IDS:
            p = lm[i]
            if getattr(p, 'visibility', 0) > 0.2 and 0 <= p.y <= 1:
                ys.append(p.y * H)
        return int(min(ys)) if ys else int(0.27 * H)

    def _estimate_shoulder_y(self, lm, H):
        if lm is None:
            return int(0.36 * H)
        l_sh = lm[self.LEFT_SHOULDER]
        r_sh = lm[self.RIGHT_SHOULDER]
        if (
            getattr(l_sh, 'visibility', 0) > 0.2 and
            getattr(r_sh, 'visibility', 0) > 0.2 and
            0 <= l_sh.y <= 1 and
            0 <= r_sh.y <= 1
        ):
            return int(((l_sh.y + r_sh.y) * 0.5) * H)
        return int(0.36 * H)

    def process_frame(self, frame_bgr):
        model = self.yolo_model or self.model_yolo
        if model is None:
            payload = {
                "status": "error",
                "msg": "YOLO model not set",
                "frame_idx": int(self.frame_idx)
            }
            self._async_emit(payload)
            self.frame_idx += 1
            return payload

        H, W = frame_bgr.shape[:2]

        pose_res = self.pose.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        lm = pose_res.pose_landmarks.landmark if pose_res.pose_landmarks else None

        head_line_y = self._estimate_head_y(lm, H)
        shoulder_line_y = self._estimate_shoulder_y(lm, H)
        ground_line_y = self._estimate_ground_y(lm, H)

        chest_top_y = max(0, head_line_y - self.HEAD_MARGIN_PIXELS)
        chest_bot_y = min(H - 1, shoulder_line_y + self.CHEST_MARGIN_PIXELS)
        head_ceiling_y = head_line_y - self.HEAD_MARGIN_PIXELS

        ground_tolerance = max(3, int(self.GROUND_TOLERANCE_FRAC * H))

        infer_h = max(16, int(H * self.INFERENCE_WIDTH / max(1, W)))
        small_img = cv2.resize(frame_bgr, (self.INFERENCE_WIDTH, infer_h))
        try:
            yolo_out = model(small_img, verbose=False, conf=self.YOLO_CONFIDENCE)
        except Exception:
            yolo_out = None

        ball_box_small = self._pick_ball_box(
            yolo_out,
            getattr(model, "names", {}),
            infer_h,
            self.INFERENCE_WIDTH,
            conf_min=self.YOLO_CONFIDENCE
        )

        ball_center_x = ball_center_y = ball_bottom_y = None
        if ball_box_small:
            x1s, y1s, x2s, y2s = ball_box_small
            scale_x = W / self.INFERENCE_WIDTH
            scale_y = H / infer_h
            x1 = int(x1s * scale_x)
            y1 = int(y1s * scale_y)
            x2 = int(x2s * scale_x)
            y2 = int(y2s * scale_y)
            ball_center_x = (x1 + x2) // 2
            ball_center_y = (y1 + y2) // 2
            ball_bottom_y = y2

        wrists_visible = False
        distance_ball_to_wrist = None
        left_wrist_y = right_wrist_y = None

        if lm is not None and ball_center_x is not None and ball_center_y is not None:
            dists = []
            for idx in (self.LEFT_WRIST, self.RIGHT_WRIST):
                p = lm[idx]
                if getattr(p, 'visibility', 0) > 0.2 and 0 <= p.x <= 1 and 0 <= p.y <= 1:
                    wx, wy = int(p.x * W), int(p.y * H)
                    dists.append(math.hypot(ball_center_x - wx, ball_center_y - wy))
                    if idx == self.LEFT_WRIST:
                        left_wrist_y = wy
                    else:
                        right_wrist_y = wy
            if dists:
                wrists_visible = True
                distance_ball_to_wrist = min(dists)

        near_threshold = max(self.NEAR_MIN_PIXELS, int(self.NEAR_DISTANCE_FRACTION * min(W, H)))
        far_threshold = max(self.FAR_MIN_PIXELS, int(self.FAR_DISTANCE_FRACTION * min(W, H)))

        released_now = self._release_detector_update(
            distance_ball_to_wrist,
            wrists_visible,
            near_threshold,
            far_threshold
        )

        if ball_bottom_y is not None and ball_bottom_y >= ground_line_y - ground_tolerance:
            if self.last_labeled_type != "GROUND":
                self.ground_count += 1
                self.last_labeled_type = "GROUND"
                self.idle_frame_streak = 0
        else:
            self.idle_frame_streak += 1
            if self.idle_frame_streak >= self.IDLE_RESET_FRAMES:
                self.last_labeled_type = None

        snap = PassFrameStatus(
            chest=self.chest_count,
            overhead=self.overhead_count,
            ground=self.ground_count,
            released_now=bool(released_now),
            frame_idx=int(self.frame_idx),
            head_y=int(head_line_y) if head_line_y is not None else None,
            chest_top_y=int(chest_top_y),
            chest_bot_y=int(chest_bot_y),
            ground_y=int(ground_line_y),
            ball_center_y=int(ball_center_y) if ball_center_y is not None else None,
            dist_ball_to_wrist_px=float(distance_ball_to_wrist) if distance_ball_to_wrist is not None else None
        )

        self._async_emit({
            "status": "ok",
            "frame_idx": snap.frame_idx,
            "chest": snap.chest,
            "overhead": snap.overhead,
            "ground": snap.ground,
            "released_now": snap.released_now,
            "ball_center_y": snap.ball_center_y,
            "dist_ball_to_wrist_px": snap.dist_ball_to_wrist_px,
        })

        self.frame_idx += 1

        return {
            "status": "ok",
            "frame_idx": snap.frame_idx,
            "chest": snap.chest,
            "overhead": snap.overhead,
            "ground": snap.ground,
            "released_now": snap.released_now,
            "ball_center_y": snap.ball_center_y,
            "dist_ball_to_wrist_px": snap.dist_ball_to_wrist_px,
        }

    def final_summary(self):
        total_reps = self.chest_count + self.overhead_count + self.ground_count
        did_all_three = (
            self.chest_count > 0 and
            self.overhead_count > 0 and
            self.ground_count > 0
        )
        passed = (total_reps >= self.PASS_THRESHOLD_REPS) or did_all_three

        summary = {
            "status": "final",
            "chest": int(self.chest_count),
            "overhead": int(self.overhead_count),
            "ground": int(self.ground_count),
            "total": int(total_reps),
            "passed": bool(passed),
        }

        self._async_emit(summary)
        return summary