import os
import math
import asyncio
from statistics import mean
from collections import deque
from dataclasses import asdict

import cv2
import torch
import mediapipe as mp
from ultralytics import YOLO


def angle_3pts(a, b, c):
    bax = a[0] - b[0]; bay = a[1] - b[1]
    bcx = c[0] - b[0]; bcy = c[1] - b[1]
    dot = bax * bcx + bay * bcy
    na = math.hypot(bax, bay) + 1e-6
    nc = math.hypot(bcx, bcy) + 1e-6
    cosv = max(-1.0, min(1.0, dot / (na * nc)))
    return math.degrees(math.acos(cosv))


class ShootingAnalyzer:

    def __init__(self, model_path=None, callback=None):
        self.callback = callback

        self.yolo = None
        self.model_yolo = None
        if model_path is None:
            model_path = os.getenv("YOLO_MODEL_PATH")
        if model_path:
            try:
                self.yolo = YOLO(model_path)
                self.yolo.to('cuda' if torch.cuda.is_available() else 'cpu')
            except Exception:
                self.yolo = None

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )

        self.KNEE_BENT_THRESHOLD_DEG = 165
        self.NEAR_FRACTION = 0.06
        self.FAR_FRACTION  = 0.12
        self.NEAR_MIN_PX   = 24
        self.FAR_MIN_PX    = 65

        self.CLOSE_REQ = 3
        self.FAR_REQ   = 2

        self.EMA_ALPHA = 0.3
        self.smoothed_ball_y = None

        self.FACE_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8]

        self.fps = 30.0
        self.frame_idx = 0

        self.shot_count = 0
        self.ball_controlled = False
        self.close_frames = 0
        self.far_frames   = 0

        self.prebend_recorded_this_cycle = False
        self.preelbow_recorded_this_cycle = False

        self.prebend_total   = 0
        self.preelbow_list   = []
        self.shot_release_records = []

        self.last_angles = {
            "shoulder": None,
            "elbow": None,
            "wrist": None,
            "knee_L": None,
            "knee_R": None,
            "knee_bent": None
        }

        self._shot_side = True
        self._release_snapshot = None
        self._prev_ball_is_far = False

    def set_callback(self, cb):
        self.callback = cb

    def set_fps(self, fps):
        try:
            self.fps = float(fps) if fps and fps > 0 else 30.0
        except Exception:
            self.fps = 30.0

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

    def _top_of_face_y(self, lm, H):
        ys = [
            lm[i].y
            for i in self.FACE_IDS
            if getattr(lm[i], 'visibility', 0) > 0.2 and 0 <= lm[i].y <= 1
        ]
        if not ys:
            return None
        return int(min(ys) * H)

    def _get_pose_angles(self, lm, W, H, ball_center, use_right=True):
        if use_right:
            shoulder = (lm[12].x * W, lm[12].y * H)
            elbow    = (lm[14].x * W, lm[14].y * H)
            wrist    = (lm[16].x * W, lm[16].y * H)
            hip      = (lm[24].x * W, lm[24].y * H)
        else:
            shoulder = (lm[11].x * W, lm[11].y * H)
            elbow    = (lm[13].x * W, lm[13].y * H)
            wrist    = (lm[15].x * W, lm[15].y * H)
            hip      = (lm[23].x * W, lm[23].y * H)

        hipL   = (lm[23].x * W, lm[23].y * H)
        kneeL  = (lm[25].x * W, lm[25].y * H)
        ankleL = (lm[27].x * W, lm[27].y * H)
        hipR   = (lm[24].x * W, lm[24].y * H)
        kneeR  = (lm[26].x * W, lm[26].y * H)
        ankleR = (lm[28].x * W, lm[28].y * H)

        shoulder_ang = angle_3pts(hip, shoulder, elbow)
        elbow_ang    = angle_3pts(shoulder, elbow, wrist)

        wrist_anchor = ball_center if ball_center is not None else (wrist[0], wrist[1] - 50)
        wrist_ang    = angle_3pts(elbow, wrist, wrist_anchor)

        kneeL_ang    = angle_3pts(hipL, kneeL, ankleL)
        kneeR_ang    = angle_3pts(hipR, kneeR, ankleR)

        return (
            round(shoulder_ang, 1),
            round(elbow_ang, 1),
            round(wrist_ang, 1),
            round(kneeL_ang, 1),
            round(kneeR_ang, 1)
        )

    def _detect_ball(self, frame_bgr):
        model = self.model_yolo or self.yolo
        if model is None:
            return None, None

        results = model(frame_bgr, verbose=False)
        names = getattr(model, "names", {})

        for box in results[0].boxes:
            cls_id = int(box.cls.detach().cpu().numpy())

            if isinstance(names, dict):
                cls_name = str(names.get(cls_id, "")).lower()
            elif isinstance(names, list) and 0 <= cls_id < len(names):
                cls_name = str(names[cls_id]).lower()
            else:
                cls_name = ""

            if cls_id == 0 or "ball" in cls_name or "basket" in cls_name:
                x1, y1, x2, y2 = box.xyxy.detach().cpu().numpy()[0].astype(int)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                if self.smoothed_ball_y is None:
                    self.smoothed_ball_y = cy
                else:
                    alpha = self.EMA_ALPHA
                    self.smoothed_ball_y = int(alpha * cy + (1 - alpha) * self.smoothed_ball_y)

                return (cx, self.smoothed_ball_y), (x1, y1, x2, y2)

        return None, None

    def process_frame(self, frame_bgr):
        H, W = frame_bgr.shape[:2]

        ball_center, ball_box = self._detect_ball(frame_bgr)

        pose_out = self.pose.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        shoulder = elbow = wrist = kneeL = kneeR = None
        knee_bent_now = None
        face_top_y = None
        dist_ball_to_hand = None
        hands_visible = False

        lm = None
        if pose_out.pose_landmarks:
            lm = pose_out.pose_landmarks.landmark

            sh, el, wr, kL, kR = self._get_pose_angles(
                lm, W, H, ball_center, use_right=self._shot_side
            )
            shoulder, elbow, wrist, kneeL, kneeR = sh, el, wr, kL, kR

            if kneeL is not None and kneeR is not None:
                knee_bent_now = (min(kneeL, kneeR) < self.KNEE_BENT_THRESHOLD_DEG)

            self.last_angles.update({
                "shoulder": shoulder,
                "elbow": elbow,
                "wrist": wrist,
                "knee_L": kneeL,
                "knee_R": kneeR,
                "knee_bent": knee_bent_now
            })

            face_top_y = self._top_of_face_y(lm, H)

            wrist_pts = []
            rw = lm[16]; lw = lm[15]
            if getattr(rw, 'visibility', 0) > 0.2 and 0 <= rw.x <= 1 and 0 <= rw.y <= 1:
                wrist_pts.append((int(rw.x * W), int(rw.y * H)))
            if getattr(lw, 'visibility', 0) > 0.2 and 0 <= lw.x <= 1 and 0 <= lw.y <= 1:
                wrist_pts.append((int(lw.x * W), int(lw.y * H)))

            if ball_center is not None and wrist_pts:
                hands_visible = True
                d_list = [math.hypot(ball_center[0] - wx, ball_center[1] - wy)
                          for (wx, wy) in wrist_pts]
                dist_ball_to_hand = min(d_list)

        near_thr_px = max(self.NEAR_MIN_PX, int(self.NEAR_FRACTION * min(W, H)))
        far_thr_px  = max(self.FAR_MIN_PX,  int(self.FAR_FRACTION  * min(W, H)))

        ball_is_close = (
            hands_visible and
            dist_ball_to_hand is not None and
            dist_ball_to_hand < near_thr_px
        )
        ball_is_far = (
            hands_visible and
            dist_ball_to_hand is not None and
            dist_ball_to_hand > far_thr_px
        )

        self.close_frames = self.close_frames + 1 if ball_is_close else 0
        self.far_frames   = self.far_frames   + 1 if ball_is_far   else 0

        if ball_is_far and (not self._prev_ball_is_far) and self._release_snapshot is None:
            self._release_snapshot = {
                "shoulder": self.last_angles["shoulder"],
                "elbow":    self.last_angles["elbow"],
                "wrist":    self.last_angles["wrist"],
                "knee_L":   self.last_angles["knee_L"],
                "knee_R":   self.last_angles["knee_R"],
                "knee_bent_at_release": self.last_angles["knee_bent"],
                "had_prebend":          self.prebend_recorded_this_cycle,
                "had_preelbow":         self.preelbow_recorded_this_cycle
            }
        self._prev_ball_is_far = ball_is_far

        if (not self.ball_controlled) and (self.close_frames >= self.CLOSE_REQ):
            self.ball_controlled = True
            self.prebend_recorded_this_cycle = False
            self.preelbow_recorded_this_cycle = False
            self._release_snapshot = None
            self._prev_ball_is_far = False

            if ball_center is not None and lm is not None:
                rw, lw = lm[16], lm[15]
                dR = dL = None
                if getattr(rw, 'visibility', 0) > 0.2:
                    dR = math.hypot(ball_center[0] - rw.x * W, ball_center[1] - rw.y * H)
                if getattr(lw, 'visibility', 0) > 0.2:
                    dL = math.hypot(ball_center[0] - lw.x * W, ball_center[1] - lw.y * H)
                if dR is not None and dL is not None:
                    self._shot_side = (dR <= dL)
                elif dR is not None:
                    self._shot_side = True
                elif dL is not None:
                    self._shot_side = False
                else:
                    self._shot_side = True

        if self.ball_controlled:
            if (not self.prebend_recorded_this_cycle) and (knee_bent_now is True):
                self.prebend_total += 1
                self.prebend_recorded_this_cycle = True

            if (not self.preelbow_recorded_this_cycle):
                if (ball_center is not None) and (face_top_y is not None) and (elbow is not None):
                    if ball_center[1] <= face_top_y:
                        self.preelbow_list.append(elbow)
                        self.preelbow_recorded_this_cycle = True

            if self.far_frames >= self.FAR_REQ:
                self.shot_count += 1

                rec = self._release_snapshot if self._release_snapshot is not None else {
                    "shoulder": self.last_angles["shoulder"],
                    "elbow":    self.last_angles["elbow"],
                    "wrist":    self.last_angles["wrist"],
                    "knee_L":   self.last_angles["knee_L"],
                    "knee_R":   self.last_angles["knee_R"],
                    "knee_bent_at_release": self.last_angles["knee_bent"],
                    "had_prebend":          self.prebend_recorded_this_cycle,
                    "had_preelbow":         self.preelbow_recorded_this_cycle
                }
                self.shot_release_records.append(rec)

                self.ball_controlled = False
                self._release_snapshot = None
                self._prev_ball_is_far = False

        snapshot = {
            "status": "ok",
            "frame_idx": int(self.frame_idx),
            "shots": int(self.shot_count),
            "ball_controlled": bool(self.ball_controlled),
            "ball_close_frames": int(self.close_frames),
            "ball_far_frames": int(self.far_frames),
            "near_threshold_px": int(near_thr_px),
            "far_threshold_px": int(far_thr_px),
            "dist_ball_to_hand_px": float(dist_ball_to_hand) if dist_ball_to_hand is not None else None,
            "knee_bent_now": bool(knee_bent_now) if knee_bent_now is not None else None,
            "face_top_y": int(face_top_y) if face_top_y is not None else None,
            "ball_center_y": int(ball_center[1]) if ball_center is not None else None,
        }

        self._async_emit(snapshot)

        self.frame_idx += 1
        return snapshot

    def final_summary(self):
        rel_shoulder = [r["shoulder"] for r in self.shot_release_records if r["shoulder"] is not None]
        rel_elbow    = [r["elbow"]    for r in self.shot_release_records if r["elbow"]    is not None]
        rel_wrist    = [r["wrist"]    for r in self.shot_release_records if r["wrist"]    is not None]
        rel_kL       = [r["knee_L"]   for r in self.shot_release_records if r["knee_L"]   is not None]
        rel_kR       = [r["knee_R"]   for r in self.shot_release_records if r["knee_R"]   is not None]
        rel_bent = [r["knee_bent_at_release"] for r in self.shot_release_records
                    if r["knee_bent_at_release"] is not None]

        def _avg(arr):
            return sum(arr) / len(arr) if arr else None

        avg_shoulder = _avg(rel_shoulder)
        avg_elbow = _avg(rel_elbow)
        avg_wrist = _avg(rel_wrist)
        avg_kL = _avg(rel_kL)
        avg_kR = _avg(rel_kR)

        if rel_bent:
            bend_rate = round(100.0 * (sum(1 for b in rel_bent if b) / len(rel_bent)), 1)
        else:
            bend_rate = None

        avg_preelbow = _avg(self.preelbow_list)

        summary = {
            "status": "final",
            "Total_shots": int(self.shot_count),
            "prebend_total": int(self.prebend_total),
            "preelbow_captured": int(len(self.preelbow_list)),
            "Avg_shoulder": (round(avg_shoulder, 1) if avg_shoulder is not None else None),
            "Avg_elbow":    (round(avg_elbow,    1) if avg_elbow    is not None else None),
            "Avg_wrist":    (round(avg_wrist,    1) if avg_wrist    is not None else None),
            "Avg_knee_L":   (round(avg_kL,       1) if avg_kL       is not None else None),
            "Avg_knee_R_":  (round(avg_kR,       1) if avg_kR       is not None else None),
            "Knee_bend_rate": (bend_rate if bend_rate is not None else None),
            "Set_elbow_avg":   (round(avg_preelbow, 1) if avg_preelbow is not None else None),
            "shots_detail": self.shot_release_records
        }

        self._async_emit(summary)
        return summary
