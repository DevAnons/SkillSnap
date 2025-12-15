import math
import statistics
from dataclasses import dataclass, asdict
import asyncio
import mediapipe as mp

POSE_VIS_THR = 0.5
CALIB_FRAMES = 40
MARGIN_PCT = 0.018

KNEE_BEND_THR = 175.0
PRE_BEND_TIME_SEC = 2.0
FOOT_SPAN_RANGE = (0.8, 1.4)

BEND_SCORE_THR = 175.0
BEND_SCORE_MAX = 5
BEND_HOLD_TIME_SEC = 0.30

TARGET_REPS = 5

mp_pose = mp.solutions.pose


def angle_deg(a, b, c):
    bax, bay = a[0] - b[0], a[1] - b[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    dot = bax * bcx + bay * bcy
    na = math.hypot(bax, bay) + 1e-6
    nc = math.hypot(bcx, bcy) + 1e-6
    cosv = max(-1.0, min(1.0, dot / (na * nc)))
    return math.degrees(math.acos(cosv))


def is_vis_ok(*vals, thr=POSE_VIS_THR):
    return all(v is not None and v >= thr for v in vals)


def to_xyv(lm, w, h):
    return (lm.x * w, lm.y * h, lm.visibility)


@dataclass
class Rep:
    bend: bool
    arms: bool
    feet: bool
    time_s: float
    peak_air_gap_px: int
    knee_angle_at_takeoff: float
    pre_bend_avg: float


class ReboundAnalyzer:
    def __init__(self):
        self.pose = mp_pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.callback = None
        self.fps = 30.0
        self.frame_idx = 0

        self.reps = []
        self.in_air = False

        self.floor_y = None
        self.floor_samples = []

        self.knee_min_hist = []
        self.arms_up_in_air = False
        self.feet_ok_at_bend = False

        self.landed_cooldown = 0
        self.takeoff_frame = None
        self.max_air_gap_px = 0
        self.knee_at_takeoff = None

        self.bend_points = 0
        self.bend_gate_open = True
        self.bend_hold_count = 0
        self.arm_points = 0

        self._pre_bend_frames = max(1, int(PRE_BEND_TIME_SEC * self.fps))
        self._bend_hold_frames = max(1, int(BEND_HOLD_TIME_SEC * self.fps))

        self._bend_ok_cache = False

    def reset_session(self):
        self.__init__()

    def set_callback(self, cb):
        self.callback = cb

    def set_fps(self, fps: float):
        try:
            self.fps = float(fps) if fps and fps > 0 else 30.0
        except Exception:
            self.fps = 30.0
        self._pre_bend_frames = max(1, int(PRE_BEND_TIME_SEC * self.fps))
        self._bend_hold_frames = max(1, int(BEND_HOLD_TIME_SEC * self.fps))

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

    def process_frame(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        margin = max(6, int(MARGIN_PCT * h))

        result = self.pose.process(frame_bgr[:, :, ::-1])
        knee_min_now = None

        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark

            def L(i):
                return to_xyv(lm[i], w, h)

            L_sh = L(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
            R_sh = L(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
            L_hip = L(mp_pose.PoseLandmark.LEFT_HIP.value)
            R_hip = L(mp_pose.PoseLandmark.RIGHT_HIP.value)
            L_knee = L(mp_pose.PoseLandmark.LEFT_KNEE.value)
            R_knee = L(mp_pose.PoseLandmark.RIGHT_KNEE.value)
            L_ank = L(mp_pose.PoseLandmark.LEFT_ANKLE.value)
            R_ank = L(mp_pose.PoseLandmark.RIGHT_ANKLE.value)
            L_heel = L(mp_pose.PoseLandmark.LEFT_HEEL.value)
            R_heel = L(mp_pose.PoseLandmark.RIGHT_HEEL.value)
            L_wrist = L(mp_pose.PoseLandmark.LEFT_WRIST.value)
            R_wrist = L(mp_pose.PoseLandmark.RIGHT_WRIST.value)
            nose = L(mp_pose.PoseLandmark.NOSE.value)

            if is_vis_ok(
                L_sh[2], R_sh[2], L_hip[2], R_hip[2],
                L_knee[2], R_knee[2], L_ank[2], R_ank[2],
                L_heel[2], R_heel[2], L_wrist[2], R_wrist[2], nose[2]
            ):
                knee_L = angle_deg((L_hip[0], L_hip[1]), (L_knee[0], L_knee[1]), (L_ank[0], L_ank[1]))
                knee_R = angle_deg((R_hip[0], R_hip[1]), (R_knee[0], R_knee[1]), (R_ank[0], R_ank[1]))
                knee_min_now = min(knee_L, knee_R)
                self.knee_min_hist.append(knee_min_now)

                shoulder_w = abs(R_sh[0] - L_sh[0]) + 1e-6
                feet_span_ratio = abs(R_ank[0] - L_ank[0]) / shoulder_w
                feet_span_ok_now = FOOT_SPAN_RANGE[0] <= feet_span_ratio <= FOOT_SPAN_RANGE[1]

                foot_bottom_y = max(L_ank[1], R_ank[1], L_heel[1], R_heel[1])
                if self.frame_idx < CALIB_FRAMES:
                    self.floor_samples.append(foot_bottom_y)
                    if len(self.floor_samples) >= CALIB_FRAMES:
                        tail = self.floor_samples[int(0.5 * CALIB_FRAMES):]
                        self.floor_y = statistics.median(tail if tail else self.floor_samples)
                elif self.floor_y is not None:
                    self.floor_y = 0.98 * self.floor_y + 0.02 * foot_bottom_y

                arms_up_now = (L_wrist[1] < nose[1] and R_wrist[1] < nose[1])

                if self.floor_y is not None:
                    left_contact_y = min(L_ank[1], L_heel[1])
                    right_contact_y = min(R_ank[1], R_heel[1])

                    feet_air = (
                        left_contact_y < self.floor_y - margin and
                        right_contact_y < self.floor_y - margin
                    )
                    feet_ground = (
                        left_contact_y > self.floor_y - margin / 2 and
                        right_contact_y > self.floor_y - margin / 2
                    )

                    if (not self.in_air) and self.bend_gate_open and knee_min_now is not None \
                       and self.bend_points < BEND_SCORE_MAX:
                        if knee_min_now < BEND_SCORE_THR:
                            self.bend_hold_count += 1
                            if self.bend_hold_count >= self._bend_hold_frames:
                                self.bend_points += 1
                                self.bend_gate_open = False
                                self.bend_hold_count = 0
                        else:
                            self.bend_hold_count = 0
                    else:
                        if self.in_air:
                            self.bend_hold_count = 0

                    if (not self.in_air) and feet_air and self.landed_cooldown == 0:
                        self.in_air = True
                        self.bend_hold_count = 0

                        if len(self.knee_min_hist) > self._pre_bend_frames:
                            recent_window = self.knee_min_hist[-self._pre_bend_frames:]
                        else:
                            recent_window = self.knee_min_hist[:]

                        pre_bend_avg = sum(recent_window) / len(recent_window) if recent_window else 180.0

                        self._bend_ok_cache = pre_bend_avg < KNEE_BEND_THR
                        self.feet_ok_at_bend = feet_span_ok_now
                        self.arms_up_in_air = False
                        self.takeoff_frame = self.frame_idx
                        self.max_air_gap_px = 0
                        self.knee_at_takeoff = knee_min_now

                    if self.in_air and self.floor_y is not None:
                        air_gap_now = max(self.floor_y - left_contact_y,
                                          self.floor_y - right_contact_y)
                        self.max_air_gap_px = max(self.max_air_gap_px, int(air_gap_now))

                    if self.in_air and arms_up_now:
                        self.arms_up_in_air = True

                    if self.in_air and feet_ground:
                        self.in_air = False
                        self.landed_cooldown = 5
                        time_s = (self.takeoff_frame or self.frame_idx) / (self.fps if self.fps > 0 else 30.0)

                        if len(self.knee_min_hist) > self._pre_bend_frames:
                            recent_window = self.knee_min_hist[-self._pre_bend_frames:]
                        else:
                            recent_window = self.knee_min_hist[:]
                        pre_bend_avg_now = sum(recent_window) / len(recent_window) if recent_window else 180.0

                        rep = Rep(
                            bend=bool(self._bend_ok_cache),
                            arms=bool(self.arms_up_in_air),
                            feet=bool(self.feet_ok_at_bend),
                            time_s=float(time_s),
                            peak_air_gap_px=int(self.max_air_gap_px),
                            knee_angle_at_takeoff=float(self.knee_at_takeoff or 0.0),
                            pre_bend_avg=float(pre_bend_avg_now)
                        )
                        self.reps.append(rep)

                        if rep.arms and self.arm_points < 5:
                            self.arm_points += 1

                        self.bend_gate_open = True
                        self.bend_hold_count = 0

        if self.landed_cooldown > 0:
            self.landed_cooldown -= 1

        frame_status = {
            "status": "ok",
            "frame_idx": int(self.frame_idx),
            "time_s": self.frame_idx / (self.fps if self.fps > 0 else 30.0),
            "total_reps": int(len(self.reps)),
            "bend_points": int(self.bend_points),
            "arm_points": int(self.arm_points),
            "in_air": bool(self.in_air),
            "floor_y": float(self.floor_y) if self.floor_y is not None else None,
            "last_knee_min": float(knee_min_now) if knee_min_now is not None else None,
        }

        self._async_emit(frame_status)
        self.frame_idx += 1
        return frame_status

    def final_summary(self):
        total_reps = len(self.reps)
        passed = (
            total_reps >= TARGET_REPS and
            self.bend_points >= BEND_SCORE_MAX and
            self.arm_points >= 5
        )

        suggestions = []
        if passed:
            suggestions.append("✅ Great job! Excellent knee bend and arm swing consistency.")
        else:
            if total_reps < TARGET_REPS:
                suggestions.append("- Increase total jumps to at least 5 clear takeoff/land cycles.")
            if self.bend_points < BEND_SCORE_MAX:
                suggestions.append(f"- Bend deeper and hold ~{BEND_HOLD_TIME_SEC:.2f}s before takeoff.")
                suggestions.append("- Tip: push hips slightly back, keep weight midfoot-to-heel.")
            if self.arm_points < 5:
                suggestions.append("- Swing your arms upward past nose level during airtime every jump.")
                suggestions.append("- Tip: start the arm swing while you're bending down.")
            suggestions.append("- Keep your feet roughly shoulder-width at the load phase.")

        return {
            "status": "final",
            "total_reps": int(total_reps),
            "bend_points": int(self.bend_points),
            "arm_points": int(self.arm_points),
            "passed": bool(passed),
            "reps_detail": [asdict(r) for r in self.reps],
            "suggestions": suggestions
        }
