import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import math, asyncio
from dataclasses import dataclass
import mediapipe as mp

ARM_RATIO_GOOD = 2.2
ARM_RATIO_OK = 1.8
ELBOW_STRAIGHT_DEG = 150

KNEE_BEND_GOOD_MIN = 90
KNEE_BEND_GOOD_MAX = 125
KNEE_TOO_STRAIGHT = 160

POSE_VIS_THR = 0.5
PASS_THR = 80.0

mp_pose = mp.solutions.pose


def angle_at(beg, mid, end):
    bax, bay = beg[0] - mid[0], beg[1] - mid[1]
    bcx, bcy = end[0] - mid[0], end[1] - mid[1]
    dot = bax * bcx + bay * bcy
    na, nc = math.hypot(bax, bay) + 1e-6, math.hypot(bcx, bcy) + 1e-6
    cosv = max(-1.0, min(1.0, dot / (na * nc)))
    return float(math.degrees(math.acos(cosv)))


def to_xyv(lm, w, h):
    return (lm.x * w, lm.y * h, lm.visibility)


def has_good_visibility(*vis_values, thr=POSE_VIS_THR):
    return all(v is not None and v >= thr for v in vis_values)


def score_arms(wrist_span_ratio, elbow_L_deg, elbow_R_deg):
    if wrist_span_ratio >= ARM_RATIO_GOOD:
        span_score = 100.0
    elif wrist_span_ratio >= ARM_RATIO_OK:
        span_score = 60.0 + 40.0 * (wrist_span_ratio - ARM_RATIO_OK) / (ARM_RATIO_GOOD - ARM_RATIO_OK)
    else:
        span_score = max(0.0, 60.0 * (wrist_span_ratio / max(1e-6, ARM_RATIO_OK)))

    def elbow_component(e):
        if e >= ELBOW_STRAIGHT_DEG:
            return 100.0
        if e >= 120.0:
            return 40.0 + 60.0 * (e - 120.0) / (ELBOW_STRAIGHT_DEG - 120.0)
        return max(0.0, 40.0 * (e / 120.0))

    elbow_score = (elbow_component(elbow_L_deg) + elbow_component(elbow_R_deg)) / 2.0
    return 0.6 * span_score + 0.4 * elbow_score


def score_knees(knee_L_deg, knee_R_deg):
    k_avg = (knee_L_deg + knee_R_deg) / 2.0
    if KNEE_BEND_GOOD_MIN <= k_avg <= KNEE_BEND_GOOD_MAX:
        return 100.0
    if k_avg >= KNEE_TOO_STRAIGHT:
        return 0.0
    if k_avg < KNEE_BEND_GOOD_MIN:
        return max(20.0, 100.0 - (KNEE_BEND_GOOD_MIN - k_avg) * 2.0)
    return max(20.0, 100.0 - (k_avg - KNEE_BEND_GOOD_MAX) * 2.0)


@dataclass
class FrameStats:
    wrist_span_ratio: float = 0.0
    elbow_L: float = 0.0
    elbow_R: float = 0.0
    knee_L: float = 180.0
    knee_R: float = 180.0
    arm_score: float = 0.0
    knee_score: float = 0.0
    arms_out_pass: bool = False
    knees_bent_pass: bool = False


class DefenseAnalyzer:
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.frames_valid = 0
        self.arm_scores = []
        self.knee_scores = []
        self.pass_arm_frames = 0
        self.pass_knee_frames = 0
        self.pass_both_frames = 0

        self.fps = 30.0
        self.callback = None

    def set_callback(self, cb):
        self.callback = cb

    def set_fps(self, fps: float):
        try:
            self.fps = float(fps) if fps and fps > 0 else 30.0
        except Exception:
            self.fps = 30.0

    def process_frame(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        res = self.pose.process(frame_bgr[:, :, ::-1])

        stats = FrameStats()

        if res.pose_landmarks:
            lmk = res.pose_landmarks.landmark

            def pick(index):
                return to_xyv(lmk[index], w, h)

            L_sh_x, L_sh_y, L_sh_vis = pick(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
            R_sh_x, R_sh_y, R_sh_vis = pick(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
            L_el_x, L_el_y, L_el_vis = pick(mp_pose.PoseLandmark.LEFT_ELBOW.value)
            R_el_x, R_el_y, R_el_vis = pick(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
            L_wr_x, L_wr_y, L_wr_vis = pick(mp_pose.PoseLandmark.LEFT_WRIST.value)
            R_wr_x, R_wr_y, R_wr_vis = pick(mp_pose.PoseLandmark.RIGHT_WRIST.value)
            L_hip_x, L_hip_y, L_hip_vis = pick(mp_pose.PoseLandmark.LEFT_HIP.value)
            R_hip_x, R_hip_y, R_hip_vis = pick(mp_pose.PoseLandmark.RIGHT_HIP.value)
            L_kn_x, L_kn_y, L_kn_vis = pick(mp_pose.PoseLandmark.LEFT_KNEE.value)
            R_kn_x, R_kn_y, R_kn_vis = pick(mp_pose.PoseLandmark.RIGHT_KNEE.value)
            L_an_x, L_an_y, L_an_vis = pick(mp_pose.PoseLandmark.LEFT_ANKLE.value)
            R_an_x, R_an_y, R_an_vis = pick(mp_pose.PoseLandmark.RIGHT_ANKLE.value)

            if has_good_visibility(
                L_sh_vis, R_sh_vis,
                L_el_vis, R_el_vis, L_wr_vis, R_wr_vis,
                L_hip_vis, R_hip_vis, L_kn_vis, R_kn_vis,
                L_an_vis, R_an_vis
            ):
                self.frames_valid += 1

                shoulder_width_px = abs(R_sh_x - L_sh_x)
                wrist_span_px = abs(R_wr_x - L_wr_x)
                stats.wrist_span_ratio = float(wrist_span_px / max(1e-6, shoulder_width_px))

                stats.elbow_L = angle_at((L_sh_x, L_sh_y), (L_el_x, L_el_y), (L_wr_x, L_wr_y))
                stats.elbow_R = angle_at((R_sh_x, R_sh_y), (R_el_x, R_el_y), (R_wr_x, R_wr_y))

                stats.arm_score = float(score_arms(stats.wrist_span_ratio, stats.elbow_L, stats.elbow_R))
                stats.arms_out_pass = (
                    stats.wrist_span_ratio >= ARM_RATIO_OK and
                    stats.elbow_L >= ELBOW_STRAIGHT_DEG and
                    stats.elbow_R >= ELBOW_STRAIGHT_DEG
                )
                if stats.arms_out_pass:
                    self.pass_arm_frames += 1

                stats.knee_L = angle_at((L_hip_x, L_hip_y), (L_kn_x, L_kn_y), (L_an_x, L_an_y))
                stats.knee_R = angle_at((R_hip_x, R_hip_y), (R_kn_x, R_kn_y), (R_an_x, R_an_y))

                stats.knee_score = float(score_knees(stats.knee_L, stats.knee_R))
                stats.knees_bent_pass = (
                    KNEE_BEND_GOOD_MIN <= (stats.knee_L + stats.knee_R) / 2.0 <= KNEE_BEND_GOOD_MAX
                )
                if stats.knees_bent_pass:
                    self.pass_knee_frames += 1
                if stats.arms_out_pass and stats.knees_bent_pass:
                    self.pass_both_frames += 1

                self.arm_scores.append(stats.arm_score)
                self.knee_scores.append(stats.knee_score)

        arm_avg = float(sum(self.arm_scores) / len(self.arm_scores)) if self.arm_scores else 0.0
        knee_avg = float(sum(self.knee_scores) / len(self.knee_scores)) if self.knee_scores else 0.0
        overall_score = float(0.5 * arm_avg + 0.5 * knee_avg) if (self.arm_scores and self.knee_scores) else 0.0

        arm_pass_ratio = float(self.pass_arm_frames / self.frames_valid) if self.frames_valid else 0.0
        knee_pass_ratio = float(self.pass_knee_frames / self.frames_valid) if self.frames_valid else 0.0
        both_pass_ratio = float(self.pass_both_frames / self.frames_valid) if self.frames_valid else 0.0

        passed_training = arm_avg >= PASS_THR and knee_avg >= PASS_THR and overall_score >= PASS_THR

        tips = []
        if passed_training:
            tips.append("✅ Great job! You maintained a solid defensive stance.")
        else:
            if arm_avg < PASS_THR:
                tips.append("- Spread your arms wider and keep them firm. Avoid bending the elbows.")
            else:
                tips.append("- Your arm position looks good—keep them wide and steady.")
            if knee_avg < PASS_THR:
                tips.append("- Lower your stance slightly and bend your knees more for balance and quick moves.")
            else:
                tips.append("- Good stance! Maintain your knee bend to stay agile.")

        result = {
            "status": "ok",
            "frames_valid": int(self.frames_valid),
            "arm_avg": round(arm_avg, 2),
            "knee_avg": round(knee_avg, 2),
            "overall_score": round(overall_score, 2),
            "arm_pass_ratio": round(arm_pass_ratio, 3),
            "knee_pass_ratio": round(knee_pass_ratio, 3),
            "both_pass_ratio": round(both_pass_ratio, 3),
            "passed_training": passed_training,
            "tips": tips
        }

        if self.callback:
            try:
                if asyncio.iscoroutinefunction(self.callback):
                    asyncio.create_task(self.callback(result))
                else:
                    self.callback(result)
            except Exception as e:
                print("[callback error]", e)

        return result

    def final_summary(self):
        arm_avg = float(sum(self.arm_scores) / len(self.arm_scores)) if self.arm_scores else 0.0
        knee_avg = float(sum(self.knee_scores) / len(self.knee_scores)) if self.knee_scores else 0.0
        overall_score = float(0.5 * arm_avg + 0.5 * knee_avg) if (self.arm_scores and self.knee_scores) else 0.0

        arm_pass_ratio = float(self.pass_arm_frames / self.frames_valid) if self.frames_valid else 0.0
        knee_pass_ratio = float(self.pass_knee_frames / self.frames_valid) if self.frames_valid else 0.0
        both_pass_ratio = float(self.pass_both_frames / self.frames_valid) if self.frames_valid else 0.0

        passed_training = arm_avg >= PASS_THR and knee_avg >= PASS_THR and overall_score >= PASS_THR

        tips = []
        if passed_training:
            tips.append("✅ Great job! You maintained a solid defensive stance.")
        else:
            if arm_avg < PASS_THR:
                tips.append("- Spread your arms wider and keep them firm. Avoid bending the elbows.")
            else:
                tips.append("- Your arm position looks good—keep them wide and steady.")
            if knee_avg < PASS_THR:
                tips.append("- Lower your stance slightly and bend your knees more for balance and quick moves.")
            else:
                tips.append("- Good stance! Maintain your knee bend to stay agile.")

        return {
            "status": "final",
            "frames_valid": int(self.frames_valid),
            "arm_avg": round(arm_avg, 2),
            "knee_avg": round(knee_avg, 2),
            "overall_score": round(overall_score, 2),
            "arm_pass_ratio": round(arm_pass_ratio, 3),
            "knee_pass_ratio": round(knee_pass_ratio, 3),
            "both_pass_ratio": round(both_pass_ratio, 3),
            "passed_training": passed_training,
            "tips": tips
        }
