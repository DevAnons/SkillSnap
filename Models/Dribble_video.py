import time, math, collections
import cv2
import mediapipe as mp

class DribbleAnalyzer:

    def __init__(self):
        self.RESIZE_LONG_SIDE = 640
        self.YOLO_CONFIDENCE = 0.20
        self.YOLO_IMAGE_SIZE = 416

        self.HAND_PROXIMITY_PX = 40
        self.RETURN_DISTANCE_GAIN = 1.6
        self.RELEASE_DY_PX = 0.5
        self.MIN_DOWNWARD_V = 1
        self.GROUND_TOLERANCE_PX = 15
        self.STATE_TIMEOUT_SEC = 1.5
        self.KEEP_BALL_FOR_FRAMES = 12

        self.SMOOTH_Y_WINDOW = 3
        self.IN_HAND_HITS = 2
        self.NOT_IN_HAND_HITS = 2

        self.WAIST_BAND_PX = 6

        mp_pose = mp.solutions.pose
        self.mp_pose = mp_pose
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False
        )

        self.LMK = dict(
            LW=15, RW=16,
            LH=23, RH=24,
            LA=27, RA=28,
            LHHEEL=29, RHHEEL=30,
            LTOE=31, RTOE=32,
            LSH=11, RSH=12
        )

        self.model_yolo = None
        self.callback = None
        self._reset_runtime()

    def set_model(self, yolo_model):
        self.model_yolo = yolo_model

    def set_callback(self, cb):
        self.callback = cb

    def process_frame(self, frame_bgr):
        if self.model_yolo is None:
            return {"status": "error", "message": "ยังไม่ได้ตั้งค่าโมเดล YOLO"}

        proc, sx, sy = self._resize_long_side(frame_bgr, self.RESIZE_LONG_SIDE)
        h, w = proc.shape[:2]

        hand_thr   = max(30, int(self.HAND_PROXIMITY_PX * sy))
        return_thr = int(hand_thr * self.RETURN_DISTANCE_GAIN)
        rel_dy     = max(1, int(self.RELEASE_DY_PX * sy))
        min_down   = max(1, int(self.MIN_DOWNWARD_V * sy))
        ground_tol = max(3, int(self.GROUND_TOLERANCE_PX * sy))

        try:
            yout = self.model_yolo.predict(
                proc,
                imgsz=self.YOLO_IMAGE_SIZE,
                conf=self.YOLO_CONFIDENCE,
                verbose=False
            )
            box = self._pick_ball_box(
                yout,
                self.model_yolo.names,
                h,
                w,
                conf_min=self.YOLO_CONFIDENCE
            )
        except Exception:
            box = None

        if box is None:
            self.miss_frames += 1
            if self.last_box is None or self.miss_frames > self.KEEP_BALL_FOR_FRAMES:
                return self._result("no_ball")
            x1, y1, x2, y2 = self.last_box
        else:
            self.last_box = box
            self.miss_frames = 0
            x1, y1, x2, y2 = box

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        bottom = y2

        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        if not res.pose_landmarks:
            return self._result("no_pose")

        lm = res.pose_landmarks.landmark
        lwx, lwy = int(lm[self.LMK['LW']].x * w), int(lm[self.LMK['LW']].y * h)
        rwx, rwy = int(lm[self.LMK['RW']].x * w), int(lm[self.LMK['RW']].y * h)
        waist_y  = int(((lm[self.LMK['LH']].y + lm[self.LMK['RH']].y) / 2.0) * h)
        ground_y = self._estimate_ground_y(lm, h)

        tilt_deg = self._back_tilt_from_pose(lm, w, h)
        if tilt_deg is not None:
            self.back_tilt_samples.append(tilt_deg)

        dL = self._euclid(cx, cy, lwx, lwy)
        dR = self._euclid(cx, cy, rwx, rwy)
        near = min(dL, dR)

        if near <= hand_thr:
            self.in_streak += 1
            self.out_streak = 0
        else:
            self.out_streak += 1
            self.in_streak = 0

        if self.in_streak >= self.IN_HAND_HITS:
            self.in_hand = True
        if self.out_streak >= self.NOT_IN_HAND_HITS:
            self.in_hand = False

        self.cy_history.append(cy)
        cy_s = sum(self.cy_history) / len(self.cy_history)
        vy = None if self.prev_cy is None else (cy_s - self.prev_cy)
        self.prev_cy = cy_s

        now = time.time()
        if (now - self.last_state_time) > self.STATE_TIMEOUT_SEC and self.state != "WAIT_RELEASE":
            self.state = "WAIT_RELEASE"
            self.last_state_time = now

        if self.state == "WAIT_RELEASE":
            if vy is not None and (not self.in_hand) and vy > rel_dy:
                self.state = "WAIT_GROUND"
                self.last_state_time = now

        elif self.state == "WAIT_GROUND":
            if vy is not None and vy > min_down and bottom >= (ground_y - ground_tol):
                self.state = "WAIT_RETURN"
                self.last_state_time = now
            elif self.in_hand:
                self.state = "WAIT_RELEASE"
                self.last_state_time = now

        elif self.state == "WAIT_RETURN":
            if near <= return_thr:
                L_level = "High Dribble" if (lwy < (waist_y - self.WAIST_BAND_PX)) else "Low Dribble"
                R_level = "High Dribble" if (rwy < (waist_y - self.WAIST_BAND_PX)) else "Low Dribble"
                if dL < dR:
                    if L_level == "Low Dribble":
                        self.L_low += 1
                    else:
                        self.L_high += 1
                else:
                    if R_level == "Low Dribble":
                        self.R_low += 1
                    else:
                        self.R_high += 1
                self.state = "WAIT_RELEASE"
                self.last_state_time = now

        result = {
            "status": "ok",
            "state": self.state,
            "left_low":   int(self.L_low),
            "left_high":  int(self.L_high),
            "right_low":  int(self.R_low),
            "right_high": int(self.R_high),
            "back_tilt": self._avg_back_tilt(),
            "back_verdict": self._verdict_from_avg_tilt(self._avg_back_tilt())
        }

        if self.callback:
            try:
                import asyncio
                if asyncio.iscoroutinefunction(self.callback):
                    asyncio.create_task(self.callback(result))
                else:
                    self.callback(result)
            except Exception as e:
                print("[callback error]", e)

        return result

    def final_summary(self, status: str = "final"):
        avg_tilt = self._avg_back_tilt()
        return {
            "status": status,
            "state": self.state,
            "left_low":   int(self.L_low),
            "left_high":  int(self.L_high),
            "right_low":  int(self.R_low),
            "right_high": int(self.R_high),
            "back_tilt": avg_tilt,
            "back_verdict": self._verdict_from_avg_tilt(avg_tilt)
        }

    def _reset_runtime(self):
        self.L_low = self.L_high = self.R_low = self.R_high = 0
        self.state = "WAIT_RELEASE"
        self.last_state_time = time.time()
        self.last_box = None
        self.miss_frames = 0
        self.prev_cy = None
        self.cy_history = collections.deque(maxlen=self.SMOOTH_Y_WINDOW)
        self.in_streak = 0
        self.out_streak = 0
        self.in_hand = False
        self.back_tilt_samples = []

    def _resize_long_side(self, frame, long_side=640):
        h, w = frame.shape[:2]
        if max(h, w) == long_side:
            return frame, 1.0, 1.0
        scale = long_side / float(h if h >= w else w)
        nw, nh = int(w * scale), int(h * scale)
        return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR), scale, scale

    def _pick_ball_box(self, yolo_results, names, h, w, conf_min=0.2):
        if not yolo_results:
            return None
        boxes = yolo_results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None
        for b in boxes:
            if float(b.conf.cpu().numpy()) < conf_min:
                continue
            cls_id = int(b.cls.cpu().numpy())
            ok = True
            if isinstance(names, dict):
                cname = str(names.get(cls_id, "")).lower()
                ok = ("basketball" in cname) or ("ball" in cname) or (cname == "0")
            elif isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
                cname = str(names[cls_id]).lower()
                ok = ("basketball" in cname) or ("ball" in cname)
            if not ok:
                continue
            x1, y1, x2, y2 = b.xyxy.cpu().numpy()[0].astype(int)
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))
            if x2 > x1 and y2 > y1:
                return (x1, y1, x2, y2)
        return None

    @staticmethod
    def _euclid(ax, ay, bx, by):
        return math.hypot(ax - bx, ay - by)

    def _estimate_ground_y(self, lm, h):
        idxs = [
            self.LMK['LA'], self.LMK['RA'],
            self.LMK['LHHEEL'], self.LMK['RHHEEL'],
            self.LMK['LTOE'], self.LMK['RTOE']
        ]
        ys = []
        for i in idxs:
            p = lm[i]
            if getattr(p, 'visibility', 0) > 0.2:
                ys.append(p.y * h)
        return int(min(h - 1, max(ys))) if ys else int(0.85 * h)

    def _back_tilt_from_pose(self, lm, w, h):
        lsh, rsh = lm[self.LMK['LSH']], lm[self.LMK['RSH']]
        lhip, rhip = lm[self.LMK['LH']], lm[self.LMK['RH']]
        if not all(getattr(p, 'visibility', 0) > 0.2 for p in (lsh, rsh, lhip, rhip)):
            return None
        shx = (lsh.x + rsh.x) * 0.5 * w
        shy = (lsh.y + rsh.y) * 0.5 * h
        hpx = (lhip.x + rhip.x) * 0.5 * w
        hpy = (lhip.y + rhip.y) * 0.5 * h
        vx, vy = (shx - hpx), (shy - hpy)
        norm = math.hypot(vx, vy)
        if norm < 1e-6:
            return None
        dot = vy * -1
        cosang = max(-1.0, min(1.0, dot / norm))
        return math.degrees(math.acos(cosang))

    def _avg_back_tilt(self):
        return sum(self.back_tilt_samples) / len(self.back_tilt_samples) if self.back_tilt_samples else None

    @staticmethod
    def _verdict_from_avg_tilt(avg_tilt):
        if avg_tilt is None:
            return "n/a"
        if avg_tilt <= 10:
            return "Straight"
        if avg_tilt <= 20:
            return "Slight lean"
        return "Leaning"

    def _result(self, status):
        avg_tilt = self._avg_back_tilt()
        return {
            "status": status,
            "state": self.state,
            "left_low":   int(self.L_low),
            "left_high":  int(self.L_high),
            "right_low":  int(self.R_low),
            "right_high": int(self.R_high),
            "back_tilt": avg_tilt,
            "back_verdict": self._verdict_from_avg_tilt(avg_tilt)
        }
