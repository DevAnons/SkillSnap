import os, asyncio, tempfile, time, base64, numbers, sys

from datetime import datetime
from typing import Dict, Optional, Any

import cv2, numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO

from Dribble_video import DribbleAnalyzer
from Passed_video import PassAnalyzer
from Defense_video import DefenseAnalyzer
from Shooted_video import ShootingAnalyzer
from Rebound_video import ReboundAnalyzer

YOLO_MODEL_PATH = r"C:\Users\1234n\Desktop\Project\basketballAI\Models\Models\BasketballDetect.pt"
SHARED_YOLO: Optional[YOLO] = None

# ---------------- Logger helpers ----------------
def _ts():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]

def log(*args):
    print(f"[{_ts()}]", *args, file=sys.stdout, flush=True)

def log_err(*args):
    print(f"[{_ts()}][ERR]", *args, file=sys.stderr, flush=True)

# ---------------- JSON sanitizer ----------------
LARGE_ARRAY_KEYS = {"img", "image", "frame", "mask", "debug_image", "overlay", "vis"}

def to_jsonable(obj, *, max_array_items=1000):
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        try:
            return obj.tolist() if obj.size <= max_array_items else f"<ndarray shape={obj.shape} dtype={obj.dtype}>"
        except Exception:
            return "<ndarray>"
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            k = k.decode("utf-8", "ignore") if isinstance(k, bytes) else str(k)
            if k in LARGE_ARRAY_KEYS:
                continue
            out[k] = to_jsonable(v, max_array_items=max_array_items)
        return out
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v, max_array_items=max_array_items) for v in obj]
    if isinstance(obj, bytes):
        return f"<bytes len={len(obj)}>"
    if isinstance(obj, numbers.Number):
        try:
            return obj + 0
        except Exception:
            pass
    return str(obj)

# ---------------- YOLO Loader ----------------
def get_shared_yolo() -> Optional[YOLO]:
    global SHARED_YOLO
    if SHARED_YOLO is None:
        try:
            SHARED_YOLO = YOLO(YOLO_MODEL_PATH)
            log(f"[YOLO] Loaded once from: {YOLO_MODEL_PATH}")
        except Exception as e:
            log_err(f"[YOLO] Failed to load model: {e}")
            return None
    return SHARED_YOLO

# ---------------- FastAPI Init ----------------
app = FastAPI(title="SkillSnap – Analyzer API (final unified)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ---------------- WebSocket Manager ----------------
class ConnectionManager:
    def __init__(self):
        self.active: Dict[str, WebSocket] = {}
        self.meta: Dict[str, dict] = {}
        self.status: Dict[str, str] = {}

    async def connect(self, ws: WebSocket, sid: str):
        await ws.accept()
        self.active[sid] = ws
        self.status[sid] = "connected"
        log(f"[WS connected] sid={sid} from={ws.client}")

    async def disconnect(self, sid: str):
        ws = self.active.pop(sid, None)
        self.status[sid] = "inactive"
        if ws:
            try:
                await ws.close()
            except Exception:
                pass
        log(f"[WS disconnected] sid='{sid}'")

    def mark_inactive(self, sid: str):
        if sid in self.active:
            self.active.pop(sid, None)
        self.status[sid] = "inactive"
        log(f"[WS inactive] sid={sid}")

    def alive(self, sid: str) -> bool:
        return sid in self.active

    def is_connected(self, sid: str) -> bool:
        return self.status.get(sid) == "connected"

    def has_meta(self, sid: str) -> bool:
        return bool(self.meta.get(sid))

    async def send(self, sid: str, data: dict) -> bool:
        payload = to_jsonable(data)
        try:
            # ✅ แก้ logger ให้แสดง class/done/total ถูกต้อง
            cls_name = payload.get('class') or payload.get('class_name')
            done = payload.get('done')
            total = payload.get('total') or payload.get('frames')
            log(f"[WS→client] sid={sid} status={payload.get('status')} cls={cls_name} done={done} total={total}")
        except Exception:
            pass

        ws = self.active.get(sid)
        if not ws:
            log(f"[WS send skipped] sid={sid} (no active socket)")
            return False
        try:
            # ส่งเป็น JSON โดยตรง
            await ws.send_json(payload)
            return True
        except Exception as e:
            log_err(f"[WS send error {sid}] {e}")
            self.mark_inactive(sid)
            return False

    def set_meta(self, sid: str, **kv):
        m = self.meta.get(sid, {})
        m.update(kv)
        self.meta[sid] = m
        log(f"[WS meta] sid={sid} -> {m}")

    def get_meta(self, sid: str) -> dict:
        return self.meta.get(sid, {})

manager = ConnectionManager()

# ---------------- Analyzer Factory ----------------
def make_analyzer(name: str):
    mapping = {
        "dribble": DribbleAnalyzer,
        "pass": PassAnalyzer,
        "defenses": DefenseAnalyzer,
        "shooting": ShootingAnalyzer,
        "rebound": ReboundAnalyzer,
    }
    ctor = mapping.get(name)
    return ctor() if ctor else None

def inject_yolo_if_needed(an, name: str) -> bool:
    if name in ("dribble", "pass", "shooting"):
        yolo = get_shared_yolo()
        if yolo is None:
            return False
        setattr(an, "model_yolo", yolo)
    return True

# ---------------- Core Video Processing ----------------
RESULTS = {}
async def _process_video_file(session_id: str, class_name: str, temp_path: str):
    analyzer = None
    try:
        # ✅ ไม่ abort ถ้า WS ยังไม่ต่อ: ให้ประมวลผลและเก็บผลไว้สำหรับ REST
        if not manager.alive(session_id):
            log("[WARN] WS is not alive; will process and store result for REST.")

        analyzer = make_analyzer(class_name)
        if analyzer is None:
            log_err(f"[PROC error] sid={session_id} unknown class={class_name}")
            await manager.send(session_id, {"status": "error", "message": f"Unknown class: {class_name}"})
            return

        if not inject_yolo_if_needed(analyzer, class_name):
            log_err(f"[PROC error] sid={session_id} YOLO model failed to load")
            await manager.send(session_id, {"status": "error", "message": "YOLO model failed to load"})
            return

        # ✅ บังคับใช้ FFMPEG เพื่อให้ read() เดินครบเฟรม
        cap = cv2.VideoCapture(temp_path, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            log_err(f"[PROC error] sid={session_id} cannot open video: {temp_path}")
            await manager.send(session_id, {"status": "error", "message": "Cannot open video"})
            return

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        log(f"[PROC start] sid={session_id} class={class_name} frames={total} fps={fps:.2f}")
        await manager.send(session_id, {
            "status": "video_started",
            "class": class_name,
            "frames": total,
            "fps": fps,
            "note": "Processing... please wait"
        })

        last_result = None
        last_emit = 0.0
        idx = 0
        first_ok = False

        while True:
            ok, frame = cap.read()
            if not ok:
                if idx == 0:
                    log_err(f"[READ fail] sid={session_id} at frame=0 (codec/backend?) path={temp_path}")
                break
            idx += 1

            if not first_ok:
                h, w = frame.shape[:2]
                log(f"[FIRST frame OK] sid={session_id} size={w}x{h}")
                first_ok = True

            if hasattr(analyzer, "process_frame"):
                try:
                    r = analyzer.process_frame(frame)
                    if isinstance(r, dict):
                        last_result = to_jsonable(r)
                except Exception as e:
                    log_err(f"[process_frame error @frame {idx}] {e}")

            now = time.time()
            if now - last_emit > 0.5:
                await manager.send(session_id, {"status": "progress", "done": idx, "total": total})
                last_emit = now

        cap.release()

        final_payload = {
            "status": "final",
            "class": class_name,
            "frames_processed": idx,
        }

        if hasattr(analyzer, "final_summary"):
            try:
                summary = analyzer.final_summary()
                log(f"[PROC final_summary] sid={session_id} class={class_name} summary={summary}")

                if summary:
                    final_payload["summary"] = to_jsonable(summary)
            except Exception as e:
                log_err(f"[final_summary error] {e}")

        if "summary" not in final_payload and last_result:
            log(f"[PROC final uses last_result] sid={session_id}")
            final_payload["summary"] = to_jsonable(last_result)

        # ส่งทาง WS (ถ้ามี) + เก็บผลไว้สำหรับ REST
        await manager.send(session_id, final_payload)
        await manager.send(session_id, {"status": "video_finished"})
        RESULTS[session_id] = final_payload
        log(f"[PROC end] sid={session_id} class={class_name} frames_processed={idx} final_payload={final_payload}")

    except Exception as e:
        log_err(f"[PROC fatal] sid={session_id} {e}")
        await manager.send(session_id, {"status": "error", "message": f"video processing: {e}"})

    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                log_err(f"[WARN] cannot remove temp file: {e}")

        if analyzer and hasattr(analyzer, "pose"):
            try:
                analyzer.pose.close()
            except Exception:
                pass

# ---------------- Helpers ----------------
def _strip_data_prefix(b64: str) -> str:
    return b64.split(",", 1)[1] if b64.startswith("data:") else b64

def _ext_from_mime(m: str) -> str:
    m = (m or "").lower().strip()
    if m == "video/quicktime": return ".mov"
    if m == "video/mp4": return ".mp4"
    if m == "application/octet-stream": return ".mp4"
    return ".mp4"

VALID_CLASSES = {"dribble","pass","defenses","shooting","rebound"}

def _try_b64_to_text(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    try:
        return base64.b64decode(s).decode("utf-8", "ignore")
    except Exception as e:
        log_err(f"[WS meta] decode file_name_b64 error: {e}")
        return None

# ---------------- API Endpoints ----------------
@app.websocket("/ws/{session_id}")
async def ws_endpoint(ws: WebSocket, session_id: str):
    await manager.connect(ws, session_id)
    try:
        while True:
            msg = await ws.receive_text()
            log(f"[WS recv] sid={session_id} text={msg[:120]!r}")
            await manager.send(session_id, {"status": "info", "message": "WS alive"})
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(session_id)

import re
SAFE_NAME_RE = re.compile(r'[^A-Za-z0-9._-]+')

def sanitize_filename(name: str, fallback: str = "upload.mp4") -> str:
    if not name:
        return fallback
    name = os.path.basename(name).strip()
    name = SAFE_NAME_RE.sub('_', name)
    return name or fallback

import re
SAFE_NAME_RE = re.compile(r'[^A-Za-z0-9._-]+')

def sanitize_filename(name: str, fallback: str = "upload.mp4") -> str:
    if not name:
        return fallback
    name = os.path.basename(name).strip()
    name = SAFE_NAME_RE.sub('_', name)
    return name or fallback

# --------------- Debug helper (NEW) ---------------
def debug_filename_info(label: str, name: Optional[str]):
    try:
        if not name:
            log(f"[DEBUG] {label}: <empty or None>")
            return
        shown = (name[:80] + '...') if len(name) > 80 else name
        has_ext = bool(os.path.splitext(name)[1])
        has_path_chars = ('/' in name) or ('\\' in name)
        log(f"[DEBUG] {label}: raw='{shown}' len={len(name)} has_ext={has_ext} has_path_chars={has_path_chars}")
    except Exception as e:
        log_err(f"[DEBUG] {label}: error while inspecting filename: {e}")

@app.post("/upload/{session_id}")
async def upload_video_base64(
    session_id: str,
    payload: Dict[str, Any] = Body(...)
):
    # --- อ่านค่าจาก JSON ---
    file_b64   = payload.get("FileBase64")or ""
    class_name = (payload.get("ClassName") or "").strip()
    file_name  = (payload.get("file_name")  or "").strip() or "video.mp4"
    mime_type  = (payload.get("mime_type")  or "").strip() or "video/mp4"

    log(f"[UPLOAD base64] sid={session_id} class={class_name!r} file={file_name!r} mime={mime_type!r}")

    # --- ตรวจสอบ input ---
    if not file_b64 or not class_name:
        return JSONResponse(status_code=400, content={
            "status": "error",
            "message": "Missing file_base64 or class_name"
        })

    if class_name not in VALID_CLASSES:
        return JSONResponse(status_code=400, content={
            "status": "error",
            "message": f"Invalid class_name: {class_name}"
        })

    # แปลง Base64 to ไฟล์ mp4
    def strip_prefix(file_b64: str) -> str:
        return file_b64.split(",", 1)[1] if file_b64.startswith("data:") else file_b64

    try:
        raw = base64.b64decode(strip_prefix(file_b64), validate=False)
    except Exception as e:
        return JSONResponse(status_code=400, content={
            "status": "error",
            "message": f"Invalid base64: {e}"
        })

    # --- บันทึกไฟล์ชั่วคราว ---
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        temp_path = tmp.name
    with open(temp_path, "wb") as f:
        f.write(raw)

    log(f"[UPLOAD base64] sid={session_id} saved={temp_path} bytes={len(raw)}")

    # --- เริ่มวิเคราะห์วิดีโอตาม class ---
    asyncio.create_task(_process_video_file(session_id, class_name, temp_path))
    return {"status": "accepted", "session_id": session_id, "class": class_name, "via": "base64"}

@app.get("/health")
async def health():
    return {"status": "healthy", "time": datetime.now().isoformat()}

@app.get("/result/{session_id}")
async def get_result(session_id: str):
    if session_id in RESULTS:
        return RESULTS[session_id]
    else:
        return {"status": "not_found", "message": f"No result for session_id={session_id}"}
@app.get("/summary")
async def summary():
    return {}

# ---------------- Run via ngrok ----------------
if __name__ == "__main__":
    import uvicorn
    import nest_asyncio
    from pyngrok import ngrok
    import os

    # 🔐 ดึง token จาก Environment Variable
    NGROK_AUTH = os.getenv("NGROK_AUTHTOKEN")

    if not NGROK_AUTH:
        raise RuntimeError("NGROK_AUTHTOKEN is not set")

    try:
        ngrok.set_auth_token(NGROK_AUTH)
        tunnel = ngrok.connect(8001, "http")
        log("Public URL:", tunnel.public_url)
        log("WS URL:", tunnel.public_url.replace("http", "ws") + "/ws/<session_id>")
    except Exception as e:
        log_err("[ngrok] error:", e)

    nest_asyncio.apply()
    _ = get_shared_yolo()
    uvicorn.run(app, host="0.0.0.0", port=8001)