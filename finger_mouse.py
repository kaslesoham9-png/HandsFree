"""
Finger Mouse engine  (index finger = cursor, head = clicks & scroll)

  * Move index finger       -> moves the cursor
  * Turn head LEFT / RIGHT  -> left click / right click
  * Head UP / DOWN          -> scroll up / down (hold = keep scrolling)

This file is the tracking ENGINE. It can be used two ways:
  1) GUI:  python finger_mouse_gui.py        (recommended)
  2) CLI:  python finger_mouse.py            (simple OpenCV window)

Install:
  pip install -r requirements.txt
"""

# --- Python 3.15+ & PyInstaller -------------------------------------------
# Python 3.15 added "lazy imports" (PEP 810) and its own standard library uses
# them (e.g. concurrent.futures). Inside a PyInstaller app those lazy objects
# don't resolve ("'lazy_import' object is not callable"), so force every import
# to be eager. Must run before anything else is imported. No-op on <= 3.14.
import sys
if hasattr(sys, "set_lazy_imports_filter"):
    sys.set_lazy_imports_filter(lambda *args: False)
# ---------------------------------------------------------------------------

import math
import os
import sys
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass, fields

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

pyautogui.PAUSE = 0
pyautogui.FAILSAFE = True   # slam cursor into a screen corner to abort


# =============================== CONFIG =====================================
@dataclass
class Config:
    # camera / cursor
    camera_index: int = 0
    cam_w: int = 640
    cam_h: int = 480
    frame_margin: int = 100        # px border; inner box maps to full screen
    smoothing: float = 0.25        # 0..1, lower = smoother but laggier
    cursor_hz: int = 120           # cursor updates / s (independent of camera FPS)

    # features
    enable_cursor: bool = True
    enable_clicks: bool = True
    enable_scroll: bool = True
    show_overlay: bool = True

    # clicks (head left / right)
    head_mode: str = "turn"        # "turn" (yaw) or "tilt" (roll)
    trigger_deg: float = 20.0
    return_deg: float = 10.0
    click_cooldown: float = 0.30

    # scroll (head up / down)
    pitch_trigger_deg: float = 12.0
    pitch_return_deg: float = 7.0
    scroll_min_rate: float = 4.0   # notches / s
    scroll_max_rate: float = 25.0
    scroll_ramp_deg: float = 15.0
    scroll_unit: int = 120 if sys.platform == "win32" else 1

    # general
    gesture_frames: int = 2
    calibration_frames: int = 30
    neutral_drift: float = 0.01

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        names = {f.name: f.type for f in fields(cls)}
        cfg = cls()
        for k, v in (d or {}).items():
            if k in names:
                try:
                    setattr(cfg, k, type(getattr(cfg, k))(v))
                except (TypeError, ValueError):
                    pass
        return cfg


# =============================== PATHS ======================================
APP_ID = "FingerMouse"
IS_FROZEN = getattr(sys, "frozen", False)          # True inside a PyInstaller build
# Where bundled read-only files live (PyInstaller unpacks them to sys._MEIPASS)
BUNDLE_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
# Folder containing the .exe (frozen) or this script (dev)
APP_DIR = (os.path.dirname(sys.executable) if IS_FROZEN
           else os.path.dirname(os.path.abspath(__file__)))
DIR = APP_DIR  # backwards compatibility


def user_data_dir():
    """Writable per-user folder for settings, logs and downloaded models."""
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
    elif sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    path = os.path.join(base, APP_ID)
    os.makedirs(path, exist_ok=True)
    return path


def resource_path(filename):
    """First existing copy of a resource: bundled -> next to app -> user data dir."""
    for folder in (BUNDLE_DIR, APP_DIR, user_data_dir()):
        candidate = os.path.join(folder, filename)
        if os.path.exists(candidate):
            return candidate
    return None


# =============================== MODELS =====================================
HAND_MODEL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
              "hand_landmarker/float16/1/hand_landmarker.task", "hand_landmarker.task")
FACE_MODEL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
              "face_landmarker/float16/1/face_landmarker.task", "face_landmarker.task")

INDEX_TIP, NOSE_TIP = 8, 1
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def ensure_models(log=print):
    """Return {filename: path}; downloads any model that isn't found locally."""
    paths = {}
    for url, filename in (HAND_MODEL, FACE_MODEL):
        path = resource_path(filename)
        if path is None:
            path = os.path.join(user_data_dir(), filename)
            log(f"Downloading {filename} ...")
            tmp = path + ".part"
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, path)
        paths[filename] = path
    return paths


def _model_options(path):
    # Load the model into memory: avoids native-path issues (non-ASCII user
    # names, PyInstaller temp folders) inside MediaPipe's C++ code.
    with open(path, "rb") as f:
        return mp_python.BaseOptions(model_asset_buffer=f.read())


# ============================ CONTROLLERS ===================================
class HeadClicker:
    """Signed horizontal head angle (negative = user's left) -> click events.
    Fires once past trigger, re-arms only after returning near centre."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.state = "center"
        self._side, self._count = None, 0
        self._last_click = -1e9

    def update(self, angle, now):
        c = self.cfg
        if self.state == "center":
            side = "left" if angle < -c.trigger_deg else "right" if angle > c.trigger_deg else None
            if side is not None and side == self._side:
                self._count += 1
            else:
                self._side, self._count = side, (1 if side else 0)
            if side and self._count >= c.gesture_frames and now - self._last_click >= c.click_cooldown:
                self.state, self._count, self._last_click = "turned", 0, now
                return side
        else:
            self._count = self._count + 1 if abs(angle) < c.return_deg else 0
            if self._count >= c.gesture_frames:
                self.state, self._side, self._count = "center", None, 0
        return None


class HeadScroller:
    """Head pitch (positive = up) -> continuous scrolling. update() returns
    wheel notches for this frame (+ up / - down); speed grows with tilt."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.direction = 0
        self._pending, self._count, self._acc = 0, 0, 0.0

    def stop(self):
        self.direction, self._pending, self._count, self._acc = 0, 0, 0, 0.0

    def update(self, pitch, dt, allowed=True):
        c = self.cfg
        if self.direction == 0:
            want = 1 if pitch > c.pitch_trigger_deg else -1 if pitch < -c.pitch_trigger_deg else 0
            if not allowed:
                want = 0
            if want and want == self._pending:
                self._count += 1
            else:
                self._pending, self._count = want, (1 if want else 0)
            if want and self._count >= c.gesture_frames:
                self.direction, self._count, self._acc = want, 0, 0.0
        else:
            self._count = self._count + 1 if pitch * self.direction < c.pitch_return_deg else 0
            if self._count >= c.gesture_frames:
                self.stop()

        if self.direction == 0:
            return 0
        excess = max(0.0, abs(pitch) - c.pitch_trigger_deg)
        rate = min(c.scroll_max_rate, c.scroll_min_rate + excess / max(c.scroll_ramp_deg, 1e-6)
                   * (c.scroll_max_rate - c.scroll_min_rate))
        self._acc += rate * min(dt, 0.1)
        notches = int(self._acc)
        self._acc -= notches
        return notches * self.direction


def head_pose(matrix, mode="turn"):
    """(horizontal, pitch) in degrees from the face pose matrix, on the MIRRORED
    image. horizontal < 0 = user's LEFT; pitch > 0 = head UP."""
    m = np.array(matrix)[:3, :3]
    if mode == "tilt":
        horizontal = -math.degrees(math.atan2(m[1, 0], m[0, 0]))
    else:
        horizontal = math.degrees(math.atan2(-m[2, 0], math.hypot(m[2, 1], m[2, 2])))
    pitch = math.degrees(math.asin(float(np.clip(m[1, 2], -1.0, 1.0))))
    return horizontal, pitch



# ============================ PERFORMANCE ===================================
# Windows 11 puts minimised / background apps into "Efficiency mode" (EcoQoS):
# slower CPU cores, lower clocks and coarse timers -> the camera loop slows down
# and the cursor gets laggy. macOS does the same with "App Nap". Opt out of both.
_APP_NAP_ACTIVITY = None
_TIMER_PERIOD_SET = False


class _ThrottlingState:  # filled in lazily (ctypes only on Windows)
    cls = None


def _win32():
    """kernel32 / winmm with full argtypes. Every HANDLE is declared explicitly:
    the pseudo-handle -1 returned by GetCurrentProcess/Thread otherwise turns
    into 0xFFFFFFFFFFFFFFFF and overflows a plain C int."""
    import ctypes
    from ctypes import wintypes
    if _ThrottlingState.cls is None:
        class _State(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG),
                        ("ControlMask", wintypes.ULONG),
                        ("StateMask", wintypes.ULONG)]
        _ThrottlingState.cls = _State
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    H, BOOL, DWORD = wintypes.HANDLE, wintypes.BOOL, wintypes.DWORD
    k32.GetCurrentProcess.argtypes, k32.GetCurrentProcess.restype = [], H
    k32.GetCurrentThread.argtypes, k32.GetCurrentThread.restype = [], H
    k32.SetProcessInformation.argtypes = [H, ctypes.c_int, ctypes.c_void_p, DWORD]
    k32.SetProcessInformation.restype = BOOL
    k32.SetThreadInformation.argtypes = [H, ctypes.c_int, ctypes.c_void_p, DWORD]
    k32.SetThreadInformation.restype = BOOL
    k32.SetPriorityClass.argtypes, k32.SetPriorityClass.restype = [H, DWORD], BOOL
    k32.SetThreadPriority.argtypes, k32.SetThreadPriority.restype = [H, ctypes.c_int], BOOL
    winmm = ctypes.WinDLL("winmm")
    winmm.timeBeginPeriod.argtypes, winmm.timeBeginPeriod.restype = [ctypes.c_uint], ctypes.c_uint
    winmm.timeEndPeriod.argtypes, winmm.timeEndPeriod.restype = [ctypes.c_uint], ctypes.c_uint
    return ctypes, k32, winmm, _ThrottlingState.cls


def disable_background_throttling(log=print):
    global _APP_NAP_ACTIVITY, _TIMER_PERIOD_SET
    if sys.platform == "win32":
        try:
            ctypes, k32, winmm, State = _win32()
            proc = k32.GetCurrentProcess()
            ok = False
            # EXECUTION_SPEED (0x1) | IGNORE_TIMER_RESOLUTION (0x4, Win11 only);
            # StateMask = 0 means "never throttle me".
            for mask in (0x1 | 0x4, 0x1):
                st = State(1, mask, 0)
                if k32.SetProcessInformation(proc, 4, ctypes.byref(st), ctypes.sizeof(st)):
                    ok = True
                    break
            prio = k32.SetPriorityClass(proc, 0x00008000)   # ABOVE_NORMAL_PRIORITY_CLASS
            if not _TIMER_PERIOD_SET:
                winmm.timeBeginPeriod(1)                     # 1 ms timer precision
                _TIMER_PERIOD_SET = True
            log(("Background throttling disabled (Efficiency mode off)" if ok
                 else "Could not disable Efficiency mode on this Windows version")
                + ("" if prio else " | priority unchanged"))
        except Exception as e:  # noqa: BLE001
            log(f"Performance tweak failed: {e}")
    elif sys.platform == "darwin" and _APP_NAP_ACTIVITY is None:
        try:
            from Foundation import NSProcessInfo
            # NSActivityUserInitiated | NSActivityLatencyCritical
            _APP_NAP_ACTIVITY = NSProcessInfo.processInfo().beginActivityWithOptions_reason_(
                0x00FFFFFF | 0xFF00000000, "Camera mouse control")
            log("App Nap disabled")
        except Exception as e:  # noqa: BLE001
            log(f"Could not disable App Nap: {e}")


def restore_timer_resolution():
    global _TIMER_PERIOD_SET
    if sys.platform == "win32" and _TIMER_PERIOD_SET:
        try:
            _win32()[2].timeEndPeriod(1)
        except Exception:  # noqa: BLE001
            pass
        _TIMER_PERIOD_SET = False


def boost_current_thread(level=1):
    """Raise the calling thread's priority and exempt it from EcoQoS (Windows)."""
    if sys.platform != "win32":
        return
    try:
        ctypes, k32, _, State = _win32()
        thread = k32.GetCurrentThread()
        st = State(1, 0x1, 0)                     # THREAD_POWER_THROTTLING_EXECUTION_SPEED off
        k32.SetThreadInformation(thread, 3, ctypes.byref(st), ctypes.sizeof(st))
        k32.SetThreadPriority(thread, level)      # 1 = ABOVE_NORMAL, 2 = HIGHEST
    except Exception:  # noqa: BLE001
        pass


class CursorMover(threading.Thread):
    """Moves the cursor at a fixed high rate (default 120 Hz), gliding toward the
    latest finger position. Keeps the cursor smooth even if the camera / AI
    only runs at 15-30 FPS (e.g. when the window is minimised)."""

    TARGET_TIMEOUT = 0.25   # s - stop moving if no fresh finger position arrives

    def __init__(self, cfg, start_xy):
        super().__init__(daemon=True, name="CursorMover")
        self.cfg = cfg
        self.error = None
        self._lock = threading.Lock()
        self._target = None
        self._target_time = 0.0
        self._cur = [float(start_xy[0]), float(start_xy[1])]
        self._running = True

    def set_target(self, x, y):
        with self._lock:
            self._target = (float(x), float(y))
            self._target_time = time.monotonic()

    def clear(self):
        with self._lock:
            self._target = None

    def stop(self):
        self._running = False

    def run(self):
        boost_current_thread(2)
        last = time.monotonic()
        last_sent = None
        try:
            while self._running:
                now = time.monotonic()
                dt, last = now - last, now
                with self._lock:
                    target, t_time = self._target, self._target_time
                if target is not None and now - t_time < self.TARGET_TIMEOUT:
                    # Same "feel" as the old per-frame smoothing at 30 FPS,
                    # but applied continuously at cursor_hz.
                    s = min(max(self.cfg.smoothing, 0.0), 1.0)
                    alpha = 1.0 - (1.0 - s) ** (dt * 30.0)
                    self._cur[0] += (target[0] - self._cur[0]) * alpha
                    self._cur[1] += (target[1] - self._cur[1]) * alpha
                    pos = (int(round(self._cur[0])), int(round(self._cur[1])))
                    if pos != last_sent:
                        pyautogui.moveTo(*pos)
                        last_sent = pos
                time.sleep(1.0 / max(self.cfg.cursor_hz, 10))
        except Exception as e:  # noqa: BLE001 - e.g. pyautogui.FailSafeException
            self.error = e


# =============================== ENGINE =====================================
class FingerMouseEngine:
    """Feed it raw camera frames with process(); it moves/clicks/scrolls the
    mouse and returns (annotated_frame, state_dict)."""

    def __init__(self, cfg: Config = None, on_event=None):
        self.cfg = cfg or Config()
        self.on_event = on_event or (lambda msg: None)
        self.hands = self.face = None
        self.paused = False
        self.screen_w, self.screen_h = pyautogui.size()
        self.mover = None
        self._last_ts = 0
        self._prev_time = None
        self._flash_text, self._flash_until = "", 0.0
        self.recalibrate()

    # ---- lifecycle ----
    def start(self):
        disable_background_throttling(self.on_event)
        boost_current_thread(1)
        models = ensure_models(self.on_event)
        self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=_model_options(models[HAND_MODEL[1]]),
            running_mode=vision.RunningMode.VIDEO, num_hands=1,
            min_hand_detection_confidence=0.7, min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6))
        self.face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=_model_options(models[FACE_MODEL[1]]),
            running_mode=vision.RunningMode.VIDEO, num_faces=1,
            min_face_detection_confidence=0.5, min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5, output_facial_transformation_matrixes=True))
        self.mover = CursorMover(self.cfg, pyautogui.position())
        self.mover.start()

    def close(self):
        if self.mover is not None:
            self.mover.stop()
            self.mover.join(timeout=1)
            self.mover = None
        restore_timer_resolution()
        for lm in (self.hands, self.face):
            if lm is not None:
                lm.close()
        self.hands = self.face = None

    # ---- controls ----
    def recalibrate(self):
        self.clicker = HeadClicker(self.cfg)
        self.scroller = HeadScroller(self.cfg)
        self._calib_h, self._calib_p = [], []
        self.neutral_h = self.neutral_p = None
        self._last_mode = self.cfg.head_mode

    def set_paused(self, paused):
        self.paused = paused
        self.scroller.stop()
        if self.mover is not None:
            self.mover.clear()

    def _flash(self, text, now):
        self._flash_text, self._flash_until = text, now + 0.5
        self.on_event(text.title())

    # ---- main per-frame step ----
    def process(self, frame_bgr):
        c = self.cfg
        if self.mover is not None and self.mover.error is not None:
            raise self.mover.error            # e.g. fail-safe triggered in cursor thread
        now = time.time()
        dt = 0.0 if self._prev_time is None else now - self._prev_time
        self._prev_time = now
        if c.head_mode != self._last_mode:   # mode change invalidates calibration
            self.recalibrate()
            self.on_event(f"Head mode changed to '{c.head_mode}' - recalibrating")

        frame = cv2.flip(frame_bgr, 1)
        fh, fw = frame.shape[:2]
        margin = int(min(c.frame_margin, fw // 2 - 10, fh // 2 - 10))
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ts = max(int(time.monotonic() * 1000), self._last_ts + 1)
        self._last_ts = ts
        hand_res = self.hands.detect_for_video(mp_img, ts)
        face_res = self.face.detect_for_video(mp_img, ts)

        state = dict(hand=False, face=False, calibrated=self.neutral_h is not None,
                     calib_progress=0.0, h_angle=0.0, p_angle=0.0,
                     click_armed=True, scroll_dir=0, paused=self.paused)

        if c.show_overlay:
            cv2.rectangle(frame, (margin, margin), (fw - margin, fh - margin), (255, 0, 0), 2)

        # ---------- cursor ----------
        cursor_target = None
        if hand_res.hand_landmarks:
            state["hand"] = True
            lm = hand_res.hand_landmarks[0]
            if c.show_overlay:
                self._draw_hand(frame, lm)
            if not self.paused and c.enable_cursor:
                ix, iy = lm[INDEX_TIP].x * fw, lm[INDEX_TIP].y * fh
                tx = np.interp(ix, (margin, fw - margin), (0, self.screen_w - 1))
                ty = np.interp(iy, (margin, fh - margin), (0, self.screen_h - 1))
                cursor_target = (tx, ty)
        if self.mover is not None:
            if cursor_target is not None:
                self.mover.set_target(*cursor_target)   # cursor thread glides there
            else:
                self.mover.clear()

        # ---------- head ----------
        if face_res.face_landmarks and face_res.facial_transformation_matrixes:
            state["face"] = True
            if c.show_overlay:
                nose = face_res.face_landmarks[0][NOSE_TIP]
                cv2.circle(frame, (int(nose.x * fw), int(nose.y * fh)), 5, (0, 255, 255), -1)
            raw_h, raw_p = head_pose(face_res.facial_transformation_matrixes[0], c.head_mode)

            if self.neutral_h is None:
                self._calib_h.append(raw_h)
                self._calib_p.append(raw_p)
                state["calib_progress"] = len(self._calib_h) / c.calibration_frames
                if len(self._calib_h) >= c.calibration_frames:
                    self.neutral_h = float(np.median(self._calib_h))
                    self.neutral_p = float(np.median(self._calib_p))
                    state["calibrated"] = True
                    self.on_event("Calibrated - head clicks & scroll ready")
            else:
                h_ang, p_ang = raw_h - self.neutral_h, raw_p - self.neutral_p
                if (self.clicker.state == "center" and self.scroller.direction == 0
                        and abs(h_ang) < c.return_deg and abs(p_ang) < c.pitch_return_deg):
                    self.neutral_h += (raw_h - self.neutral_h) * c.neutral_drift
                    self.neutral_p += (raw_p - self.neutral_p) * c.neutral_drift

                if not self.paused:
                    horiz_dominant = abs(h_ang) >= abs(p_ang)
                    if c.enable_scroll:
                        notches = self.scroller.update(
                            p_ang, dt, allowed=(self.clicker.state == "center" and not horiz_dominant))
                        if notches:
                            pyautogui.scroll(int(notches * c.scroll_unit))
                    else:
                        self.scroller.stop()

                    if self.scroller.direction == 0:
                        event = self.clicker.update(h_ang if horiz_dominant else 0.0, now)
                        if event and c.enable_clicks:
                            if event == "left":
                                pyautogui.click()
                                self._flash("LEFT CLICK", now)
                            else:
                                pyautogui.rightClick()
                                self._flash("RIGHT CLICK", now)

                state.update(h_angle=h_ang, p_angle=p_ang,
                             click_armed=self.clicker.state == "center",
                             scroll_dir=self.scroller.direction)
        else:
            self.scroller.stop()

        # ---------- status ----------
        if self.paused:
            status = "PAUSED"
        elif not state["calibrated"]:
            status = "CALIBRATING - LOOK STRAIGHT" if state["face"] else "NO FACE DETECTED"
        elif self.scroller.direction > 0:
            status = "SCROLLING UP"
        elif self.scroller.direction < 0:
            status = "SCROLLING DOWN"
        elif now < self._flash_until:
            status = self._flash_text
        elif self.clicker.state == "turned":
            status = "RETURN HEAD TO CENTRE"
        else:
            status = "ACTIVE"
        state["status"] = status
        return frame, state

    @staticmethod
    def _draw_hand(frame, landmarks):
        h, w = frame.shape[:2]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 200, 0), 2)
        for p in pts:
            cv2.circle(frame, p, 3, (255, 255, 255), -1)
        cv2.circle(frame, pts[INDEX_TIP], 10, (255, 0, 255), -1)


# ============================== SIMPLE CLI ==================================
def main():
    cfg = Config()
    engine = FingerMouseEngine(cfg, on_event=print)
    engine.start()
    cap = cv2.VideoCapture(cfg.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.cam_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.cam_h)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")
    print("Running. c = recalibrate, p = pause, q = quit")
    prev = time.time()
    while True:
        ok, raw = cap.read()
        if not ok:
            break
        frame, st = engine.process(raw)
        now = time.time()
        fps, prev = 1.0 / max(now - prev, 1e-6), now
        cv2.putText(frame, f"{st['status']}  FPS {fps:.0f}  H {st['h_angle']:+.0f}  "
                           f"P {st['p_angle']:+.0f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Finger Mouse", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("p"):
            engine.set_paused(not engine.paused)
        if key == ord("c"):
            engine.recalibrate()
    cap.release()
    cv2.destroyAllWindows()
    engine.close()


if __name__ == "__main__":
    main()
