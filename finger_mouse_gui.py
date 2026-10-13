"""
Finger Mouse - desktop frontend (PySide6 / Qt)

Run:
    pip install -r requirements.txt 
    python finger_mouse_gui.py

Build a standalone app:   see build.bat / build.sh  (PyInstaller)

Optional (system-wide hotkeys that work even when this window is not focused):
    pip install pynput
    Ctrl+Alt+P = pause / resume      Ctrl+Alt+C = recalibrate

In-window shortcuts:  F5 start/stop   Space pause   C recalibrate
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

import json
import os
import shutil
import sys
import time
import traceback

# PyInstaller --windowed builds have no console: sys.stdout / sys.stderr are
# None, and any library that writes to them would crash the app.
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

import cv2

# Some OpenCV builds (Linux) ship their own Qt plugins and point Qt at them via
# environment variables, which breaks PySide6. Undo that before Qt loads.
for _var in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
    if "cv2" in os.environ.get(_var, ""):
        os.environ.pop(_var, None)

import pyautogui
from PySide6.QtCore import QEvent, QObject, QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import (QColor, QFont, QImage, QKeySequence, QPainter, QPen,
                           QPixmap, QShortcut)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QFrame,
                               QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                               QMainWindow, QMessageBox, QPlainTextEdit,
                               QProgressBar, QPushButton, QScrollArea,
                               QSizePolicy, QSlider, QSpinBox, QTabWidget,
                               QVBoxLayout, QWidget)

import finger_mouse as fm

APP_NAME = "Finger Mouse"
DATA_DIR = fm.user_data_dir()       # e.g. %APPDATA%\\FingerMouse on Windows
SETTINGS_PATH = os.path.join(DATA_DIR, "settings.json")
LOG_PATH = os.path.join(DATA_DIR, "finger_mouse.log")

# one-time migration of settings saved by older versions next to the script
_OLD_SETTINGS = os.path.join(fm.APP_DIR, "finger_mouse_settings.json")
if not os.path.exists(SETTINGS_PATH) and os.path.exists(_OLD_SETTINGS):
    try:
        shutil.copy(_OLD_SETTINGS, SETTINGS_PATH)
    except OSError:
        pass

# ------------------------------- theme -------------------------------------
ACCENT = "#7c6cff"
STATUS_COLORS = {
    "ACTIVE": "#2ecc71", "PAUSED": "#f1c40f", "LEFT CLICK": "#3498db",
    "RIGHT CLICK": "#9b59b6", "SCROLLING UP": "#1abc9c", "SCROLLING DOWN": "#e67e22",
    "RETURN HEAD TO CENTRE": "#95a5a6", "CALIBRATING - LOOK STRAIGHT": "#e67e22",
    "NO FACE DETECTED": "#e74c3c", "STOPPED": "#636e72", "STARTING...": "#7c6cff",
}

STYLESHEET = f"""
QWidget {{ background: #15161c; color: #e6e6ef; font-size: 13px; }}
QGroupBox {{ border: 1px solid #2a2c38; border-radius: 10px; margin-top: 14px;
             padding: 10px 8px 8px 8px; background: #1b1c24; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px;
                    color: #a9a9c4; font-weight: 600; }}
QGroupBox QWidget {{ background: transparent; }}
QPushButton {{ background: #2a2c38; border: 1px solid #353848; border-radius: 8px;
               padding: 8px 14px; font-weight: 600; }}
QPushButton:hover {{ background: #343747; }}
QPushButton:disabled {{ color: #5d6070; background: #202129; }}
QPushButton#primary {{ background: {ACCENT}; border: none; color: white; }}
QPushButton#primary:hover {{ background: #9184ff; }}
QPushButton#danger {{ background: #e74c3c; border: none; color: white; }}
QSlider::groove:horizontal {{ height: 6px; background: #2a2c38; border-radius: 3px; }}
QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 3px; }}
QSlider::handle:horizontal {{ background: white; width: 16px; margin: -6px 0; border-radius: 8px; }}
QTabWidget::pane {{ border: 1px solid #2a2c38; border-radius: 8px; top: -1px; background: #1b1c24; }}
QTabBar::tab {{ background: #1f2029; padding: 7px 12px; border-top-left-radius: 7px;
                border-top-right-radius: 7px; margin-right: 2px; color: #a9a9c4; }}
QTabBar::tab:selected {{ background: #2a2c38; color: white; }}
QPlainTextEdit {{ background: #101116; border: 1px solid #2a2c38; border-radius: 8px;
                  font-family: Consolas, Menlo, monospace; font-size: 12px; }}
QComboBox, QSpinBox {{ background: #2a2c38; border: 1px solid #353848; border-radius: 6px; padding: 4px 8px; }}
QProgressBar {{ background: #2a2c38; border: none; border-radius: 4px; height: 8px; text-align: center; }}
QProgressBar::chunk {{ background: #e67e22; border-radius: 4px; }}
QCheckBox {{ spacing: 8px; padding: 3px 0; }}
QCheckBox::indicator {{ width: 18px; height: 18px; border-radius: 5px; border: 1px solid #4a4d60; background: #2a2c38; }}
QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}
QScrollArea {{ border: none; }}
QToolTip {{ background: #2a2c38; color: white; border: 1px solid #4a4d60; padding: 6px; }}
"""


# ------------------------------- settings ----------------------------------
def load_config():
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            return fm.Config.from_dict(json.load(f))
    except (OSError, ValueError):
        return fm.Config()


def save_config(cfg):
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)


# ------------------------------- worker ------------------------------------
class TrackerWorker(QThread):
    """Runs camera capture + the tracking engine off the GUI thread."""
    frame_ready = Signal(QImage)
    state_ready = Signal(dict)
    event = Signal(str)
    error = Signal(str)
    running = Signal()

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._running = False
        self._pause_req = None
        self._calib_req = False
        self.send_preview = True     # False while the window is minimised

    def request_pause(self, paused):
        self._pause_req = paused

    def request_calibrate(self):
        self._calib_req = True

    def stop(self):
        self._running = False

    def run(self):
        self._running = True
        cap = engine = None
        try:
            self.event.emit("Loading hand & face models...")
            engine = fm.FingerMouseEngine(self.cfg, on_event=self.event.emit)
            engine.start()
            self.event.emit(f"Opening camera {self.cfg.camera_index}...")
            cap = cv2.VideoCapture(self.cfg.camera_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.cam_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.cam_h)
            if not cap.isOpened():
                raise RuntimeError(f"Could not open camera {self.cfg.camera_index}. "
                                   "Try another camera index in the Cursor tab.")
            self.running.emit()
            self.event.emit("Tracking started - look straight at the screen to calibrate")
            prev, fps = time.time(), 0.0
            while self._running:
                if self._pause_req is not None:
                    engine.set_paused(self._pause_req)
                    self.event.emit("Paused" if self._pause_req else "Resumed")
                    self._pause_req = None
                if self._calib_req:
                    engine.recalibrate()
                    self._calib_req = False
                    self.event.emit("Recalibrating - look straight at the screen")

                ok, raw = cap.read()
                if not ok:
                    raise RuntimeError("Camera stopped delivering frames.")
                frame, state = engine.process(raw)

                now = time.time()
                inst = 1.0 / max(now - prev, 1e-6)
                prev = now
                fps = inst if fps == 0 else fps * 0.9 + inst * 0.1
                state["fps"] = fps

                # Skip the (costly) preview conversion when nobody can see it
                if self.send_preview:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, _ = rgb.shape
                    self.frame_ready.emit(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy())
                self.state_ready.emit(state)
        except pyautogui.FailSafeException:
            self.error.emit("Fail-safe triggered (cursor hit a screen corner). Tracking stopped.")
        except Exception as e:  # noqa: BLE001 - show any failure to the user
            self.error.emit(f"{type(e).__name__}: {e}")
        finally:
            if cap is not None:
                cap.release()
            if engine is not None:
                engine.close()


# ------------------------------- widgets -----------------------------------
class VideoView(QLabel):
    """Camera preview that scales to fit while keeping aspect ratio."""

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setStyleSheet("background:#0d0e12; border:1px solid #2a2c38; border-radius:12px;"
                           "color:#5d6070; font-size:16px;")
        self._pix = None
        self.show_placeholder()

    def show_placeholder(self, text="Camera is off\n\nPress  ▶ Start  (F5)"):
        self._pix = None
        self.setPixmap(QPixmap())
        self.setText(text)

    def set_image(self, img: QImage):
        self._pix = QPixmap.fromImage(img)
        self._rescale()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._rescale()

    def _rescale(self):
        if self._pix is not None:
            self.setPixmap(self._pix.scaled(max(self.width() - 4, 1), max(self.height() - 4, 1),
                                            Qt.KeepAspectRatio, Qt.SmoothTransformation))


class HeadPad(QWidget):
    """2-D map of head position: x = left/right (clicks), y = up/down (scroll)."""
    MAX_H, MAX_P = 40.0, 30.0

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.h = self.p = 0.0
        self.active = False
        self.armed = True
        self.scroll_dir = 0
        self.setMinimumSize(260, 200)

    def update_state(self, st):
        self.active = st.get("face", False) and st.get("calibrated", False)
        self.h, self.p = st.get("h_angle", 0.0), st.get("p_angle", 0.0)
        self.armed, self.scroll_dir = st.get("click_armed", True), st.get("scroll_dir", 0)
        self.update()

    def reset(self):
        self.active = False
        self.update()

    def paintEvent(self, _):
        c = self.cfg
        qp = QPainter(self)
        qp.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(4, 4, -4, -4)
        qp.setPen(QPen(QColor("#2a2c38"), 1))
        qp.setBrush(QColor("#111218"))
        qp.drawRoundedRect(r, 10, 10)

        cx, cy = r.center().x(), r.center().y()
        sx, sy = (r.width() / 2) / self.MAX_H, (r.height() / 2) / self.MAX_P
        X = lambda d: cx + max(-self.MAX_H, min(self.MAX_H, d)) * sx
        Y = lambda d: cy - max(-self.MAX_P, min(self.MAX_P, d)) * sy

        # trigger zones
        zones = [
            (QRectF(r.left(), r.top(), X(-c.trigger_deg) - r.left(), r.height()), "#3498db", "LEFT\nCLICK"),
            (QRectF(X(c.trigger_deg), r.top(), r.right() - X(c.trigger_deg), r.height()), "#9b59b6", "RIGHT\nCLICK"),
            (QRectF(X(-c.trigger_deg), r.top(), X(c.trigger_deg) - X(-c.trigger_deg),
                    Y(c.pitch_trigger_deg) - r.top()), "#1abc9c", "SCROLL ▲"),
            (QRectF(X(-c.trigger_deg), Y(-c.pitch_trigger_deg), X(c.trigger_deg) - X(-c.trigger_deg),
                    r.bottom() - Y(-c.pitch_trigger_deg)), "#e67e22", "SCROLL ▼"),
        ]
        qp.setFont(QFont(self.font().family(), 8, QFont.Bold))
        for rect, col, label in zones:
            fill = QColor(col)
            fill.setAlpha(40)
            qp.setPen(Qt.NoPen)
            qp.setBrush(fill)
            qp.drawRect(rect)
            txt = QColor(col)
            txt.setAlpha(200)
            qp.setPen(txt)
            qp.drawText(rect, Qt.AlignCenter, label)

        # return (neutral) box + crosshair
        qp.setPen(QPen(QColor("#f1f1f1"), 1, Qt.DashLine))
        qp.setBrush(Qt.NoBrush)
        qp.drawRect(QRectF(QPointF(X(-c.return_deg), Y(c.pitch_return_deg)),
                           QPointF(X(c.return_deg), Y(-c.pitch_return_deg))))
        qp.setPen(QPen(QColor("#3a3d4e"), 1))
        qp.drawLine(QPointF(cx, r.top()), QPointF(cx, r.bottom()))
        qp.drawLine(QPointF(r.left(), cy), QPointF(r.right(), cy))

        if not self.active:
            qp.setPen(QColor("#7f8296"))
            qp.setFont(QFont(self.font().family(), 10))
            qp.drawText(r, Qt.AlignCenter, "\n\n\nwaiting for face / calibration")
            return

        if self.scroll_dir:
            col = "#1abc9c" if self.scroll_dir > 0 else "#e67e22"
        else:
            col = "#2ecc71" if self.armed else "#f39c12"
        qp.setPen(QPen(QColor("white"), 2))
        qp.setBrush(QColor(col))
        qp.drawEllipse(QPointF(X(self.h), Y(self.p)), 9, 9)
        qp.setPen(QColor("#c9c9d9"))
        qp.setFont(QFont(self.font().family(), 9))
        qp.drawText(r.adjusted(8, 4, -8, -4), Qt.AlignBottom | Qt.AlignRight,
                    f"H {round(self.h) or 0:+d}°   V {round(self.p) or 0:+d}°")


class Pill(QLabel):
    """Small coloured status indicator."""

    def __init__(self, text):
        super().__init__()
        self.base = text
        self.set_on(False)

    def set_on(self, on, text=None):
        col = "#2ecc71" if on else "#e74c3c"
        self.setText(f"<span style='color:{col}; font-size:15px'>●</span>&nbsp; {text or self.base}")
        self.setStyleSheet("background:#1b1c24; border:1px solid #2a2c38; border-radius:12px;"
                           "padding:4px 10px;")


class SliderRow(QWidget):
    """Label + slider + value; writes straight into the config (live)."""

    def __init__(self, cfg, key, label, lo, hi, step, fmt="{:.0f}", tooltip=""):
        super().__init__()
        self.cfg, self.key, self.step, self.fmt = cfg, key, step, fmt
        self.is_int = isinstance(getattr(cfg, key), int)
        lay = QGridLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)
        self.name = QLabel(label)
        self.value = QLabel()
        self.value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.value.setStyleSheet(f"color:{ACCENT}; font-weight:600;")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(round(lo / step), round(hi / step))
        lay.addWidget(self.name, 0, 0)
        lay.addWidget(self.value, 0, 1)
        lay.addWidget(self.slider, 1, 0, 1, 2)
        if tooltip:
            self.setToolTip(tooltip)
        self.slider.valueChanged.connect(self._changed)
        self.refresh()

    def refresh(self):
        self.slider.blockSignals(True)
        self.slider.setValue(round(getattr(self.cfg, self.key) / self.step))
        self.slider.blockSignals(False)
        self._show()

    def _changed(self, v):
        val = v * self.step
        setattr(self.cfg, self.key, int(round(val)) if self.is_int else float(val))
        self._show()

    def _show(self):
        self.value.setText(self.fmt.format(getattr(self.cfg, self.key)))


class HotkeyBridge(QObject):
    pause = Signal()
    calibrate = Signal()


# ------------------------------- main window -------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cfg = load_config()
        self.worker = None
        self.paused = False
        self._refreshers = []
        self._hotkeys = None

        self.setWindowTitle(APP_NAME)
        self.resize(1180, 760)
        self._build_ui()
        self._setup_shortcuts()
        self._set_status("STOPPED")
        self.log(f"Welcome! Settings & logs: {DATA_DIR}")

    # ---------- UI construction ----------
    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(14)

        # ===== left: header, video, indicators, log =====
        left = QVBoxLayout()
        left.setSpacing(10)
        header = QHBoxLayout()
        title = QLabel(APP_NAME)
        title.setStyleSheet("font-size:22px; font-weight:700;")
        subtitle = QLabel("index finger = cursor · head ← → = click · head ↑ ↓ = scroll")
        subtitle.setStyleSheet("color:#8a8ca3;")
        tbox = QVBoxLayout()
        tbox.addWidget(title)
        tbox.addWidget(subtitle)
        header.addLayout(tbox)
        header.addStretch()
        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setMinimumWidth(230)
        header.addWidget(self.status_label)
        left.addLayout(header)

        self.video = VideoView()
        left.addWidget(self.video, 1)

        ind = QHBoxLayout()
        self.pill_hand = Pill("Hand")
        self.pill_face = Pill("Face")
        self.pill_cal = Pill("Calibrated")
        self.calib_bar = QProgressBar()
        self.calib_bar.setRange(0, 100)
        self.calib_bar.setTextVisible(False)
        self.calib_bar.setFixedWidth(120)
        self.calib_bar.hide()
        self.fps_label = QLabel("FPS –")
        self.fps_label.setStyleSheet("color:#8a8ca3;")
        for w in (self.pill_hand, self.pill_face, self.pill_cal, self.calib_bar):
            ind.addWidget(w)
        ind.addStretch()
        ind.addWidget(self.fps_label)
        left.addLayout(ind)

        log_box = QGroupBox("Activity")
        lb = QVBoxLayout(log_box)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(300)
        self.log_view.setFixedHeight(110)
        lb.addWidget(self.log_view)
        left.addWidget(log_box)
        outer.addLayout(left, 1)

        # ===== right: controls in a scroll area =====
        panel = QWidget()
        right = QVBoxLayout(panel)
        right.setContentsMargins(0, 0, 6, 0)
        right.setSpacing(6)

        # controls
        ctl = QGroupBox("Controls")
        cl = QGridLayout(ctl)
        self.btn_start = QPushButton("▶  Start")
        self.btn_start.setObjectName("primary")
        self.btn_start.setToolTip("Start / stop the camera and tracking (F5)")
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setToolTip("Pause mouse control (Space, or Ctrl+Alt+P globally)")
        self.btn_cal = QPushButton("Recalibrate")
        self.btn_cal.setToolTip("Look straight at the screen, then press (C, or Ctrl+Alt+C globally)")
        self.btn_start.clicked.connect(self.toggle_start)
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_cal.clicked.connect(self.recalibrate)
        cl.addWidget(self.btn_start, 0, 0, 1, 2)
        cl.addWidget(self.btn_pause, 1, 0)
        cl.addWidget(self.btn_cal, 1, 1)
        right.addWidget(ctl)

        # head pad
        pad_box = QGroupBox("Head position")
        pl = QVBoxLayout(pad_box)
        self.pad = HeadPad(self.cfg)
        pl.addWidget(self.pad)
        right.addWidget(pad_box)

        # features
        feat = QGroupBox("Features")
        fl = QGridLayout(feat)
        self._check(fl, 0, 0, "Cursor (finger)", "enable_cursor")
        self._check(fl, 0, 1, "Head clicks", "enable_clicks")
        self._check(fl, 1, 0, "Head scroll", "enable_scroll")
        self._check(fl, 1, 1, "Show overlay", "show_overlay")
        self.chk_top = QCheckBox("Keep window on top")
        self.chk_top.toggled.connect(self._set_on_top)
        fl.addWidget(self.chk_top, 2, 0, 1, 2)
        right.addWidget(feat)

        # settings tabs
        tabs = QTabWidget()
        tabs.addTab(self._tab_cursor(), "Cursor")
        tabs.addTab(self._tab_clicks(), "Clicks")
        tabs.addTab(self._tab_scroll(), "Scroll")
        tabs.addTab(self._tab_advanced(), "Advanced")
        right.addWidget(tabs)

        btns = QHBoxLayout()
        b_save = QPushButton("Save settings")
        b_save.clicked.connect(self.save_settings)
        b_reset = QPushButton("Reset defaults")
        b_reset.clicked.connect(self.reset_defaults)
        btns.addWidget(b_save)
        btns.addWidget(b_reset)
        right.addLayout(btns)
        right.addStretch()

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(380)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(scroll)

    def _check(self, layout, row, col, text, key):
        cb = QCheckBox(text)
        cb.setChecked(getattr(self.cfg, key))
        cb.toggled.connect(lambda v, k=key: setattr(self.cfg, k, bool(v)))
        self._refreshers.append(lambda cb=cb, k=key: cb.setChecked(getattr(self.cfg, k)))
        layout.addWidget(cb, row, col)

    def _slider(self, layout, *args, **kw):
        row = SliderRow(self.cfg, *args, **kw)
        row.slider.valueChanged.connect(self.pad.update)   # redraw zones live
        self._refreshers.append(row.refresh)
        layout.addWidget(row)
        return row

    @staticmethod
    def _tab_page():
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        return w, lay

    def _tab_cursor(self):
        w, lay = self._tab_page()
        self._slider(lay, "smoothing", "Smoothing (responsiveness)", 0.05, 1.0, 0.01, "{:.2f}",
                     "Lower = smoother but laggier cursor. Higher = snappier but shakier.")
        self._slider(lay, "frame_margin", "Active area margin (px)", 20, 200, 5, "{:.0f}",
                     "Bigger = smaller hand movements cover the whole screen.")
        self._slider(lay, "cursor_hz", "Cursor update rate", 30, 240, 10, "{:.0f} Hz",
                     "How often the cursor glides toward your finger. Higher = smoother "
                     "(independent of camera FPS).")
        row = QHBoxLayout()
        row.addWidget(QLabel("Camera index"))
        self.cam_spin = QSpinBox()
        self.cam_spin.setRange(0, 9)
        self.cam_spin.setValue(self.cfg.camera_index)
        self.cam_spin.valueChanged.connect(self._camera_changed)
        self._refreshers.append(lambda: self.cam_spin.setValue(self.cfg.camera_index))
        row.addStretch()
        row.addWidget(self.cam_spin)
        lay.addLayout(row)
        note = QLabel("Camera changes apply on next Start.")
        note.setStyleSheet("color:#8a8ca3; font-size:11px;")
        lay.addWidget(note)
        lay.addStretch()
        return w

    def _tab_clicks(self):
        w, lay = self._tab_page()
        row = QHBoxLayout()
        row.addWidget(QLabel("Head gesture"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Turn (look left / right)", "turn")
        self.mode_combo.addItem("Tilt (ear to shoulder)", "tilt")
        self.mode_combo.setCurrentIndex(0 if self.cfg.head_mode == "turn" else 1)
        self.mode_combo.currentIndexChanged.connect(
            lambda i: setattr(self.cfg, "head_mode", self.mode_combo.itemData(i)))
        self._refreshers.append(lambda: self.mode_combo.setCurrentIndex(
            0 if self.cfg.head_mode == "turn" else 1))
        row.addStretch()
        row.addWidget(self.mode_combo)
        lay.addLayout(row)
        self._slider(lay, "trigger_deg", "Click angle", 8, 40, 1, "{:.0f}°",
                     "How far to turn your head before a click fires.")
        self._slider(lay, "return_deg", "Re-arm angle", 2, 25, 1, "{:.0f}°",
                     "Head must come back within this angle before the next click.")
        self._slider(lay, "click_cooldown", "Click cooldown", 0.1, 1.5, 0.05, "{:.2f} s",
                     "Minimum time between clicks.")
        lay.addStretch()
        return w

    def _tab_scroll(self):
        w, lay = self._tab_page()
        self._slider(lay, "pitch_trigger_deg", "Scroll start angle", 5, 30, 1, "{:.0f}°",
                     "How far up/down to tilt your head before scrolling starts.")
        self._slider(lay, "pitch_return_deg", "Scroll stop angle", 2, 20, 1, "{:.0f}°",
                     "Scrolling stops when your head returns within this angle.")
        self._slider(lay, "scroll_min_rate", "Min speed (notches/s)", 1, 20, 1, "{:.0f}")
        self._slider(lay, "scroll_max_rate", "Max speed (notches/s)", 5, 60, 1, "{:.0f}")
        self._slider(lay, "scroll_unit", "Scroll amount per notch", 1, 240, 1, "{:.0f}",
                     "Windows: 120 = one wheel notch. macOS/Linux: 1 = one notch/line.")
        lay.addStretch()
        return w

    def _tab_advanced(self):
        w, lay = self._tab_page()
        self._slider(lay, "gesture_frames", "Gesture confirm frames", 1, 6, 1, "{:.0f}",
                     "Frames a head gesture must persist. Higher = fewer accidents, more delay.")
        self._slider(lay, "scroll_ramp_deg", "Scroll speed ramp", 5, 30, 1, "{:.0f}°",
                     "Extra degrees beyond start angle to reach max scroll speed.")
        self._slider(lay, "calibration_frames", "Calibration frames", 10, 90, 5, "{:.0f}")
        self._slider(lay, "neutral_drift", "Posture drift follow", 0.0, 0.05, 0.002, "{:.3f}",
                     "How fast the neutral head position follows slow posture changes.")
        lay.addStretch()
        return w

    # ---------- shortcuts ----------
    def _setup_shortcuts(self):
        QShortcut(QKeySequence("F5"), self, activated=self.toggle_start)
        QShortcut(QKeySequence("Space"), self, activated=self.toggle_pause)
        QShortcut(QKeySequence("C"), self, activated=self.recalibrate)

        self._bridge = HotkeyBridge()
        self._bridge.pause.connect(self.toggle_pause)
        self._bridge.calibrate.connect(self.recalibrate)
        try:
            from pynput import keyboard  # optional dependency
            self._hotkeys = keyboard.GlobalHotKeys({
                "<ctrl>+<alt>+p": self._bridge.pause.emit,
                "<ctrl>+<alt>+c": self._bridge.calibrate.emit,
            })
            self._hotkeys.daemon = True
            self._hotkeys.start()
            self.log("Global hotkeys on: Ctrl+Alt+P pause, Ctrl+Alt+C recalibrate")
        except Exception:  # noqa: BLE001 - pynput missing or not permitted
            self.log("Tip: 'pip install pynput' for global hotkeys (Ctrl+Alt+P / Ctrl+Alt+C)")

    # ---------- actions ----------
    def toggle_start(self):
        if self.worker is not None:
            self.stop_tracking()
        else:
            self.start_tracking()

    def start_tracking(self):
        self.paused = False
        self.btn_pause.setText("Pause")
        self.worker = TrackerWorker(self.cfg)
        self.worker.frame_ready.connect(self.video.set_image)
        self.worker.state_ready.connect(self._on_state)
        self.worker.event.connect(self.log)
        self.worker.error.connect(self._on_error)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.send_preview = not self.isMinimized()
        self.worker.start(QThread.HighPriority)
        self.btn_start.setText("■  Stop")
        self.btn_start.setObjectName("danger")
        self._restyle(self.btn_start)
        self._set_status("STARTING...")
        self.video.show_placeholder("Starting camera...")

    def stop_tracking(self):
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(3000)

    def _on_worker_finished(self):
        self.worker = None
        self.btn_start.setText("▶  Start")
        self.btn_start.setObjectName("primary")
        self._restyle(self.btn_start)
        self._set_status("STOPPED")
        self.video.show_placeholder()
        self.pad.reset()
        for p in (self.pill_hand, self.pill_face, self.pill_cal):
            p.set_on(False)
        self.calib_bar.hide()
        self.fps_label.setText("FPS –")
        self.log("Tracking stopped")

    def toggle_pause(self):
        if self.worker is None:
            return
        self.paused = not self.paused
        self.worker.request_pause(self.paused)
        self.btn_pause.setText("Resume" if self.paused else "Pause")

    def recalibrate(self):
        if self.worker is not None:
            self.worker.request_calibrate()

    def save_settings(self):
        try:
            save_config(self.cfg)
            self.log("Settings saved")
        except OSError as e:
            QMessageBox.warning(self, APP_NAME, f"Could not save settings:\n{e}")

    def reset_defaults(self):
        defaults = fm.Config()
        for k, v in defaults.to_dict().items():
            setattr(self.cfg, k, v)          # mutate in place: engine sees it live
        for refresh in self._refreshers:
            refresh()
        self.pad.update()
        self.log("Settings reset to defaults")

    def _camera_changed(self, v):
        self.cfg.camera_index = int(v)
        if self.worker is not None:
            self.log("Camera index changed - press Stop/Start to apply")

    def _set_on_top(self, on):
        self.setWindowFlag(Qt.WindowStaysOnTopHint, on)
        self.show()

    # ---------- worker callbacks ----------
    def _on_state(self, st):
        self._set_status(st["status"])
        self.pill_hand.set_on(st["hand"])
        self.pill_face.set_on(st["face"])
        self.pill_cal.set_on(st["calibrated"])
        if st["calibrated"]:
            self.calib_bar.hide()
        else:
            self.calib_bar.show()
            self.calib_bar.setValue(int(st["calib_progress"] * 100))
        self.fps_label.setText(f"FPS {st.get('fps', 0):.0f}")
        self.pad.update_state(st)

    def _on_error(self, msg):
        self.log(f"ERROR: {msg}")
        QMessageBox.critical(self, APP_NAME, msg)

    # ---------- helpers ----------
    def _set_status(self, status):
        col = STATUS_COLORS.get(status, "#7c6cff")
        self.status_label.setText(status)
        self.status_label.setStyleSheet(
            f"background:rgba({int(col[1:3],16)},{int(col[3:5],16)},{int(col[5:7],16)},40); "
            f"color:{col}; border:2px solid {col}; border-radius:16px;"
            "padding:8px 16px; font-size:15px; font-weight:700;")

    @staticmethod
    def _restyle(w):
        w.style().unpolish(w)
        w.style().polish(w)

    def log(self, msg):
        self.log_view.appendPlainText(f"{time.strftime('%H:%M:%S')}  {msg}")

    def changeEvent(self, e):
        super().changeEvent(e)
        if e.type() == QEvent.WindowStateChange and self.worker is not None:
            minimized = self.isMinimized()
            self.worker.send_preview = not minimized
            if minimized:
                self.video.show_placeholder("Preview paused while minimised\n(tracking keeps running)")

    def closeEvent(self, e):
        self.stop_tracking()
        if self._hotkeys is not None:
            self._hotkeys.stop()
        try:
            save_config(self.cfg)
        except OSError:
            pass
        super().closeEvent(e)


def _setup_logging():
    """In the packaged app, send output + crashes to a log file."""
    if fm.IS_FROZEN:
        try:
            log = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
            sys.stdout = sys.stderr = log
        except OSError:
            pass
    print(f"\n===== {APP_NAME} started {time.strftime('%Y-%m-%d %H:%M:%S')} =====")

    def excepthook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        print(text)
        try:
            QMessageBox.critical(None, APP_NAME, f"Unexpected error:\n\n{exc}\n\nDetails: {LOG_PATH}")
        except Exception:  # noqa: BLE001
            pass
    sys.excepthook = excepthook


def _app_icon():
    from PySide6.QtGui import QIcon
    path = fm.resource_path("icon.png")
    return QIcon(path) if path else QIcon()


def selftest(quiet=False):
    """`FingerMouse --selftest [--quiet]`: check the packaged app step by step.
    Writes the report to selftest.txt and (unless --quiet) shows it in a dialog."""
    import platform
    import tempfile
    report = [f"Finger Mouse self-test  {time.strftime('%Y-%m-%d %H:%M:%S')}"]
    step = report.append
    app = QApplication.instance() or QApplication(sys.argv)
    code = 1
    try:
        step(f"Python {platform.python_version()} | {platform.platform()}")
        step(f"Frozen: {fm.IS_FROZEN} | onefile temp/bundle dir: {fm.BUNDLE_DIR}")
        try:
            free = shutil.disk_usage(tempfile.gettempdir()).free / 2**30
            step(f"Temp folder: {tempfile.gettempdir()} ({free:.1f} GB free)")
        except OSError:
            pass

        step("[1/5] Locating models...")
        for name in (fm.HAND_MODEL[1], fm.FACE_MODEL[1]):
            step(f"      {name}: {fm.resource_path(name) or 'NOT FOUND (will download)'}")

        step("[2/5] Loading MediaPipe native library...")
        import ctypes
        from importlib import resources
        libname = {"win32": "libmediapipe.dll", "darwin": "libmediapipe.dylib"}.get(sys.platform, "libmediapipe.so")
        lib = resources.files("mediapipe.tasks.c") / libname
        step(f"      {lib}  exists={os.path.exists(str(lib))}")
        ctypes.CDLL(str(lib))
        step("      OK")

        step("[3/5] Creating hand + face trackers...")
        import numpy as np
        import mediapipe as mp
        engine = fm.FingerMouseEngine(fm.Config(), on_event=lambda m: step(f"      {m}"))
        engine.start()
        step("[4/5] Running trackers on a test image...")
        blank = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.zeros((480, 640, 3), np.uint8))
        engine.hands.detect_for_video(blank, 1)
        engine.face.detect_for_video(blank, 1)
        engine.close()
        step("      OK")

        step("[5/5] Opening the window...")
        win = MainWindow()
        win.show()
        app.processEvents()
        win.close()
        step("      OK")
        step("SELFTEST PASSED")
        code = 0
    except BaseException:  # noqa: BLE001 - report absolutely everything
        step(traceback.format_exc())
        step("SELFTEST FAILED")

    text = "\n".join(report)
    out = os.path.join(DATA_DIR, "selftest.txt")
    try:
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
    except OSError:
        pass
    print(text)
    if not quiet:
        box = QMessageBox(QMessageBox.Information if code == 0 else QMessageBox.Critical,
                          f"{APP_NAME} self-test", "SELFTEST PASSED" if code == 0 else
                          "SELFTEST FAILED - click 'Show Details...' and copy the text.")
        box.setDetailedText(text + f"\n\n(saved to {out})")
        box.exec()
    return code


def main():
    if "--selftest" in sys.argv:
        sys.exit(selftest(quiet="--quiet" in sys.argv))
    _setup_logging()
    if sys.platform == "win32":
        # Show our own icon in the Windows taskbar (not Python's)
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("FingerMouse.App")
        except Exception:  # noqa: BLE001
            pass
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    app.setWindowIcon(_app_icon())
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
