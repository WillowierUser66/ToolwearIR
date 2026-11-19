import sys, os, time, platform, threading
from collections import deque
import cv2 as cv
import numpy as np

from comtypes import CoInitialize
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

SCALE_PERCENT   = 200
FRAME_DELAY_MS  = int(1000/60)
THRESH_8U       = 110
PLOT_WINDOW_S   = 10     # seconds of history visible in the live plots
PLOT_HISTORY_LEN = 300   # points kept for plotting (~30 s at the Lepton's ~9 fps)
# Order matches _horizontal_rois_from_vline() and the CSV columns
HLINE_LABELS    = ["Line 1", "Line 2", "Line 3", "Line 4", "Baseline"]
# Total length of each horizontal line in image px (slider range; default = original 20 px)
HLINE_LEN_MIN, HLINE_LEN_MAX, HLINE_LEN_DEFAULT = 4, 80, 20

# ----- UI theme -----
UI_FONT = "Segoe UI" if sys.platform == "win32" else "Helvetica"
THEME = {
    "bg": "#0f1115", "panel": "#161a21", "card": "#1c2029", "card_hover": "#262b36",
    "border": "#2a2f3a", "text": "#e6e8ee", "muted": "#8a93a6",
    "accent": "#3b82f6", "accent_hover": "#2563eb",
    "danger": "#ef4444", "danger_hover": "#dc2626", "danger_bg": "#3a1518",
    "success": "#22c55e", "success_bg": "#12301f",
}
LINE_COLORS = ["#60a5fa", "#fbbf24", "#34d399", "#f472b6", "#a78bfa"]  # Line 1-4, Baseline
MAX_COLOR   = "#fb923c"
MASK_COLOR  = "#facc15"
CONTOUR_COLOR = "#4ade80"
LEGEND_ONE_ROW_MIN_W = 420  # plot width (px) needed to fit the 5-entry legend on one row
DISPLAY_ZOOM = 2   # initial on-screen enlargement; it then follows the window size.
                   # Display only: processing and CSV values stay at SCALE_PERCENT.
VLINE_COLOR = "#e5e7eb"
HINT_IDLE   = "Draw a mask around the endmill, then a vertical line along it for the temperature profile."

# On the Lepton 3.5 the radiometric range is selected by the gain mode:
# HIGH gain = -10..140 °C (finer resolution), LOW gain = -10..400 °C.
TEMP_RANGES = {
    "Low (-10°C to 140°C)":  "HIGH",
    "High (-10°C to 400°C)": "LOW",
}
DEFAULT_TEMP_RANGE = "High (-10°C to 400°C)"

# ----- Camera connection indicator -----
# FLIR part numbers reported by the camera (oem.GetFlirPartNumber) -> model name.
# 500-0771-01 verified against this project's camera. Unknown part numbers fall
# back to the sensor family implied by the frame size, and the P/N is still shown.
KNOWN_LEPTON_PARTS = {
    "500-0771-01": "Lepton 3.5",
}
LEPTON_FAMILY_BY_RES = {(160, 120): "Lepton 3.x", (80, 60): "Lepton 1.x / 2.x"}
CAMERA_POLL_S     = 1.0   # how often the background thread pings the camera
CAMERA_TIMEOUT_S  = 3.5   # no successful ping for this long -> "Not connected"
VIDEO_TIMEOUT_S   = 2.0   # no frame for this long -> "No video"
LED_COLORS = {"ok": "#22c55e", "warn": "#f59e0b", "off": "#ef4444"}

CoInitialize()
bits, _ = platform.architecture()
sys.path.append(os.path.join("", "x64" if bits == "64bit" else "x86"))

import clr
clr.AddReference("LeptonUVC")
from Lepton import CCI

found_device = next((d for d in CCI.GetDevices() if d.Name.startswith("PureThermal")), None)
if not found_device:
    print("Could not find Lepton device")
    sys.exit(1)

lep = found_device.Open()
# Serialises camera control calls between the UI and the connection monitor thread.
CAMERA_LOCK = threading.Lock()

def read_camera_info(device, cam):
    """Collect whatever identifying metadata the camera reports; missing fields are skipped."""
    info = {"board": device.Name}
    try:
        raw = cam.oem.GetFlirPartNumber().value
        part = bytes(b & 0xFF for b in raw).split(b"\x00")[0].decode("ascii", "replace").strip()
        if part:
            info["part"] = part
    except Exception:
        pass
    try:
        info["serial"] = int(cam.sys.GetFlirSerialNumber())
    except Exception:
        pass
    try:
        cam.rad.GetTLinearEnableState()  # only radiometric Leptons support TLinear
        info["radiometric"] = True
    except Exception:
        pass
    return info

def describe_camera(info, resolution=None):
    """-> (model, details) for the header, e.g. ("Lepton 3.5", "P/N … · S/N … · PureThermal …")."""
    model = KNOWN_LEPTON_PARTS.get(info.get("part"))
    if model is None:
        model = LEPTON_FAMILY_BY_RES.get(resolution, "Lepton (model not reported)")
        if info.get("radiometric"):
            model += " (radiometric)"
    details = []
    if "part" in info:
        details.append(f"P/N {info['part']}")
    if "serial" in info:
        details.append(f"S/N {info['serial']}")
    details.append(info["board"])
    return model, "  ·  ".join(details)

camera_info = read_camera_info(found_device, lep)

clr.AddReference("ManagedIR16Filters")
from IR16Filters import IR16Capture, NewBytesFrameEvent

incoming_frames = deque(maxlen=10)
last_frame_time = 0.0  # wall-clock time the last camera frame arrived

def got_a_frame(frame, width, height):
    global last_frame_time
    incoming_frames.append((height, width, frame))
    last_frame_time = time.time()

capture = IR16Capture()
capture.SetupGraphWithBytesCallback(NewBytesFrameEvent(got_a_frame))
capture.RunGraph()
time.sleep(3)

def short_array_to_numpy(h, w, frame):
    # Read the .NET UInt16[] directly through the buffer protocol; iterating it
    # element by element (np.fromiter) costs ~100 ms per frame.
    try:
        arr = np.frombuffer(memoryview(frame), dtype=np.uint16).copy()
    except (TypeError, ValueError, BufferError):
        arr = np.fromiter(frame, dtype=np.uint16)
    return arr.reshape(h, w)

def centikelvin_to_celsius(t_ck):
    return (t_ck - 27315) / 100.0

class ThermalVideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Tool wear IR Monitor")
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)

        self.running = True
        self.suspend_video = False
        self.recording = False
        self.recording_start_time = None
        self.stream_start_time = time.time()
        # rows: [elapsed_s, edge_avg, edge_max, edge_area, hline1..hline5]
        self.data_records = []
        self.after_id = None
        self._last_frame = None  # last camera frame processed, to skip repeats

        self.user_mask = None
        self.mask_bounds = None
        self._drag_start = None
        self._rubberband_id = None

        self.vline_coords = None   # (x, y0, y1)
        self.vline_length_px = 0
        self.hline_len_px = HLINE_LEN_DEFAULT  # total length of each horizontal line
        self._line_drag_start = None
        self._line_rubberband_id = None

        self._frame_times = deque(maxlen=20)  # for the fps readout
        # Display transform: canvas = image * zoom + offset (updated on resize)
        self.zoom = float(DISPLAY_ZOOM)
        self.ox = self.oy = 0.0
        self._last_draw = None  # (frame_bgr, contour, line temps) for repaints

        if incoming_frames:
            h, w, f = incoming_frames[-1]
            preview = short_array_to_numpy(h, w, f)
            self.sensor_res = (w, h)
        else:
            self.sensor_res = None
            preview = np.zeros((240, 320), np.uint16)

        self.widthScaled  = int(preview.shape[1] * SCALE_PERCENT / 100)
        self.heightScaled = int(preview.shape[0] * SCALE_PERCENT / 100)
        self.dim = (self.widthScaled, self.heightScaled)

        self.user_mask = np.zeros((self.heightScaled, self.widthScaled), np.uint8)

        # Plot history is bounded; only the last PLOT_WINDOW_S seconds are shown.
        self.time_data = deque(maxlen=PLOT_HISTORY_LEN)
        self.max_series = deque(maxlen=PLOT_HISTORY_LEN)
        self.hline_series = [deque(maxlen=PLOT_HISTORY_LEN) for _ in HLINE_LABELS]

        self._build_style()
        self._build_layout()
        self._use_dark_titlebar()
        self.change_temp_range()

        self._start_camera_monitor()
        self._refresh_camera_led()
        self.update_video()

    # ----- Camera connection monitor -----
    def _start_camera_monitor(self):
        """Ping the camera from a background thread so a hung or unplugged camera
        never freezes the UI. The UI reads only the timestamp of the last good ping."""
        self._last_ping_ok = time.time()

        def ping():
            global lep
            if not CAMERA_LOCK.acquire(timeout=1):
                return False
            try:
                lep.sys.GetCameraUpTime()
                return True
            except Exception:
                # Handle may be stale after an unplug/replug: try to reopen the camera.
                try:
                    dev = next((d for d in CCI.GetDevices() if d.Name.startswith("PureThermal")), None)
                    if dev is None:
                        return False
                    cam = dev.Open()
                    cam.sys.GetCameraUpTime()
                    lep = cam
                    return True
                except Exception:
                    return False
            finally:
                CAMERA_LOCK.release()

        def run():
            CoInitialize()  # COM is per-thread
            while self.running:
                if ping():
                    self._last_ping_ok = time.time()
                time.sleep(CAMERA_POLL_S)

        threading.Thread(target=run, name="camera-monitor", daemon=True).start()

    def _refresh_camera_led(self):
        if not self.running:
            return
        now = time.time()
        if now - self._last_ping_ok > CAMERA_TIMEOUT_S:
            state, text = "off", "Not connected"
        elif now - last_frame_time > VIDEO_TIMEOUT_S:
            state, text = "warn", "Connected · no video"
        else:
            state, text = "ok", "Connected"
        self.led.itemconfigure(self.led_dot, fill=LED_COLORS[state])
        self.cam_status_var.set(text)
        self.camera_state = state
        self.root.after(500, self._refresh_camera_led)

    # ----- Look & layout -----
    def _build_style(self):
        T = THEME
        self.root.configure(bg=T["bg"])
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", background=T["bg"], foreground=T["text"],
                        font=(UI_FONT, 10), borderwidth=0, focuscolor=T["bg"])
        style.configure("TFrame", background=T["bg"])
        style.configure("Card.TFrame", background=T["card"])
        style.configure("Side.TFrame", background=T["panel"])
        style.configure("TLabel", background=T["bg"], foreground=T["text"])
        style.configure("Title.TLabel", font=(UI_FONT, 15, "bold"))
        style.configure("Subtitle.TLabel", foreground=T["muted"], font=(UI_FONT, 9))
        style.configure("Footer.TLabel", foreground=T["muted"], font=(UI_FONT, 9))
        style.configure("CamStatus.TLabel", foreground=T["text"], font=(UI_FONT, 9, "bold"))
        style.configure("CamModel.TLabel", foreground=T["text"], font=(UI_FONT, 9))
        style.configure("Side.TLabel", background=T["panel"], foreground=T["text"])
        style.configure("Section.TLabel", background=T["panel"], foreground=T["muted"],
                        font=(UI_FONT, 8, "bold"))
        style.configure("SideMuted.TLabel", background=T["panel"], foreground=T["muted"],
                        font=(UI_FONT, 9))
        style.configure("Value.TLabel", background=T["panel"], foreground=T["text"],
                        font=(UI_FONT, 15, "bold"))
        style.configure("TSeparator", background=T["border"])
        style.configure("Horizontal.TScale", background=T["accent"], troughcolor=T["card"],
                        bordercolor=T["border"], lightcolor=T["card"], darkcolor=T["card"],
                        sliderthickness=14, gripcount=0,
                        padding=0, focusthickness=0)  # else the accent bg shows as a ring
        style.map("Horizontal.TScale", background=[("active", T["accent_hover"])])

        def button(name, bg, hover, fg="#ffffff"):
            style.configure(name, background=bg, foreground=fg, padding=(12, 8),
                            relief="flat", borderwidth=0, focuscolor=bg,
                            font=(UI_FONT, 10))
            style.map(name, background=[("pressed", hover), ("active", hover)],
                      focuscolor=[("focus", bg)])
        button("TButton", T["card"], T["card_hover"], T["text"])
        button("Accent.TButton", T["accent"], T["accent_hover"])
        button("Danger.TButton", T["danger"], T["danger_hover"])

        style.configure("TCombobox", fieldbackground=T["card"], background=T["card"],
                        foreground=T["text"], arrowcolor=T["muted"], bordercolor=T["border"],
                        lightcolor=T["card"], darkcolor=T["card"], padding=6)
        style.map("TCombobox",
                  fieldbackground=[("readonly", T["card"])],
                  foreground=[("readonly", T["text"])],
                  selectbackground=[("readonly", T["card"])],
                  selectforeground=[("readonly", T["text"])],
                  bordercolor=[("focus", T["accent"])])
        self.root.option_add("*TCombobox*Listbox.background", T["card"])
        self.root.option_add("*TCombobox*Listbox.foreground", T["text"])
        self.root.option_add("*TCombobox*Listbox.selectBackground", T["accent"])
        self.root.option_add("*TCombobox*Listbox.font", (UI_FONT, 10))

    def _build_layout(self):
        T = THEME
        root = self.root

        # Header: title + live/recording status pill
        header = ttk.Frame(root, padding=(16, 12, 16, 4))
        header.grid(row=0, column=0, columnspan=3, sticky="ew")
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="Tool Wear IR Monitor", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        # Camera connection LED + status + model details from the camera's metadata
        cam_row = ttk.Frame(header)
        cam_row.grid(row=1, column=0, sticky="w", pady=(2, 0))
        self.led = tk.Canvas(cam_row, width=12, height=12, bg=T["bg"], highlightthickness=0, bd=0)
        self.led.grid(row=0, column=0, padx=(1, 6))
        self.led_dot = self.led.create_oval(1, 1, 11, 11, fill=LED_COLORS["warn"], outline="")
        self.cam_status_var = tk.StringVar(value="Checking camera…")
        ttk.Label(cam_row, textvariable=self.cam_status_var, style="CamStatus.TLabel").grid(row=0, column=1)
        self.cam_model, cam_details = describe_camera(camera_info, self.sensor_res)
        ttk.Label(cam_row, text=f"{self.cam_model}", style="CamModel.TLabel").grid(row=0, column=2, padx=(10, 0))
        ttk.Label(cam_row, text=f"·  {cam_details}", style="Subtitle.TLabel").grid(row=0, column=3, padx=(8, 0))
        self.status_pill = tk.Label(header, text="", font=(UI_FONT, 9, "bold"), padx=12, pady=4)
        self.status_pill.grid(row=0, column=2, rowspan=2, sticky="e")

        # Sidebar with grouped controls
        side = ttk.Frame(root, style="Side.TFrame", padding=16)
        side.grid(row=1, column=0, sticky="ns", padx=(16, 8), pady=8)
        side.columnconfigure(0, weight=1); side.columnconfigure(1, weight=1)
        r = 0

        def section(title, first=False):
            nonlocal r
            if not first:
                ttk.Separator(side).grid(row=r, column=0, columnspan=2, sticky="ew", pady=12); r += 1
            ttk.Label(side, text=title.upper(), style="Section.TLabel").grid(
                row=r, column=0, columnspan=2, sticky="w", pady=(0, 6)); r += 1

        def button_pair(left, right):
            nonlocal r
            ttk.Button(side, text=left[0], command=left[1]).grid(row=r, column=0, sticky="ew", padx=(0, 4))
            ttk.Button(side, text=right[0], command=right[1]).grid(row=r, column=1, sticky="ew", padx=(4, 0))
            r += 1

        section("Edge ROI mask", first=True)
        button_pair(("Draw mask", self.enable_mask_draw), ("Clear", self.clear_mask))

        section("Temperature profile")
        button_pair(("Draw line", self.enable_vertical_line), ("Clear", self.clear_line))
        self.line_length_var = tk.StringVar(value="No line drawn")
        ttk.Label(side, textvariable=self.line_length_var, style="SideMuted.TLabel").grid(
            row=r, column=0, columnspan=2, sticky="w", pady=(6, 0)); r += 1

        # Slider: total length of each horizontal line (image px), centred on the vertical line
        ttk.Label(side, text="Horizontal line length", style="SideMuted.TLabel").grid(
            row=r, column=0, sticky="w", pady=(10, 0))
        self.hline_len_label = tk.StringVar(value=f"{self.hline_len_px} px")
        ttk.Label(side, textvariable=self.hline_len_label, style="Side.TLabel").grid(
            row=r, column=1, sticky="e", pady=(10, 0)); r += 1
        self.hline_len_var = tk.IntVar(value=self.hline_len_px)
        ttk.Scale(side, from_=HLINE_LEN_MIN, to=HLINE_LEN_MAX, orient="horizontal",
                  variable=self.hline_len_var, command=self._on_hline_len).grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=(2, 0)); r += 1

        section("Camera range")
        self.temp_range_var = tk.StringVar(value=DEFAULT_TEMP_RANGE)
        self.temp_range_dropdown = ttk.Combobox(
            side, textvariable=self.temp_range_var, state="readonly",
            values=list(TEMP_RANGES.keys()), width=22
        )
        self.temp_range_dropdown.grid(row=r, column=0, columnspan=2, sticky="ew"); r += 1
        self.temp_range_dropdown.bind("<<ComboboxSelected>>", self.change_temp_range)

        section("Edge ROI readings")
        readings = ttk.Frame(side, style="Side.TFrame")
        readings.grid(row=r, column=0, columnspan=2, sticky="ew"); r += 1
        self.reading_vars = {}
        for col, (key, label) in enumerate((("avg", "Avg °C"), ("max", "Max °C"), ("area", "Area px²"))):
            readings.columnconfigure(col, weight=1)
            var = tk.StringVar(value="—"); self.reading_vars[key] = var
            ttk.Label(readings, textvariable=var, style="Value.TLabel").grid(row=0, column=col, sticky="w")
            ttk.Label(readings, text=label, style="SideMuted.TLabel").grid(row=1, column=col, sticky="w")

        section("Recording")
        self.record_button = ttk.Button(side, command=self.toggle_recording)
        self.record_button.grid(row=r, column=0, columnspan=2, sticky="ew"); r += 1
        self.record_info_var = tk.StringVar(value="Not recording")
        ttk.Label(side, textvariable=self.record_info_var, style="SideMuted.TLabel").grid(
            row=r, column=0, columnspan=2, sticky="w", pady=(6, 0)); r += 1
        self._refresh_record_button()

        side.rowconfigure(r, weight=1); r += 1  # push Quit to the bottom
        ttk.Button(side, text="Quit", command=self.quit_app).grid(
            row=r, column=0, columnspan=2, sticky="ew")

        # Video card
        # The canvas fills its card; the image is scaled to fit and centred
        # (see _on_canvas_resize), so it grows and shrinks with the window.
        video_card = ttk.Frame(root, style="Card.TFrame", padding=8)
        video_card.grid(row=1, column=1, sticky="nsew", padx=8, pady=8)
        self.canvas = tk.Canvas(video_card, width=self.widthScaled * DISPLAY_ZOOM,
                                height=self.heightScaled * DISPLAY_ZOOM,
                                bg=T["card"], highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Plot cards share the column height equally and stretch with it
        plots = ttk.Frame(root)
        plots.grid(row=1, column=2, sticky="nsew", padx=(8, 16), pady=8)
        plots.columnconfigure(0, weight=1)
        plots.rowconfigure(0, weight=1); plots.rowconfigure(1, weight=1)

        def plot_card(row, title, pady):
            card = ttk.Frame(plots, style="Card.TFrame", padding=4)
            card.grid(row=row, column=0, pady=pady, sticky="nsew")
            fig, ax = plt.subplots(figsize=(4.6, 2.32), dpi=100)
            self._style_axes(fig, ax, title)
            canvas = FigureCanvasTkAgg(fig, master=card)
            canvas.get_tk_widget().configure(bg=T["card"], highlightthickness=0)
            canvas.get_tk_widget().pack(fill="both", expand=True)
            return fig, ax, canvas

        self.fig_lines, self.ax_lines, self.canvas_lines = plot_card(0, "Horizontal line temperatures", (0, 8))
        self.hline_plots = [self.ax_lines.plot([], [], label=lbl, color=c, linewidth=1.6)[0]
                            for lbl, c in zip(HLINE_LABELS, LINE_COLORS)]
        self._legend_ncol = None
        self._fit_legend(5)
        self.canvas_lines.get_tk_widget().bind(
            "<Configure>", lambda e: self._fit_legend(5 if e.width >= LEGEND_ONE_ROW_MIN_W else 3), add="+")
        # A layout engine re-fits margins on every draw, so plots stay tidy at any size.
        self.fig_lines.set_layout_engine("tight", pad=0.8)

        self.fig_max, self.ax_max, self.canvas_max = plot_card(1, "Edge ROI maximum temperature", 0)
        self.line_max, = self.ax_max.plot([], [], color=MAX_COLOR, linewidth=1.6)
        self.fig_max.set_layout_engine("tight", pad=0.8)

        # Footer: drawing hints + fps
        footer = ttk.Frame(root, padding=(16, 0, 16, 10))
        footer.grid(row=2, column=0, columnspan=3, sticky="ew")
        footer.columnconfigure(0, weight=1)
        self.hint_var = tk.StringVar(value=HINT_IDLE)
        ttk.Label(footer, textvariable=self.hint_var, style="Footer.TLabel").grid(row=0, column=0, sticky="w")
        self.fps_var = tk.StringVar(value="")
        ttk.Label(footer, textvariable=self.fps_var, style="Footer.TLabel").grid(row=0, column=1, sticky="e")

        # Video and plots take the extra space; the sidebar keeps its width.
        root.columnconfigure(1, weight=3)
        root.columnconfigure(2, weight=2)
        root.rowconfigure(1, weight=1)
        root.update_idletasks()
        root.minsize(min(root.winfo_reqwidth(), 1000), min(root.winfo_reqheight(), 560))

    def _fit_legend(self, ncol):
        """Line-plot legend sits between the title and the axes so it never covers
        traces; it wraps to 2 rows when the plot is too narrow for one."""
        if ncol == self._legend_ncol:
            return
        self._legend_ncol = ncol
        T = THEME
        rows = -(-len(HLINE_LABELS) // ncol)
        self.ax_lines.set_title(self.ax_lines.get_title("left"), loc="left", color=T["text"],
                                fontsize=10, fontweight="bold", pad=8 + 12 * rows)
        self.ax_lines.legend(loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2,
                             fontsize=7, ncol=ncol, frameon=False, labelcolor=T["text"],
                             handlelength=1.2, columnspacing=1.0)

    @staticmethod
    def _style_axes(fig, ax, title):
        T = THEME
        fig.patch.set_facecolor(T["card"]); ax.set_facecolor(T["card"])
        ax.set_title(title, color=T["text"], fontsize=10, loc="left", fontweight="bold", pad=8)
        ax.set_xlabel("Time (s)", color=T["muted"], fontsize=8)
        ax.set_ylabel("°C", color=T["muted"], fontsize=8)
        ax.tick_params(colors=T["muted"], labelsize=8, length=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(T["border"])
        ax.grid(True, color=T["border"], linewidth=0.6)

    def _use_dark_titlebar(self):
        # Windows 10/11: ask DWM for a dark title bar to match the theme.
        if sys.platform != "win32":
            return
        try:
            import ctypes
            self.root.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(self.root.winfo_id())
            value = ctypes.c_int(1)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(value), ctypes.sizeof(value))
        except Exception:
            pass

    def _set_drawing(self, hint):
        self.hint_var.set(hint or HINT_IDLE)
        self.canvas.configure(cursor="crosshair" if hint else "")

    def _refresh_record_button(self):
        if self.recording:
            self.record_button.configure(text="■  Stop and save CSV", style="Danger.TButton")
        else:
            self.record_button.configure(text="●  Start recording", style="Accent.TButton")

    def _update_status(self, have_signal):
        T = THEME
        if self.recording:
            secs = int(time.time() - self.recording_start_time)
            self.status_pill.configure(text=f"●  REC  {secs // 60:02d}:{secs % 60:02d}",
                                       bg=T["danger_bg"], fg=T["danger"])
            self.record_info_var.set(f"{secs // 60:02d}:{secs % 60:02d}  ·  {len(self.data_records)} samples")
        elif have_signal:
            self.status_pill.configure(text="●  LIVE", bg=T["success_bg"], fg=T["success"])
        else:
            self.status_pill.configure(text="NO SIGNAL", bg=T["card"], fg=T["muted"])
        if len(self._frame_times) > 1:
            span = self._frame_times[-1] - self._frame_times[0]
            if span > 0:
                self.fps_var.set(f"{(len(self._frame_times) - 1) / span:.1f} fps")

    # Mask drawing
    def enable_mask_draw(self):
        self._unbind_all_draw(); self.suspend_video = True
        self._set_drawing("Drag on the image to draw the mask rectangle.")
        self.canvas.bind("<ButtonPress-1>", self._on_mask_start)
        self.canvas.bind("<B1-Motion>", self._on_mask_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mask_release)

    def _on_mask_start(self, event):
        self._drag_start = (self._clamp_x(event.x), self._clamp_y(event.y))
        if self._rubberband_id is not None:
            self.canvas.delete(self._rubberband_id); self._rubberband_id = None

    def _on_mask_drag(self, event):
        if not self._drag_start: return
        x0, y0 = self._drag_start
        x1, y1 = self._clamp_x(event.x), self._clamp_y(event.y)
        coords = (*self._to_canvas(x0, y0), *self._to_canvas(x1, y1))
        if self._rubberband_id is None:
            self._rubberband_id = self.canvas.create_rectangle(
                *coords, outline=MASK_COLOR, width=2, dash=(4, 3)
            )
        else:
            self.canvas.coords(self._rubberband_id, *coords)

    def _on_mask_release(self, event):
        if not self._drag_start: return
        x0, y0 = self._drag_start
        x1, y1 = self._clamp_x(event.x), self._clamp_y(event.y)
        self._drag_start = None
        xL, xR = sorted((x0, x1)); yT, yB = sorted((y0, y1))
        self.user_mask.fill(0); self.user_mask[yT:yB, xL:xR] = 1
        self.mask_bounds = (xL, yT, xR, yB)
        if self._rubberband_id is not None:
            self.canvas.delete(self._rubberband_id); self._rubberband_id = None
        self._unbind_all_draw(); self.suspend_video = False
        self._set_drawing(None)

    def clear_mask(self):
        self.user_mask.fill(0); self.mask_bounds = None

    # Vertical line + release
    def enable_vertical_line(self):
        self._unbind_all_draw(); self.suspend_video = True
        self._set_drawing("Drag vertically along the endmill to place the temperature-profile line.")
        self.canvas.bind("<ButtonPress-1>", self._on_line_start)
        self.canvas.bind("<B1-Motion>", self._on_line_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_line_release)

    def _on_line_start(self, event):
        x0 = self._clamp_x(event.x); y0 = self._clamp_y(event.y)
        self._line_drag_start = (x0, y0)
        if self._line_rubberband_id is not None:
            self.canvas.delete(self._line_rubberband_id); self._line_rubberband_id = None
        cx, cy = self._to_canvas(x0, y0)
        self._line_rubberband_id = self.canvas.create_line(cx, cy, cx, cy, fill=VLINE_COLOR, width=2)

    def _on_line_drag(self, event):
        if not self._line_drag_start or self._line_rubberband_id is None: return
        x0, y0 = self._line_drag_start
        y1 = self._clamp_y(event.y)
        self.canvas.coords(self._line_rubberband_id,
                           *self._to_canvas(x0, y0), *self._to_canvas(x0, y1))

    def _on_line_release(self, event):
        if not self._line_drag_start: return
        x0, y0 = self._line_drag_start; y1 = self._clamp_y(event.y)
        self._line_drag_start = None
        self.vline_coords = (x0, y0, y1)
        self.vline_length_px = abs(y1 - y0)
        self.line_length_var.set(f"Line length: {self.vline_length_px} px")
        if self._line_rubberband_id is not None:
            self.canvas.delete(self._line_rubberband_id); self._line_rubberband_id = None
        self._unbind_all_draw(); self.suspend_video = False
        self._set_drawing(None)

    def clear_line(self):
        self.vline_coords = None; self.vline_length_px = 0
        self.line_length_var.set("No line drawn")

    def _on_hline_len(self, value):
        # Snap to even values so the line stays centred on the vertical line.
        length = int(round(float(value) / 2)) * 2
        length = max(HLINE_LEN_MIN, min(HLINE_LEN_MAX, length))
        self.hline_len_px = length
        self.hline_len_label.set(f"{length} px")

    def _unbind_all_draw(self):
        for seq in ("<ButtonPress-1>", "<B1-Motion>", "<ButtonRelease-1>"):
            self.canvas.unbind(seq)

    def change_temp_range(self, event=None):
        selected = self.temp_range_var.get()
        if selected in TEMP_RANGES:
            gain_name = TEMP_RANGES[selected]
            if not CAMERA_LOCK.acquire(timeout=2):
                print("Failed to set temperature range: camera busy or not responding")
                return
            try:
                lep.sys.SetGainMode(getattr(CCI.Sys.GainMode, gain_name))
                actual = lep.sys.GetGainMode()
                print(f"Temperature range set to: {selected} (gain mode {actual})")
            except Exception as e:
                print(f"Failed to set temperature range: {e}")
            finally:
                CAMERA_LOCK.release()

    # ----- Horizontal ROI helpers -----
    def _horizontal_rois_from_vline(self):
        """
        Returns y positions for 5 horizontal lines:
        - 1/5, 2/5, 3/5, 4/5 along the vertical segment
        - plus a 5th at the TOP endpoint of the segment   <-- moved here
        Also returns (x_left, x_right) for the 20 px span centered at vertical x.
        """
        if self.vline_coords is None:
            return [], None, None

        x0, y0, y1 = self.vline_coords
        y_top, y_bot = (y0, y1) if y0 <= y1 else (y1, y0)
        seg_len = y_bot - y_top
        if seg_len < 1:
            return [], None, None

        ys = []
        for k in (1, 2, 3, 4):
            yk = int(round(y_top + k * seg_len / 5.0))
            ys.append(yk)

        # Previously used y_bot (bottom). Now place the 5th line at the TOP endpoint.
        ys.append(int(y_top))  # 5th line at the top endpoint

        # Clamp to canvas bounds
        ys = [max(0, min(self.heightScaled - 1, yk)) for yk in ys]

        half_len = self.hline_len_px // 2  # set by the "Horizontal line length" slider
        x_left = max(0, x0 - half_len)
        x_right = min(self.widthScaled - 1, x0 + half_len)
        return ys, x_left, x_right

    # Frame loop
    def update_video(self):
        if not self.running: return
        if self.suspend_video:
            self.after_id = self.root.after(FRAME_DELAY_MS, self.update_video); return

        latest = incoming_frames[-1] if incoming_frames else None
        if latest is not None and latest is self._last_frame:
            # The camera delivers ~9 fps; don't reprocess/redraw the same frame.
            self.after_id = self.root.after(FRAME_DELAY_MS, self.update_video); return
        self._last_frame = latest
        if latest is not None:
            self._frame_times.append(time.time())

        if latest is not None:
            h, w, f = latest
            arr_ck = short_array_to_numpy(h, w, f)
            arr_ck_resized = cv.resize(arr_ck, self.dim, interpolation=cv.INTER_NEAREST)
            arr_c = centikelvin_to_celsius(arr_ck_resized).astype(np.float32)
            vis_8u = cv.normalize(arr_c, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
            frame_bgr = cv.applyColorMap(vis_8u, cv.COLORMAP_PLASMA)
        else:
            frame_bgr = np.zeros((self.heightScaled, self.widthScaled, 3), np.uint8)
            arr_c = np.zeros((self.heightScaled, self.widthScaled), dtype=np.float32)

        contours = []
        if self.user_mask is not None and np.any(self.user_mask):
            gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
            masked_gray = gray.copy(); masked_gray[self.user_mask == 0] = 0
            blur = cv.GaussianBlur(masked_gray, (5, 5), 0)
            _, thresh = cv.threshold(blur, THRESH_8U, 255, cv.THRESH_BINARY)
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
            thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
            thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
            contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            outside = (self.user_mask == 0)
            frame_bgr[outside] = (frame_bgr[outside] * 0.4).astype(np.uint8)

        mask = np.zeros((self.heightScaled, self.widthScaled), np.uint8)
        avg_temp_C = 0.0; max_temp_C = 0.0; roi_area_px2 = 0
        largest = None
        if contours:
            largest = max(contours, key=cv.contourArea)
            cv.drawContours(mask, [largest], -1, 1, -1)
            roi_vals = arr_c[mask == 1]
            if roi_vals.size:
                avg_temp_C = float(np.mean(roi_vals))
                max_temp_C = float(np.max(roi_vals))
                roi_area_px2 = int(roi_vals.size)

        # Horizontal ROI averages: 5 lines now
        hline_avgs = [np.nan]*5
        if self.vline_coords is not None:
            ys, x_left, x_right = self._horizontal_rois_from_vline()
            if ys and x_left is not None:
                for i, yk in enumerate(ys):
                    seg = arr_c[yk, x_left:x_right+1]
                    if seg.size > 0:
                        hline_avgs[i] = float(np.mean(seg))

        if contours:
            self.reading_vars["avg"].set(f"{avg_temp_C:.1f}")
            self.reading_vars["max"].set(f"{max_temp_C:.1f}")
            self.reading_vars["area"].set(f"{roi_area_px2}")
        else:
            for var in self.reading_vars.values():
                var.set("—")

        # Recording row
        if self.recording:
            elapsed_rec = time.time() - self.recording_start_time
            row = [elapsed_rec, avg_temp_C, max_temp_C, roi_area_px2] + hline_avgs
            self.data_records.append(row)

        # Plots: horizontal-line temperatures (top) and edge-ROI max (bottom)
        elapsed_plot = time.time() - self.stream_start_time
        self.time_data.append(elapsed_plot)
        self.max_series.append(max_temp_C)
        for series, val in zip(self.hline_series, hline_avgs):
            series.append(val)

        t = np.asarray(self.time_data)
        visible = t >= elapsed_plot - PLOT_WINDOW_S
        xlim = (max(0, elapsed_plot - PLOT_WINDOW_S), elapsed_plot + 1)

        hl = np.array([np.asarray(s) for s in self.hline_series])  # NaN = no line defined
        for plot, vals in zip(self.hline_plots, hl):
            plot.set_data(t, vals)
        self.ax_lines.set_xlim(*xlim)
        self.ax_lines.set_ylim(*self._padded_ylim(hl[:, visible]))
        self.canvas_lines.draw()

        mx = np.asarray(self.max_series)
        self.line_max.set_data(t, mx)
        self.ax_max.set_xlim(*xlim)
        self.ax_max.set_ylim(*self._padded_ylim(mx[visible]))
        self.canvas_max.draw()

        self._last_draw = (frame_bgr, largest, hline_avgs)
        self._draw_video()
        self._update_status(latest is not None)
        self.after_id = self.root.after(FRAME_DELAY_MS, self.update_video)

    def _draw_video(self):
        """Paint the last processed frame and overlays. Called per new frame and on
        resize; it never touches recording or plot history, so resizing adds no rows."""
        if self._last_draw is None:
            return
        frame_bgr, largest, hline_avgs = self._last_draw
        # Processing happens at widthScaled x heightScaled; the display is scaled
        # by self.zoom and centred (self.ox, self.oy) to fit the canvas.
        disp_w = max(1, int(round(self.widthScaled * self.zoom)))
        disp_h = max(1, int(round(self.heightScaled * self.zoom)))
        display = cv.resize(frame_bgr, (disp_w, disp_h), interpolation=cv.INTER_LINEAR)
        rgb = cv.cvtColor(display, cv.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        self.canvas.create_image(int(self.ox), int(self.oy), anchor=tk.NW, image=imgtk)
        self.canvas.imgtk = imgtk

        if largest is not None and len(largest) > 1:
            pts = (largest.reshape(-1, 2) + 0.5) * self.zoom + (self.ox, self.oy)
            self.canvas.create_polygon(*pts.ravel().tolist(), outline=CONTOUR_COLOR, fill="", width=2)

        if self.mask_bounds is not None:
            xL, yT, xR, yB = self.mask_bounds
            self.canvas.create_rectangle(*self._to_canvas(xL, yT), *self._to_canvas(xR, yB),
                                         outline=MASK_COLOR, width=2)

        if self.vline_coords is not None:
            x0, y0, y1 = self.vline_coords
            self.canvas.create_line(*self._to_canvas(x0, y0), *self._to_canvas(x0, y1),
                                    fill=VLINE_COLOR, width=1, dash=(4, 4))

            # Each horizontal line uses the same color as its trace in the plot.
            ys, x_left, x_right = self._horizontal_rois_from_vline()
            right_edge = self.ox + disp_w
            for i, yk in enumerate(ys):
                color = LINE_COLORS[i]
                (lx, cy), (rx, _) = self._to_canvas(x_left, yk), self._to_canvas(x_right, yk)
                self.canvas.create_line(lx, cy, rx, cy, fill=color, width=3)
                label = "" if np.isnan(hline_avgs[i]) else f"{hline_avgs[i]:.1f}°C"
                self._canvas_label(min(rx + 8, right_edge - 52), cy, label, color)

    def _canvas_label(self, x, y, text, color):
        # Text with a dark outline so it stays readable on any part of the colormap.
        if not text:
            return
        font = (UI_FONT, 9, "bold")
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            self.canvas.create_text(x + dx, y + dy, text=text, fill="#000000", anchor="w", font=font)
        self.canvas.create_text(x, y, text=text, fill=color, anchor="w", font=font)

    def toggle_recording(self):
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()
        self._refresh_record_button()

    def start_recording(self):
        self.data_records = []
        self.recording_start_time = time.time()
        self.recording = True
        print("Recording started")

    def stop_recording(self):
        if not self.recording or not self.data_records:
            print("No data recorded"); return
        self.recording = False
        path = filedialog.asksaveasfilename(defaultextension='.csv',
                                            filetypes=[('CSV', '*.csv')],
                                            title='Save ROI Data As')
        if path:
            import csv
            header = [
                'elapsed_s',
                'edgeROI_avg_temp_C',
                'edgeROI_max_temp_C',
                'edgeROI_area_px2',
                'line1_avg_C',
                'line2_avg_C',
                'line3_avg_C',
                'line4_avg_C',
                'baseline_avg_C'
            ]
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(self.data_records)
            print(f"Data saved to {path}")
            self.record_info_var.set(f"Saved {len(self.data_records)} samples to {os.path.basename(path)}")
        else:
            self.record_info_var.set("Recording discarded")
        self.data_records = []

    def quit_app(self):
        self.running = False
        try:
            capture.StopGraph()
        except Exception as e:
            print("Error stopping capture:", e)
        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except Exception as e:
                print("Error cancelling after callback:", e)
        self.root.quit(); self.root.destroy(); sys.exit(0)

    @staticmethod
    def _padded_ylim(values):
        finite = np.asarray(values)[np.isfinite(values)]
        if finite.size == 0:
            return 0.0, 1.0
        lo, hi = float(finite.min()), float(finite.max())
        pad = max(0.5, 0.05 * (hi - lo))
        return lo - pad, hi + pad

    # ----- Display transform (image px <-> canvas px) -----
    def _on_canvas_resize(self, event):
        # Fit the image inside the canvas, keeping its aspect ratio, and centre it.
        self.zoom = max(0.1, min(event.width / self.widthScaled, event.height / self.heightScaled))
        self.ox = (event.width - self.widthScaled * self.zoom) / 2
        self.oy = (event.height - self.heightScaled * self.zoom) / 2
        self._draw_video()  # repaint at the new size right away

    def _to_canvas(self, x, y):
        return x * self.zoom + self.ox, y * self.zoom + self.oy

    # Convert canvas (display) coordinates to processing-image coordinates.
    def _clamp_x(self, x): return max(0, min(self.widthScaled - 1, int((x - self.ox) / self.zoom)))
    def _clamp_y(self, y): return max(0, min(self.heightScaled - 1, int((y - self.oy) / self.zoom)))

if __name__ == "__main__":
    root = tk.Tk()
    app = ThermalVideoApp(root)
    root.mainloop()
