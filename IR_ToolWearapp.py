# IR_ToolWearapp_masked_threshold_vline.py
# Purpose: Live Lepton thermal view with user-defined rectangular mask;
#          high fixed threshold applied ONLY within mask to find ROI;
#          avg/max ROI temps plotted; CSV logging; vertical line measurement.

import sys, os, time, platform
from collections import deque
import cv2 as cv
import numpy as np

from comtypes import CoInitialize
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# =========================
# Configuration
# =========================
SCALE_PERCENT   = 200           # Display upscaling for the thermal frame (percent)
FRAME_DELAY_MS  = int(1000/60)  # ~60 FPS UI update target
THRESH_8U       = 110
# Relatively high 8-bit threshold (applied only inside user mask)

# Camera temperature ranges (centikelvin) exposed via dropdown
TEMP_RANGES = {
    "Low (-10°C to 140°C)":  (27315 - 10 * 100,  27315 + 140 * 100),
    "High (-10°C to 400°C)": (27315 - 10 * 100,  27315 + 400 * 100),
}

# =========================
# COM & Lepton SDK init
# =========================
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
lep.sys.SetGainMode(CCI.Sys.GainMode.LOW)  # LOW gain for wider span

clr.AddReference("ManagedIR16Filters")
from IR16Filters import IR16Capture, NewBytesFrameEvent

incoming_frames = deque(maxlen=10)

def got_a_frame(frame, width, height):
    """Capture callback: store frames as (h, w, iterable_of_uint16)."""
    incoming_frames.append((height, width, frame))

capture = IR16Capture()
capture.SetupGraphWithBytesCallback(NewBytesFrameEvent(got_a_frame))
capture.RunGraph()
time.sleep(3)  # warm-up

# =========================
# Helpers
# =========================
def short_array_to_numpy(h, w, frame_iterable):
    """Convert incoming 16-bit centikelvin buffer to (H, W) uint16 array."""
    return np.fromiter(frame_iterable, dtype=np.uint16).reshape(h, w)

def centikelvin_to_celsius(t_ck):
    """Convert centikelvin to Celsius."""
    return (t_ck - 27315) / 100.0

# =========================
# Main App
# =========================
class ThermalVideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Thermal Camera Viewer – Masked Threshold ROI + Vertical Line")
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)

        # State
        self.running = True
        self.suspend_video = False     # Pause redraw while drawing mask/line
        self.recording = False
        self.recording_start_time = None
        self.stream_start_time = time.time()
        self.data_records = []         # [elapsed_s, avg_temp_C, max_temp_C, roi_area_px2]
        self.after_id = None

        # Mask state (rectangle)
        self.user_mask = None          # uint8 mask same size as display (0 outside, 1 inside)
        self.mask_bounds = None        # (x0, y0, x1, y1)
        self._drag_start = None        # for mask draw
        self._rubberband_id = None

        # Vertical line measurement state
        self.vline_coords = None       # (x, y0, y1) in display pixels
        self.vline_length_px = 0
        self._line_drag_start = None   # (x0, y0) start for line draw
        self._line_rubberband_id = None

        # Canvas
        self.canvas = tk.Canvas(root)
        self.canvas.grid(row=0, column=0, padx=5, pady=5)

        # Initial dims from preview
        if incoming_frames:
            h, w, f = incoming_frames[-1]
            preview = short_array_to_numpy(h, w, f)
        else:
            preview = np.zeros((240, 320), np.uint16)

        self.widthScaled  = int(preview.shape[1] * SCALE_PERCENT / 100)
        self.heightScaled = int(preview.shape[0] * SCALE_PERCENT / 100)
        self.dim = (self.widthScaled, self.heightScaled)

        # Init empty mask for display size
        self.user_mask = np.zeros((self.heightScaled, self.widthScaled), np.uint8)

        # Plots
        self.time_data, self.avg_series, self.max_series = [], [], []

        self.fig_avg, self.ax_avg = plt.subplots(figsize=(4, 2.5))
        self.line_avg, = self.ax_avg.plot([], [], label='Avg Temp (°C)')
        self.ax_avg.set_title("Average ROI Temperature")
        self.ax_avg.set_xlabel("Time (s)")
        self.ax_avg.set_ylabel("Temp (°C)")
        self.ax_avg.grid(True)
        self.ax_avg.legend(loc='upper right')
        self.canvas_avg = FigureCanvasTkAgg(self.fig_avg, master=root)
        self.canvas_avg.get_tk_widget().grid(row=0, column=1, padx=5, pady=5)

        self.fig_max, self.ax_max = plt.subplots(figsize=(4, 2.5))
        self.line_max, = self.ax_max.plot([], [], label='Max Temp (°C)')
        self.ax_max.set_title("Maximum ROI Temperature")
        self.ax_max.set_xlabel("Time (s)")
        self.ax_max.set_ylabel("Temp (°C)")
        self.ax_max.grid(True)
        self.ax_max.legend(loc='upper right')
        self.canvas_max = FigureCanvasTkAgg(self.fig_max, master=root)
        self.canvas_max.get_tk_widget().grid(row=1, column=1, padx=5, pady=5)

        # Controls
        control = ttk.Frame(root)
        control.grid(row=2, column=0, columnspan=2, pady=5)

        ttk.Button(control, text="Define Mask", command=self.enable_mask_draw).pack(side="left", padx=5)
        ttk.Button(control, text="Clear Mask", command=self.clear_mask).pack(side="left", padx=5)

        ttk.Button(control, text="Define Vertical Line", command=self.enable_vertical_line).pack(side="left", padx=10)
        ttk.Button(control, text="Clear Line", command=self.clear_line).pack(side="left", padx=5)
        ttk.Label(control, text="Line length:").pack(side="left", padx=(12, 4))
        self.line_length_var = tk.StringVar(value="—")
        ttk.Label(control, textvariable=self.line_length_var).pack(side="left", padx=(0, 8))

        ttk.Button(control, text="Start Recording", command=self.start_recording).pack(side="left", padx=5)
        ttk.Button(control, text="Stop and Save CSV", command=self.stop_recording).pack(side="left", padx=5)
        ttk.Button(control, text="Quit", command=self.quit_app).pack(side="left", padx=5)

        # Temperature span control (device)
        self.temp_range_var = tk.StringVar(value="Low (-10°C to 140°C)")
        ttk.Label(control, text="Temp Range:").pack(side="left", padx=5)
        self.temp_range_dropdown = ttk.Combobox(
            control, textvariable=self.temp_range_var, state="readonly",
            values=list(TEMP_RANGES.keys()), width=22
        )
        self.temp_range_dropdown.pack(side="left", padx=5)
        self.temp_range_dropdown.bind("<<ComboboxSelected>>", self.change_temp_range)
        self.change_temp_range()

        # Start loop
        self.update_video()

    # ------------------------------------------------
    # Mask drawing (rectangle)
    # ------------------------------------------------
    def enable_mask_draw(self):
        """Enable click-and-drag rectangle drawing on the video to define a mask."""
        self._unbind_all_draw()
        self.suspend_video = True
        self.canvas.bind("<ButtonPress-1>", self._on_mask_start)
        self.canvas.bind("<B1-Motion>", self._on_mask_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mask_release)

    def _on_mask_start(self, event):
        self._drag_start = (self._clamp_x(event.x), self._clamp_y(event.y))
        if self._rubberband_id is not None:
            self.canvas.delete(self._rubberband_id)
            self._rubberband_id = None

    def _on_mask_drag(self, event):
        if not self._drag_start:
            return
        x0, y0 = self._drag_start
        x1, y1 = self._clamp_x(event.x), self._clamp_y(event.y)
        if self._rubberband_id is None:
            self._rubberband_id = self.canvas.create_rectangle(
                x0, y0, x1, y1, outline="yellow", width=2, dash=(4, 3)
            )
        else:
            self.canvas.coords(self._rubberband_id, x0, y0, x1, y1)

    def _on_mask_release(self, event):
        if not self._drag_start:
            return
        x0, y0 = self._drag_start
        x1, y1 = self._clamp_x(event.x), self._clamp_y(event.y)
        self._drag_start = None

        xL, xR = sorted((x0, x1))
        yT, yB = sorted((y0, y1))

        self.user_mask.fill(0)
        self.user_mask[yT:yB, xL:xR] = 1
        self.mask_bounds = (xL, yT, xR, yB)

        if self._rubberband_id is not None:
            self.canvas.delete(self._rubberband_id); self._rubberband_id = None

        self._unbind_all_draw()
        self.suspend_video = False

    def clear_mask(self):
        self.user_mask.fill(0)
        self.mask_bounds = None

    # ------------------------------------------------
    # Vertical line measurement
    # ------------------------------------------------
    def enable_vertical_line(self):
        """Enable click-and-drag to place a vertical line. Length is |y1 - y0| in pixels."""
        self._unbind_all_draw()
        self.suspend_video = True
        self.canvas.bind("<ButtonPress-1>", self._on_line_start)
        self.canvas.bind("<B1-Motion>", self._on_line_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_line_release)

    def _on_line_start(self, event):
        x0 = self._clamp_x(event.x)
        y0 = self._clamp_y(event.y)
        self._line_drag_start = (x0, y0)
        if self._line_rubberband_id is not None:
            self.canvas.delete(self._line_rubberband_id)
            self._line_rubberband_id = None
        # Initialize with a 1px segment so user sees it
        self._line_rubberband_id = self.canvas.create_line(x0, y0, x0, y0, fill="cyan", width=2)

    def _on_line_drag(self, event):
        if not self._line_drag_start or self._line_rubberband_id is None:
            return
        x0, y0 = self._line_drag_start
        y1 = self._clamp_y(event.y)
        # Keep line vertical at x0
        self.canvas.coords(self._line_rubberband_id, x0, y0, x0, y1)

    def _on_line_release(self, event):
        if not self._line_drag_start:
            return
        x0, y0 = self._line_drag_start
        y1 = self._clamp_y(event.y)
        self._line_drag_start = None

        # Store persistent coords (vertical line)
        self.vline_coords = (x0, y0, y1)
        self.vline_length_px = abs(y1 - y0)
        self.line_length_var.set(f"{self.vline_length_px} px")

        # Remove rubberband
        if self._line_rubberband_id is not None:
            self.canvas.delete(self._line_rubberband_id); self._line_rubberband_id = None

        self._unbind_all_draw()
        self.suspend_video = False

    def clear_line(self):
        self.vline_coords = None
        self.vline_length_px = 0
        self.line_length_var.set("—")

    def _unbind_all_draw(self):
        """Remove any draw-mode bindings to avoid conflicts between tools."""
        for seq in ("<ButtonPress-1>", "<B1-Motion>", "<ButtonRelease-1>"):
            self.canvas.unbind(seq)

    # ------------------------------------------------
    # Device temperature span
    # ------------------------------------------------
    def change_temp_range(self, event=None):
        selected = self.temp_range_var.get()
        if selected in TEMP_RANGES:
            low_ck, high_ck = TEMP_RANGES[selected]
            try:
                lep.sys.SetTempRangeHighLow(high_ck, low_ck)
                print(f"Temperature range set to: {selected}")
            except Exception as e:
                print(f"Failed to set temperature range: {e}")

    # ------------------------------------------------
    # Frame loop
    # ------------------------------------------------
    def update_video(self):
        if not self.running:
            return
        if self.suspend_video:
            self.after_id = self.root.after(FRAME_DELAY_MS, self.update_video)
            return

        if incoming_frames:
            h, w, f = incoming_frames[-1]
            arr_ck = short_array_to_numpy(h, w, f)
            arr_ck_resized = cv.resize(arr_ck, self.dim, interpolation=cv.INTER_NEAREST)
            arr_c = centikelvin_to_celsius(arr_ck_resized).astype(np.float32)
            vis_8u = cv.normalize(arr_c, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
            frame_bgr = cv.applyColorMap(vis_8u, cv.COLORMAP_PLASMA)
        else:
            frame_bgr = np.zeros((self.heightScaled, self.widthScaled, 3), np.uint8)
            arr_c = np.zeros((self.heightScaled, self.widthScaled), dtype=np.float32)

        # Threshold inside mask only
        contours = []
        if self.user_mask is not None and np.any(self.user_mask):
            gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
            masked_gray = gray.copy()
            masked_gray[self.user_mask == 0] = 0
            blur = cv.GaussianBlur(masked_gray, (5, 5), 0)
            _, thresh = cv.threshold(blur, THRESH_8U, 255, cv.THRESH_BINARY)
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
            thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
            thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
            contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            # Dim outside masked area for clarity
            outside = (self.user_mask == 0)
            frame_bgr[outside] = (frame_bgr[outside] * 0.4).astype(np.uint8)

        # ROI stats from largest contour
        mask = np.zeros((self.heightScaled, self.widthScaled), np.uint8)
        avg_temp_C = 0.0
        max_temp_C = 0.0
        roi_area_px2 = 0

        if contours:
            largest = max(contours, key=cv.contourArea)
            cv.drawContours(frame_bgr, [largest], -1, (0, 255, 0), 2)
            cv.drawContours(mask, [largest], -1, 1, -1)
            roi_vals = arr_c[mask == 1]
            if roi_vals.size:
                avg_temp_C = float(np.mean(roi_vals))
                max_temp_C = float(np.max(roi_vals))
                roi_area_px2 = int(roi_vals.size)

        # Recording
        if self.recording:
            elapsed_rec = time.time() - self.recording_start_time
            self.data_records.append([elapsed_rec, avg_temp_C, max_temp_C, roi_area_px2])

        # Plot updates
        elapsed_plot = time.time() - self.stream_start_time
        self.time_data.append(elapsed_plot)
        self.avg_series.append(avg_temp_C)
        self.max_series.append(max_temp_C)

        self.line_avg.set_data(self.time_data, self.avg_series)
        self.ax_avg.set_xlim(max(0, elapsed_plot - 10), elapsed_plot + 1)
        last_avg = self.avg_series[-60:]
        ymin_a = min(last_avg) if last_avg else 0
        ymax_a = max(last_avg) if last_avg else 1
        pad_a = max(0.5, 0.05 * (ymax_a - ymin_a))
        self.ax_avg.set_ylim(ymin_a - pad_a, ymax_a + pad_a)
        self.canvas_avg.draw()

        self.line_max.set_data(self.time_data, self.max_series)
        self.ax_max.set_xlim(max(0, elapsed_plot - 10), elapsed_plot + 1)
        last_max = self.max_series[-60:]
        ymin_m = min(last_max) if last_max else 0
        ymax_m = max(last_max) if last_max else 1
        pad_m = max(0.5, 0.05 * (ymax_m - ymin_m))
        self.ax_max.set_ylim(ymin_m - pad_m, ymax_m + pad_m)
        self.canvas_max.draw()

        # Draw on Tk canvas
        rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        self.canvas.config(width=self.widthScaled, height=self.heightScaled)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas.imgtk = imgtk  # keep ref

        # Persistent overlays
        if self.mask_bounds is not None:
            xL, yT, xR, yB = self.mask_bounds
            self.canvas.create_rectangle(xL, yT, xR, yB, outline="yellow", width=2)

        if self.vline_coords is not None:
            x0, y0, y1 = self.vline_coords
            self.canvas.create_line(x0, y0, x0, y1, fill="cyan", width=2)
            # Label near the midpoint
            y_mid = int((y0 + y1) / 2)
            self.canvas.create_text(
                min(x0 + 8, self.widthScaled - 10),
                y_mid,
                text=f"{self.vline_length_px} px",
                fill="white",
                anchor="w",
                font=("TkDefaultFont", 10, "bold")
            )

        # Next frame
        self.after_id = self.root.after(FRAME_DELAY_MS, self.update_video)

    # ------------------------------------------------
    # Recording controls
    # ------------------------------------------------
    def start_recording(self):
        self.data_records = []
        self.recording_start_time = time.time()
        self.recording = True
        print("Recording started")

    def stop_recording(self):
        if not self.recording or not self.data_records:
            print("No data recorded")
            return
        self.recording = False
        path = filedialog.asksaveasfilename(
            defaultextension='.csv',
            filetypes=[('CSV', '*.csv')],
            title='Save ROI Data As'
        )
        if path:
            import csv
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['elapsed_s', 'avg_temp_C', 'max_temp_C', 'roi_area_px2'])
                writer.writerows(self.data_records)
            print(f"Data saved to {path}")
        self.data_records = []

    # ------------------------------------------------
    # Shutdown
    # ------------------------------------------------
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
        self.root.quit()
        self.root.destroy()
        sys.exit(0)

    # ------------------------------------------------
    # Misc
    # ------------------------------------------------
    def _clamp_x(self, x): return max(0, min(self.widthScaled - 1, x))
    def _clamp_y(self, y): return max(0, min(self.heightScaled - 1, y))

# =========================
# Entrypoint
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    app = ThermalVideoApp(root)
    root.mainloop()
