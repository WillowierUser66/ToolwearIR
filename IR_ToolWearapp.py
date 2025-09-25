# IR_ToolWearapp_masked_threshold.py
# Purpose: Display Lepton thermal video, let the user define a rectangular mask by drag-and-drop
#          on the live view, then run a relatively high fixed threshold ONLY inside that mask to
#          trace edges (contours). Plots average and max ROI temperatures over time and can save CSV.

import sys
import os
import time
import platform
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
# Configuration constants
# =========================
SCALE_PERCENT = 200                      # Display upscaling for the thermal frame in percent
FRAME_DELAY_MS = int(1000 / 60)          # About 60 FPS UI update target
THRESH_8U = 110                          # Relatively high 8-bit threshold applied within mask only

# Camera temperature ranges in centikelvin exposed via dropdown
TEMP_RANGES = {
    "Low (-10°C to 140°C)":  (27315 - 10 * 100,  27315 + 140 * 100),
    "High (-10°C to 400°C)": (27315 - 10 * 100,  27315 + 400 * 100),
}

# =========================
# COM and Lepton SDK init
# =========================
# Requires FLIR Lepton and PureThermal board with .NET assemblies available
CoInitialize()

bits, _ = platform.architecture()
folder = ["x64"] if bits == "64bit" else ["x86"]
# Ensure the architecture specific folder for the managed DLLs is on sys.path
sys.path.append(os.path.join("", *folder))

import clr
clr.AddReference("LeptonUVC")
from Lepton import CCI

# Find the PureThermal device and open a session
found_device = next((d for d in CCI.GetDevices() if d.Name.startswith("PureThermal")), None)
if not found_device:
    print("Could not find Lepton device")
    sys.exit(1)

lep = found_device.Open()
# Use LOW gain for a wider temperature span
lep.sys.SetGainMode(CCI.Sys.GainMode.LOW)

# IR16 capture graph with Python callback that receives 16-bit frames
clr.AddReference("ManagedIR16Filters")
from IR16Filters import IR16Capture, NewBytesFrameEvent

incoming_frames = deque(maxlen=10)  # small buffer to smooth occasional callback jitter

def got_a_frame(frame, width, height):
    """Callback from the capture graph. Stores frames as (h, w, iterable_of_uint16)."""
    incoming_frames.append((height, width, frame))

capture = IR16Capture()
capture.SetupGraphWithBytesCallback(NewBytesFrameEvent(got_a_frame))
capture.RunGraph()
time.sleep(3)  # give the graph a moment to warm up before UI starts

# =========================
# Utility functions
# =========================
def short_array_to_numpy(height, width, frame_iterable):
    """Convert incoming 16-bit centikelvin buffer to a (H, W) uint16 NumPy array."""
    return np.fromiter(frame_iterable, dtype=np.uint16).reshape(height, width)

def centikelvin_to_celsius(t_ck):
    """Convert centikelvin to Celsius."""
    return (t_ck - 27315) / 100.0

# =========================
# Main Application
# =========================
class ThermalVideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Thermal Camera Viewer – Masked Threshold ROI")
        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)

        # App state
        self.running = True
        self.suspend_video = False       # Suspends redraw while drawing the mask
        self.recording = False
        self.recording_start_time = None
        self.stream_start_time = time.time()
        self.data_records = []           # rows: [elapsed_s, avg_temp_C, max_temp_C, roi_area_px2]
        self.after_id = None

        # Mask drawing state
        self.user_mask = None            # np.uint8 mask same size as display (0 outside, 1 inside)
        self.mask_bounds = None          # (x0, y0, x1, y1) for overlay drawing
        self._drag_start = None          # (x, y) start of drag on canvas
        self._rubberband_id = None       # Canvas rectangle id during drawing

        # Canvas for thermal image
        self.canvas = tk.Canvas(root)
        self.canvas.grid(row=0, column=0, padx=5, pady=5)

        # Establish display dimensions from a preview frame (fallback to 320x240)
        if incoming_frames:
            h, w, f = incoming_frames[-1]
            preview = short_array_to_numpy(h, w, f)
        else:
            preview = np.zeros((240, 320), np.uint16)

        self.widthScaled  = int(preview.shape[1] * SCALE_PERCENT / 100)
        self.heightScaled = int(preview.shape[0] * SCALE_PERCENT / 100)
        self.dim = (self.widthScaled, self.heightScaled)

        # Initialize empty mask sized to display
        self.user_mask = np.zeros((self.heightScaled, self.widthScaled), np.uint8)

        # Plots for Average and Maximum ROI Temperature
        self.time_data, self.avg_series, self.max_series = [], [], []

        # Average temperature plot
        self.fig_avg, self.ax_avg = plt.subplots(figsize=(4, 2.5))
        self.line_avg, = self.ax_avg.plot([], [], label='Avg Temp (°C)')
        self.ax_avg.set_title("Average ROI Temperature")
        self.ax_avg.set_xlabel("Time (s)")
        self.ax_avg.set_ylabel("Temp (°C)")
        self.ax_avg.grid(True)
        self.ax_avg.legend(loc='upper right')
        self.canvas_avg = FigureCanvasTkAgg(self.fig_avg, master=root)
        self.canvas_avg.get_tk_widget().grid(row=0, column=1, padx=5, pady=5)

        # Maximum temperature plot
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
        control_frame = ttk.Frame(root)
        control_frame.grid(row=2, column=0, columnspan=2, pady=5)

        ttk.Button(control_frame, text="Define Mask", command=self.enable_mask_draw).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Clear Mask", command=self.clear_mask).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Start Recording", command=self.start_recording).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Stop and Save CSV", command=self.stop_recording).pack(side="left", padx=5)
        ttk.Button(control_frame, text="Quit", command=self.quit_app).pack(side="left", padx=5)

        # Temperature span control sent to camera
        self.temp_range_var = tk.StringVar(value="Low (-10°C to 140°C)")
        ttk.Label(control_frame, text="Temp Range:").pack(side="left", padx=5)
        self.temp_range_dropdown = ttk.Combobox(
            control_frame,
            textvariable=self.temp_range_var,
            state="readonly",
            values=list(TEMP_RANGES.keys()),
            width=22
        )
        self.temp_range_dropdown.pack(side="left", padx=5)
        self.temp_range_dropdown.bind("<<ComboboxSelected>>", self.change_temp_range)
        self.change_temp_range()  # initialize device span

        # Start UI loop
        self.update_video()

    # -------------------------
    # Mask drawing handlers
    # -------------------------
    def enable_mask_draw(self):
        """Enable click-and-drag rectangle drawing on the video to define a mask."""
        self.suspend_video = True  # pause the regular redraw to avoid flicker while drawing
        # Bind mouse events on the canvas
        self.canvas.bind("<ButtonPress-1>", self._on_mask_start)
        self.canvas.bind("<B1-Motion>", self._on_mask_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mask_release)

    def _on_mask_start(self, event):
        self._drag_start = (self._clamp_x(event.x), self._clamp_y(event.y))
        # Remove any previous rubberband rectangle
        if self._rubberband_id is not None:
            self.canvas.delete(self._rubberband_id)
            self._rubberband_id = None

    def _on_mask_drag(self, event):
        if not self._drag_start:
            return
        x0, y0 = self._drag_start
        x1, y1 = self._clamp_x(event.x), self._clamp_y(event.y)
        # Draw or update a rubberband rectangle overlay while dragging
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

        # Normalize coords to top-left -> bottom-right
        xL, xR = sorted((x0, x1))
        yT, yB = sorted((y0, y1))

        # Build binary mask (1 inside rect, 0 outside)
        self.user_mask.fill(0)
        self.user_mask[yT:yB, xL:xR] = 1
        self.mask_bounds = (xL, yT, xR, yB)

        # Remove temporary rubberband
        if self._rubberband_id is not None:
            self.canvas.delete(self._rubberband_id)
            self._rubberband_id = None

        # Unbind drawing handlers and resume video updates
        self.canvas.unbind("<ButtonPress-1>")
        self.canvas.unbind("<B1-Motion>")
        self.canvas.unbind("<ButtonRelease-1>")
        self.suspend_video = False

    def clear_mask(self):
        """Clear the user-defined mask so no region is selected."""
        self.user_mask.fill(0)
        self.mask_bounds = None

    def _clamp_x(self, x):
        return max(0, min(self.widthScaled - 1, x))

    def _clamp_y(self, y):
        return max(0, min(self.heightScaled - 1, y))

    # -------------------------
    # Device span change
    # -------------------------
    def change_temp_range(self, event=None):
        """Apply the selected temperature span to the Lepton device."""
        selected = self.temp_range_var.get()
        if selected in TEMP_RANGES:
            low_ck, high_ck = TEMP_RANGES[selected]
            try:
                lep.sys.SetTempRangeHighLow(high_ck, low_ck)
                print(f"Temperature range set to: {selected}")
            except Exception as e:
                print(f"Failed to set temperature range: {e}")

    # -------------------------
    # Frame processing loop
    # -------------------------
    def update_video(self):
        """Fetch frame, threshold only within user mask, update plots, optionally record metrics."""
        # If drawing mask, skip frame redraw but keep scheduling
        if not self.running:
            return
        if self.suspend_video:
            self.after_id = self.root.after(FRAME_DELAY_MS, self.update_video)
            return

        # Acquire latest frame or a black placeholder
        if incoming_frames:
            h, w, f = incoming_frames[-1]
            arr_ck = short_array_to_numpy(h, w, f)
            # Resize first in CK to minimize rounding, then convert to Celsius
            arr_ck_resized = cv.resize(arr_ck, self.dim, interpolation=cv.INTER_NEAREST)
            arr_c = centikelvin_to_celsius(arr_ck_resized).astype(np.float32)
            # Build visualization for humans (note: normalized per-frame)
            vis_8u = cv.normalize(arr_c, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
            frame_bgr = cv.applyColorMap(vis_8u, cv.COLORMAP_PLASMA)
        else:
            frame_bgr = np.zeros((self.heightScaled, self.widthScaled, 3), np.uint8)
            arr_c = np.zeros((self.heightScaled, self.widthScaled), dtype=np.float32)
            vis_8u = np.zeros((self.heightScaled, self.widthScaled), dtype=np.uint8)

        # ----------------------------------------
        # Thresholding restricted to user-defined mask
        # ----------------------------------------
        gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)

        # If no mask defined, do nothing (no ROI). Otherwise apply threshold within mask only.
        contours = []
        if self.user_mask is not None and np.any(self.user_mask):
            # Zero-out pixels outside the mask to guarantee no detections there
            masked_gray = gray.copy()
            masked_gray[self.user_mask == 0] = 0

            # Light blur to reduce salt-and-pepper
            blur = cv.GaussianBlur(masked_gray, (5, 5), 0)

            # Fixed "relatively high" threshold on 8-bit normalized image
            _, thresh = cv.threshold(blur, THRESH_8U, 255, cv.THRESH_BINARY)

            # Clean small artifacts while preserving shapes
            kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
            thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
            thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

            contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            # Optional: visualize the masked area subtly (dark overlay outside mask)
            outside = (self.user_mask == 0)
            frame_bgr[outside] = (frame_bgr[outside] * 0.4).astype(np.uint8)

        # Build a mask from the largest detected region if any
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

        # Keep a rolling window of about ten seconds with padding
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

        # Draw image on Tk canvas
        rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        self.canvas.config(width=self.widthScaled, height=self.heightScaled)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas.imgtk = imgtk  # keep reference to avoid GC

        # Draw persistent mask rectangle outline (if any) over the frame
        if self.mask_bounds is not None:
            xL, yT, xR, yB = self.mask_bounds
            self.canvas.create_rectangle(xL, yT, xR, yB, outline="yellow", width=2)

        # Schedule next frame
        self.after_id = self.root.after(FRAME_DELAY_MS, self.update_video)

    # -------------------------
    # Recording controls
    # -------------------------
    def start_recording(self):
        """Start collecting ROI metrics into memory."""
        self.data_records = []
        self.recording_start_time = time.time()
        self.recording = True
        print("Recording started")

    def stop_recording(self):
        """Stop and prompt for CSV save. Writes elapsed_s, avg_temp_C, max_temp_C, roi_area_px2."""
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
            header = ['elapsed_s', 'avg_temp_C', 'max_temp_C', 'roi_area_px2']
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(self.data_records)
            print(f"Data saved to {path}")

        # Clear buffer after saving or cancel
        self.data_records = []

    # -------------------------
    # Shutdown
    # -------------------------
    def quit_app(self):
        """Gracefully stop capture graph and close the UI."""
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

# =========================
# Entrypoint
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    app = ThermalVideoApp(root)
    root.mainloop()
