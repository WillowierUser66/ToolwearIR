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

SCALE_PERCENT   = 200
FRAME_DELAY_MS  = int(1000/60)
THRESH_8U       = 110

TEMP_RANGES = {
    "Low (-10°C to 140°C)":  (27315 - 10 * 100,  27315 + 140 * 100),
    "High (-10°C to 400°C)": (27315 - 10 * 100,  27315 + 400 * 100),
}

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
lep.sys.SetGainMode(CCI.Sys.GainMode.LOW)

clr.AddReference("ManagedIR16Filters")
from IR16Filters import IR16Capture, NewBytesFrameEvent

incoming_frames = deque(maxlen=10)

def got_a_frame(frame, width, height):
    incoming_frames.append((height, width, frame))

capture = IR16Capture()
capture.SetupGraphWithBytesCallback(NewBytesFrameEvent(got_a_frame))
capture.RunGraph()
time.sleep(3)

def short_array_to_numpy(h, w, frame_iterable):
    return np.fromiter(frame_iterable, dtype=np.uint16).reshape(h, w)

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

        self.user_mask = None
        self.mask_bounds = None
        self._drag_start = None
        self._rubberband_id = None

        self.vline_coords = None   # (x, y0, y1)
        self.vline_length_px = 0
        self._line_drag_start = None
        self._line_rubberband_id = None

        self.canvas = tk.Canvas(root)
        self.canvas.grid(row=0, column=0, padx=5, pady=5)

        if incoming_frames:
            h, w, f = incoming_frames[-1]
            preview = short_array_to_numpy(h, w, f)
        else:
            preview = np.zeros((240, 320), np.uint16)

        self.widthScaled  = int(preview.shape[1] * SCALE_PERCENT / 100)
        self.heightScaled = int(preview.shape[0] * SCALE_PERCENT / 100)
        self.dim = (self.widthScaled, self.heightScaled)

        self.user_mask = np.zeros((self.heightScaled, self.widthScaled), np.uint8)

        self.time_data, self.avg_series, self.max_series = [], [], []
        self.fig_avg, self.ax_avg = plt.subplots(figsize=(4, 2.5))
        self.line_avg, = self.ax_avg.plot([], [], label='Avg Temp (°C)')
        self.ax_avg.set_title("Average ROI Temperature")
        self.ax_avg.set_xlabel("Time (s)")
        self.ax_avg.set_ylabel("Temp (°C)")
        self.ax_avg.grid(True); self.ax_avg.legend(loc='upper right')
        self.canvas_avg = FigureCanvasTkAgg(self.fig_avg, master=root)
        self.canvas_avg.get_tk_widget().grid(row=0, column=1, padx=5, pady=5)

        self.fig_max, self.ax_max = plt.subplots(figsize=(4, 2.5))
        self.line_max, = self.ax_max.plot([], [], label='Max Temp (°C)')
        self.ax_max.set_title("Maximum ROI Temperature")
        self.ax_max.set_xlabel("Time (s)")
        self.ax_max.set_ylabel("Temp (°C)")
        self.ax_max.grid(True); self.ax_max.legend(loc='upper right')
        self.canvas_max = FigureCanvasTkAgg(self.fig_max, master=root)
        self.canvas_max.get_tk_widget().grid(row=1, column=1, padx=5, pady=5)

        control = ttk.Frame(root); control.grid(row=2, column=0, columnspan=2, pady=5)
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

        self.temp_range_var = tk.StringVar(value="Low (-10°C to 140°C)")
        ttk.Label(control, text="Temp Range:").pack(side="left", padx=5)
        self.temp_range_dropdown = ttk.Combobox(
            control, textvariable=self.temp_range_var, state="readonly",
            values=list(TEMP_RANGES.keys()), width=22
        )
        self.temp_range_dropdown.pack(side="left", padx=5)
        self.temp_range_dropdown.bind("<<ComboboxSelected>>", self.change_temp_range)
        self.change_temp_range()

        self.update_video()

    # Mask drawing
    def enable_mask_draw(self):
        self._unbind_all_draw(); self.suspend_video = True
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
        if self._rubberband_id is None:
            self._rubberband_id = self.canvas.create_rectangle(
                x0, y0, x1, y1, outline="yellow", width=2, dash=(4, 3)
            )
        else:
            self.canvas.coords(self._rubberband_id, x0, y0, x1, y1)

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

    def clear_mask(self):
        self.user_mask.fill(0); self.mask_bounds = None

    # Vertical line + release
    def enable_vertical_line(self):
        self._unbind_all_draw(); self.suspend_video = True
        self.canvas.bind("<ButtonPress-1>", self._on_line_start)
        self.canvas.bind("<B1-Motion>", self._on_line_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_line_release)

    def _on_line_start(self, event):
        x0 = self._clamp_x(event.x); y0 = self._clamp_y(event.y)
        self._line_drag_start = (x0, y0)
        if self._line_rubberband_id is not None:
            self.canvas.delete(self._line_rubberband_id); self._line_rubberband_id = None
        self._line_rubberband_id = self.canvas.create_line(x0, y0, x0, y0, fill="cyan", width=2)

    def _on_line_drag(self, event):
        if not self._line_drag_start or self._line_rubberband_id is None: return
        x0, y0 = self._line_drag_start
        y1 = self._clamp_y(event.y)
        self.canvas.coords(self._line_rubberband_id, x0, y0, x0, y1)

    def _on_line_release(self, event):
        if not self._line_drag_start: return
        x0, y0 = self._line_drag_start; y1 = self._clamp_y(event.y)
        self._line_drag_start = None
        self.vline_coords = (x0, y0, y1)
        self.vline_length_px = abs(y1 - y0)
        self.line_length_var.set(f"{self.vline_length_px} px")
        if self._line_rubberband_id is not None:
            self.canvas.delete(self._line_rubberband_id); self._line_rubberband_id = None
        self._unbind_all_draw(); self.suspend_video = False

    def clear_line(self):
        self.vline_coords = None; self.vline_length_px = 0
        self.line_length_var.set("—")

    def _unbind_all_draw(self):
        for seq in ("<ButtonPress-1>", "<B1-Motion>", "<ButtonRelease-1>"):
            self.canvas.unbind(seq)

    def change_temp_range(self, event=None):
        selected = self.temp_range_var.get()
        if selected in TEMP_RANGES:
            low_ck, high_ck = TEMP_RANGES[selected]
            try:
                lep.sys.SetTempRangeHighLow(high_ck, low_ck)
                print(f"Temperature range set to: {selected}")
            except Exception as e:
                print(f"Failed to set temperature range: {e}")

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

        half_len = 10  # total 20 px line length
        x_left = max(0, x0 - half_len)
        x_right = min(self.widthScaled - 1, x0 + half_len)
        return ys, x_left, x_right

    # Frame loop
    def update_video(self):
        if not self.running: return
        if self.suspend_video:
            self.after_id = self.root.after(FRAME_DELAY_MS, self.update_video); return

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
        if contours:
            largest = max(contours, key=cv.contourArea)
            cv.drawContours(frame_bgr, [largest], -1, (0, 255, 0), 2)
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

        # Recording row
        if self.recording:
            elapsed_rec = time.time() - self.recording_start_time
            row = [elapsed_rec, avg_temp_C, max_temp_C, roi_area_px2] + hline_avgs
            self.data_records.append(row)

        # Plots (edge-ROI)
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
        self.canvas.imgtk = imgtk

        if self.mask_bounds is not None:
            xL, yT, xR, yB = self.mask_bounds
            self.canvas.create_rectangle(xL, yT, xR, yB, outline="yellow", width=2)

        if self.vline_coords is not None:
            x0, y0, y1 = self.vline_coords
            self.canvas.create_line(x0, y0, x0, y1, fill="cyan", width=2)
            y_mid = int((y0 + y1) / 2)
            self.canvas.create_text(min(x0 + 8, self.widthScaled - 10), y_mid,
                                    text=f"{self.vline_length_px} px",
                                    fill="white", anchor="w",
                                    font=("TkDefaultFont", 10, "bold"))

            ys, x_left, x_right = self._horizontal_rois_from_vline()
            for i, yk in enumerate(ys):
                self.canvas.create_line(x_left, yk, x_right, yk, fill="cyan", width=2)
                label = "" if np.isnan(hline_avgs[i]) else f"{hline_avgs[i]:.1f}°C"
                self.canvas.create_text(min(x_right + 6, self.widthScaled - 2), yk,
                                        text=label, fill="white", anchor="w",
                                        font=("TkDefaultFont", 9, "bold"))

        self.after_id = self.root.after(FRAME_DELAY_MS, self.update_video)

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

    def _clamp_x(self, x): return max(0, min(self.widthScaled - 1, x))
    def _clamp_y(self, y): return max(0, min(self.heightScaled - 1, y))

if __name__ == "__main__":
    root = tk.Tk()
    app = ThermalVideoApp(root)
    root.mainloop()
