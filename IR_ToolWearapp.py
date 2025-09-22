import clr
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

# --- Configuration ---
scale_percent = 200
numGradients = 4
FRAME_DELAY_MS = int(1000 / 60)

TEMP_RANGES = {
    "Low (-10°C to 140°C)": (27315 - 10 * 100, 27315 + 140 * 100),
    "High (-10°C to 400°C)": (27315 - 10 * 100, 27315 + 400 * 100)
}

# --- COM & SDK Init ---
CoInitialize()
bits, _ = platform.architecture()
folder = ["x64"] if bits == "64bit" else ["x86"]
sys.path.append(os.path.join("", *folder))

clr.AddReference("LeptonUVC")
from Lepton import CCI
found_device = next((d for d in CCI.GetDevices() if d.Name.startswith("PureThermal")), None)
if not found_device:
    print("Couldn't find lepton device")
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

# --- Utilities ---
def short_array_to_numpy(height, width, frame):
    return np.fromiter(frame, dtype="uint16").reshape(height, width)

def centikelvin_to_celsius(t):
    return (t - 27315) / 100

# --- App Class ---
class ThermalVideoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Thermal Camera Viewer + ROI Plot + CSV")
        self.running = True
        self.recording = False
        self.after_id = None
        self.recording_start_time = None
        self.stream_start_time = time.time()
        self.data_records = []
        self.time_data, self.temp_data, self.max_temp_data = [], [], []

        self.root.protocol("WM_DELETE_WINDOW", self.quit_app)

        self.canvas = tk.Canvas(root)
        self.canvas.grid(row=0, column=0, padx=5, pady=5)

        # Avg temp plot
        self.fig, self.ax = plt.subplots(figsize=(4, 2.5))
        self.line, = self.ax.plot([], [], label='Avg Temp (°C)')
        self.ax.set_title("Average ROI Temperature")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Temp (°C)")
        self.ax.grid(True)
        self.ax.legend(loc='upper right')
        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.plot_canvas.get_tk_widget().grid(row=0, column=1, padx=5, pady=5)

        # Max temp plot
        self.fig2, self.ax2 = plt.subplots(figsize=(4, 2.5))
        self.line2, = self.ax2.plot([], [], label='Max Temp (°C)', color='red')
        self.ax2.set_title("Maximum ROI Temperature")
        self.ax2.set_xlabel("Time (s)")
        self.ax2.set_ylabel("Temp (°C)")
        self.ax2.grid(True)
        self.ax2.legend(loc='upper right')
        self.plot_canvas2 = FigureCanvasTkAgg(self.fig2, master=root)
        self.plot_canvas2.get_tk_widget().grid(row=1, column=1, padx=5, pady=5)

        # Controls
        control_frame = ttk.Frame(root)
        control_frame.grid(row=2, column=0, columnspan=2, pady=5)

        self.start_btn = ttk.Button(control_frame, text="Start Recording", command=self.start_recording)
        self.start_btn.pack(side="left", padx=5)

        self.stop_btn = ttk.Button(control_frame, text="Stop and Save CSV", command=self.stop_recording)
        self.stop_btn.pack(side="left", padx=5)

        self.quit_btn = ttk.Button(control_frame, text="Quit", command=self.quit_app)
        self.quit_btn.pack(side="left", padx=5)

        self.edge_method = tk.StringVar(value="Canny + Largest Contour")
        ttk.Label(control_frame, text="ROI Method:").pack(side="left", padx=5)
        self.edge_dropdown = ttk.Combobox(control_frame, textvariable=self.edge_method, state="readonly",
                                          values=["Canny + Largest Contour", "Convex Hull", "Closed Edges", "Threshold", "Watershed"], width=18)
        self.edge_dropdown.pack(side="left", padx=5)

        self.temp_range_var = tk.StringVar(value="Low (-10°C to 140°C)")
        ttk.Label(control_frame, text="Temp Range:").pack(side="left", padx=5)
        self.temp_range_dropdown = ttk.Combobox(control_frame, textvariable=self.temp_range_var, state="readonly",
                                                values=list(TEMP_RANGES.keys()), width=20)
        self.temp_range_dropdown.pack(side="left", padx=5)
        self.temp_range_dropdown.bind("<<ComboboxSelected>>", self.change_temp_range)
        self.change_temp_range()

        self.threshold_value = tk.IntVar(value=150)
        ttk.Label(control_frame, text="Threshold:").pack(side="left", padx=5)
        self.threshold_slider = ttk.Scale(control_frame, from_=0, to=255, orient="horizontal", variable=self.threshold_value)
        self.threshold_slider.pack(side="left", padx=5)

        if incoming_frames:
            h, w, f = incoming_frames[-1]
            preview = short_array_to_numpy(h, w, f)
        else:
            preview = np.zeros((240, 320), np.uint16)
        self.widthScaled = int(preview.shape[1] * scale_percent / 100)
        self.heightScaled = int(preview.shape[0] * scale_percent / 100)
        self.dim = (self.widthScaled, self.heightScaled)

        self.update_video()

    def change_temp_range(self, event=None):
        selected = self.temp_range_var.get()
        if selected in TEMP_RANGES:
            low, high = TEMP_RANGES[selected]
            try:
                lep.sys.SetTempRangeHighLow(high, low)
                print(f"Temperature range set to: {selected}")
            except Exception as e:
                print(f"Failed to set temperature range: {e}")

    def update_video(self):
        if not self.running:
            return

        if incoming_frames:
            h, w, f = incoming_frames[-1]
            arr = short_array_to_numpy(h, w, f)
            arr_resized = centikelvin_to_celsius(cv.resize(arr, self.dim))
            normed = cv.normalize(arr_resized, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
            frame_bgr = cv.applyColorMap(normed, cv.COLORMAP_PLASMA)
        else:
            frame_bgr = np.zeros((self.heightScaled, self.widthScaled, 3), np.uint8)
            arr_resized = np.zeros((self.heightScaled, self.widthScaled), dtype=np.float32)

        gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
        method = self.edge_method.get()
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        contours = []

        if method == "Canny + Largest Contour":
            edges = cv.Canny(gray, 50, 150)
            edges = cv.dilate(cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel), kernel, iterations=1)
            contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        elif method == "Convex Hull":
            edges = cv.Canny(gray, 50, 150)
            contours_all, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            if contours_all:
                hull_points = [cv.convexHull(np.vstack(contours_all))]
                contours = hull_points

        elif method == "Closed Edges":
            edges = cv.Canny(gray, 50, 150)
            closed = cv.morphologyEx(edges, cv.MORPH_CLOSE, kernel)
            contours, _ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        elif method == "Threshold":
            blur = cv.GaussianBlur(gray, (5, 5), 0)
            threshold_val = self.threshold_value.get()
            _, thresh = cv.threshold(blur, threshold_val, 255, cv.THRESH_BINARY)
            thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
            thresh = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
            contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        elif method == "Watershed":
            _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            sure_bg = cv.dilate(binary, kernel, iterations=3)
            dist_transform = cv.distanceTransform(binary, cv.DIST_L2, 5)
            _, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            unknown = cv.subtract(sure_bg, sure_fg)
            _, markers = cv.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            markers = cv.watershed(frame_bgr, markers)
            mask_watershed = np.zeros_like(gray)
            mask_watershed[markers > 1] = 255
            contours, _ = cv.findContours(mask_watershed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        mask = np.zeros((self.heightScaled, self.widthScaled), np.uint8)
        avg_temp_C = 0
        max_temp_C = 0
        roi_area_px2 = 0
        roi_avgs = []

        if contours:
            largest = max(contours, key=cv.contourArea)
            cv.drawContours(frame_bgr, [largest], -1, (0, 255, 0), 2)
            cv.drawContours(mask, [largest], -1, 1, -1)
            vals = arr_resized[mask == 1]
            if len(vals):
                avg_temp_C = np.mean(vals)
                max_temp_C = np.max(vals)
            roi_area_px2 = int(np.sum(mask == 1))
            roi_avgs.append(avg_temp_C)
            while len(roi_avgs) < numGradients:
                roi_avgs.append(0)
        else:
            roi_avgs = [0] * numGradients

        if self.recording:
            elapsed = time.time() - self.recording_start_time
            self.data_records.append([elapsed] + roi_avgs + [avg_temp_C, max_temp_C, roi_area_px2])

        elapsed_plot = time.time() - self.stream_start_time
        self.time_data.append(elapsed_plot)
        self.temp_data.append(avg_temp_C)
        self.max_temp_data.append(max_temp_C)

        self.line.set_data(self.time_data, self.temp_data)
        self.ax.set_xlim(max(0, elapsed_plot - 10), elapsed_plot + 1)
        self.ax.set_ylim(min(self.temp_data[-60:] + [avg_temp_C]) - 1,
                         max(self.temp_data[-60:] + [avg_temp_C]) + 1)
        self.plot_canvas.draw()

        self.line2.set_data(self.time_data, self.max_temp_data)
        self.ax2.set_xlim(max(0, elapsed_plot - 10), elapsed_plot + 1)
        self.ax2.set_ylim(min(self.max_temp_data[-60:] + [max_temp_C]) - 1,
                          max(self.max_temp_data[-60:] + [max_temp_C]) + 1)
        self.plot_canvas2.draw()

        rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        self.canvas.config(width=self.widthScaled, height=self.heightScaled)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas.imgtk = imgtk

        self.after_id = self.root.after(FRAME_DELAY_MS, self.update_video)

    def start_recording(self):
        self.data_records = []
        self.recording_start_time = time.time()
        self.recording = True
        print("Recording started...")

    def stop_recording(self):
        if not self.recording or not self.data_records:
            print("No data recorded.")
            return
        self.recording = False
        path = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV', '*.csv')],
                                            title='Save ROI Data As')
        if path:
            import csv
            header = ['elapsed_time'] + [f'ROI_{i}' for i in range(numGradients)] + ['avg_temp_C', 'max_temp_C', 'roi_area_px2']
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
        self.root.quit()
        self.root.destroy()
        sys.exit(0)

if __name__ == "__main__":
    root = tk.Tk()
    app = ThermalVideoApp(root)
    root.mainloop()
