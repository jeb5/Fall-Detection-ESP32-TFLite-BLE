#
# NOTE:
# A large portion of this code was created with Generative AI
#
import math
import sys
import os
import time
import numpy as np
import pandas as pd
import cv2
from naive import naive, custom_feature
from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg

columns = {
    "time": {},
    "button": {},
    "acc_x": {},
    "acc_y": {},
    "acc_z": {},
    "gyro_x": {"scale": 0.02},
    "gyro_y": {"scale": 0.02},
    "gyro_z": {"scale": 0.02},
    "custom_feature":{},
    "acc_mag":{},
    "angle_lp":{},
    "dog_value":{},
    "angle":{},
    "fall":{}
}

class DataAnnotater(QtWidgets.QWidget):
    def __init__( self, video_path, data_path=None, data_start_ms=None, data_end_ms=None):
        super().__init__()
        self.setWindowTitle("Data annotation helper")

        # --- Load video ---
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Can't open video: {video_path}")

        self.video_path = video_path
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.duration = self.frame_count / self.fps if self.fps else 0.0

        # --- Data mapping setup ---
        self.data_start_ms = float(data_start_ms) if data_start_ms else 0.0
        self.data_end_ms = float(data_end_ms) if data_end_ms else self.duration * 1000.0
        self.data_time_span = max(1.0, self.data_end_ms - self.data_start_ms)

        # --- Video playback elements setup ---
        self.video_widget = QtWidgets.QLabel()
        self.video_widget.setFixedSize(360, 640)

        self.play_btn = QtWidgets.QPushButton("Play")
        self.play_btn.setCheckable(True)
        self.play_btn.clicked.connect(self.toggle_play)

        self.back_btn = QtWidgets.QPushButton("-5s")
        self.back_btn.clicked.connect(lambda: self.skip_time(-5))
        self.fwd_btn = QtWidgets.QPushButton("+5s")
        self.fwd_btn.clicked.connect(lambda: self.skip_time(5))

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(0, self.frame_count - 1))
        self.slider.sliderMoved.connect(self.on_slider_moved)
        self.slider.sliderPressed.connect(self.on_slider_pressed)
        self.slider.sliderReleased.connect(self.on_slider_released)
        self.user_is_scrubbing = False

        play_controls = QtWidgets.QHBoxLayout()
        for w in [self.play_btn, self.back_btn, self.fwd_btn]:
            play_controls.addWidget(w)

        video_layout = QtWidgets.QVBoxLayout()
        video_layout.addStretch(1)
        video_layout.addWidget(self.video_widget)
        video_layout.addStretch(1)
        video_layout.addWidget(self.slider)
        video_layout.addLayout(play_controls)

        # --- Graph setup ---
        # Top plot: Raw sensor data
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Time (s)")
        self.plot.setLabel("left", "Raw Sensor Data")
        self.plot.showGrid(x=True, y=True)
        self.cur_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(width=2))
        self.plot.addItem(self.cur_line)
        self.plot.addLegend()
        self.mark_lines = []
        
        # Bottom plot: Custom computed data
        self.plot_custom = pg.PlotWidget()
        self.plot_custom.setLabel("bottom", "Time (s)")
        self.plot_custom.setLabel("left", "Custom Data")
        self.plot_custom.showGrid(x=True, y=True)
        self.cur_line_custom = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(width=2))
        self.plot_custom.addItem(self.cur_line_custom)
        self.plot_custom.addLegend()
        self.mark_lines_custom = []

        # Mark controls
        self.mark_btn = QtWidgets.QPushButton("Add Mark")
        self.del_mark_btn = QtWidgets.QPushButton("Delete Mark")
        self.next_mark_btn = QtWidgets.QPushButton("Next Mark")
        self.prev_mark_btn = QtWidgets.QPushButton("Prev Mark")
        self.save_btn = QtWidgets.QPushButton("Save Marks")

        self.mark_btn.clicked.connect(self.add_mark)
        self.del_mark_btn.clicked.connect(self.delete_mark)
        self.next_mark_btn.clicked.connect(self.next_mark)
        self.prev_mark_btn.clicked.connect(self.prev_mark)
        self.save_btn.clicked.connect(self.save_marks)

        mark_controls = QtWidgets.QHBoxLayout()
        for w in [ self.mark_btn, self.del_mark_btn, self.prev_mark_btn, self.next_mark_btn, self.save_btn ]:
            mark_controls.addWidget(w)

        graph_layout = QtWidgets.QVBoxLayout()
        graph_layout.addWidget(self.plot, stretch=1)
        graph_layout.addWidget(self.plot_custom, stretch=1)
        graph_layout.addLayout(mark_controls)

        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(video_layout, stretch=0)
        main_layout.addLayout(graph_layout, stretch=2)
        self.setLayout(main_layout)

        # --- Data loading ---
        self.data_t = np.array([])
        self.data_v_dict = {}
        self.plot_items = []
        self.custom_data_dict = {}
        self.custom_plot_items = []
        if data_path and os.path.exists(data_path):
            self.load_data(data_path)
            self.compute_custom_data()

        # --- State ---
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.next_frame)
        self.timer.setInterval(int(1000 / self.fps) if self.fps else 33)
        self.current_frame = 0
        self.marks = []
        self.window_halfwidth = 10.0  # seconds

        # --- Load existing marks if present ---
        marks_path = os.path.splitext(video_path)[0] + "_marks.csv"
        if os.path.exists(marks_path):
            try:
                df = pd.read_csv(marks_path)
                if "time_s" in df.columns:
                    self.marks = sorted(df["time_s"].to_list())
                    print(f"Loaded {len(self.marks)} marks from {marks_path}")
            except Exception as e:
                print(f"Failed to load marks from {marks_path}: {e}")
        self.update_marks_plot()

        # Initial frame
        self.set_frame(0)
        self.plot.scene().sigMouseClicked.connect(self.on_plot_clicked)
        self.plot_custom.scene().sigMouseClicked.connect(self.on_plot_clicked)

    # ------------------- Data -------------------
    def load_data(self, path):
        df = pd.read_csv(path)

        self.data_t = df["time"].to_numpy(dtype=float) / 1000.0  # convert ms → s

        # Other numeric columns
        colors = pg.intColor
        column_names = columns.keys()
        column_names = [c for c in column_names if c != "time" and c in df.columns]
        for i, column in enumerate(column_names):
            y = df[column].to_numpy(dtype=float)
            pen = pg.mkPen(color=colors(i, hues=len(column_names)), width=1)
            column_legend_name = column.replace("_", " ")
            if "scale" in columns[column]:
                column_legend_name += f" (x{columns[column]['scale']})"
            item = self.plot.plot([], [], pen=pen, name=column_legend_name)
            self.data_v_dict[column] = y
            self.plot_items.append(item)
    
    def compute_custom_data(self):
        """Compute custom data channels from raw sensor data."""
        if self.data_t.size == 0:
            return
        
        # Acceleration magnitude
        if all(k in self.data_v_dict for k in ["acc_x", "acc_y", "acc_z"]):
            acc_x = self.data_v_dict["acc_x"]
            acc_y = self.data_v_dict["acc_y"]
            acc_z = self.data_v_dict["acc_z"]
            acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
            self.custom_data_dict["acc_magnitude"] = acc_mag
            
            # Create plot item for acceleration magnitude
            pen = pg.mkPen(color="g", width=1)
            item = self.plot_custom.plot([], [], pen=pen, name="Acc Magnitude")
            self.custom_plot_items.append(item)
        
        # # Acceleration magnitude low-pass filtered
        # if "acc_magnitude" in self.custom_data_dict:
        #     acc_mag = self.custom_data_dict["acc_magnitude"]

        #     window_size = 96  # samples
        #     # Simple averaging filter
        #     kernel = np.ones(window_size) / window_size
        #     acc_mag_lp = np.convolve(acc_mag, kernel, mode="same")

        #     self.custom_data_dict["acc_magnitude_lp"] = acc_mag_lp
            
        #     # Create plot item for low-pass filtered acceleration magnitude
        #     pen = pg.mkPen(color="r", width=1)
        #     item = self.plot_custom.plot([], [], pen=pen, name="Acc Magnitude LPF")
        #     self.custom_plot_items.append(item)

        # # Difference of Gaussians on acceleration magnitude
        # if "acc_magnitude" in self.custom_data_dict:
        #     acc_mag = self.custom_data_dict["acc_magnitude"]

        #     sigma_small = 9
        #     sigma_large = sigma_small * 3
        #     radius = int(np.ceil(3 * sigma_large))
        #     x = np.arange(-radius, radius + 1)

        #     gauss_small = np.exp(-0.5 * (x / sigma_small) ** 2)
        #     gauss_small /= gauss_small.sum()

        #     gauss_large = np.exp(-0.5 * (x / sigma_large) ** 2)
        #     gauss_large /= gauss_large.sum()

        #     dog_kernel = gauss_small - gauss_large
        #     acc_mag_dog = np.convolve(acc_mag, dog_kernel, mode="same")
        #     self.custom_data_dict["acc_magnitude_dog"] = acc_mag_dog
            
        #     # Create plot item for low-pass filtered acceleration magnitude
        #     pen = pg.mkPen(color="b", width=2)
        #     item = self.plot_custom.plot([], [], pen=pen, name="Acc Magnitude DoG")
        #     self.custom_plot_items.append(item)
        # # Angle away from low pass acceleration
        # if "acc_magnitude_lp" in self.custom_data_dict:
        #     # Get low-pass acc_x, acc_y, acc_z
        #     acc_x = self.data_v_dict["acc_x"]
        #     acc_y = self.data_v_dict["acc_y"]
        #     acc_z = self.data_v_dict["acc_z"]
        #     # window_size = 96  # samples
        #     window_size = 30  # samples
        #     kernel = np.ones(window_size) / window_size
        #     acc_x_lp = np.convolve(acc_x, kernel, mode="same")
        #     acc_y_lp = np.convolve(acc_y, kernel, mode="same")
        #     acc_z_lp = np.convolve(acc_z, kernel, mode="same")
        #     # Get angle between (acc_x, acc_y, acc_z) and (acc_x_lp, acc_y_lp, acc_z_lp)
        #     dot_product = (acc_x * acc_x_lp + acc_y * acc_y_lp + acc_z * acc_z_lp)
        #     mag_orig = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        #     mag_lp = np.sqrt(acc_x_lp**2 + acc_y_lp**2 + acc_z_lp**2)
        #     cos_angle = dot_product / (mag_orig * mag_lp + 1e-6)
        #     cos_angle = np.clip(cos_angle, -1.0, 1.0)
        #     angles = np.arccos(cos_angle)  # in radians
        #     self.custom_data_dict["angle_from_lp"] = angles
        #     # Create plot item for angle from low-pass acceleration
        #     pen = pg.mkPen(color="c", width=0.5)
        #     item = self.plot_custom.plot([], [], pen=pen, name="Angle from LP Acc")
        #     self.custom_plot_items.append(item)

        # # Low-pass angle change
        # if "angle_from_lp" in self.custom_data_dict:
        #     angles = self.custom_data_dict["angle_from_lp"]
        #     window_size = 30  # samples
        #     kernel = np.ones(window_size) / window_size
        #     angle_diff_lp = np.convolve(angles, kernel, mode="same")
        #     self.custom_data_dict["angle_change_lp"] = np.abs(angle_diff_lp)
            
        #     # Create plot item for low-pass angle change
        #     pen = pg.mkPen(color="y", width=1)
        #     item = self.plot_custom.plot([], [], pen=pen, name="Angle Change LPF")
        #     self.custom_plot_items.append(item)
        
        # low pass angle change multiplied by acc dog
        # if "angle_change_lp" in self.custom_data_dict and "acc_magnitude_dog" in self.custom_data_dict:
        #     angle_change_lp = self.custom_data_dict["angle_change_lp"]
        #     acc_mag_dog = self.custom_data_dict["acc_magnitude_dog"]
        #     combined_feature = angle_change_lp * acc_mag_dog
        #     self.custom_data_dict["combined_feature"] = combined_feature
            
        #     # Create plot item for combined feature
        #     pen = pg.mkPen(color="w", width=2)
        #     item = self.plot_custom.plot([], [], pen=pen, name="Combined Feature")
        #     self.custom_plot_items.append(item)

        # naive_values = [0] * (193 - 81)
        # for i in range(len(self.data_t)-193):
        #     input_to_naive = {
        #         "acc_x": self.data_v_dict["acc_x"][i:i+193],
        #         "acc_y": self.data_v_dict["acc_y"][i:i+193],
        #         "acc_z": self.data_v_dict["acc_z"][i:i+193],
        #     }
        #     possible_feature = naive(input_to_naive)
        #     naive_values.append(possible_feature)
        # naive_values += [0] * (len(self.data_t) - len(naive_values)) 
        # self.custom_data_dict["naive_feature"] = np.array(naive_values)
        # # Create plot item for naive feature
        # pen = pg.mkPen(color="m", width=2)
        # item = self.plot_custom.plot([], [], pen=pen, name="Naive Feature")
        # self.custom_plot_items.append(item)

        # Create np array for acc_x, acc_y, acc_z
        acc_data = np.vstack((
            self.data_v_dict["acc_x"],
            self.data_v_dict["acc_y"],
            self.data_v_dict["acc_z"]
        ))

        # optimal_params = (6.6, 40, 16, 1.0, 0.07)
        custom_feature_values, (anglechange_lp, acc_mag_dog, angles) = custom_feature(acc_data, 6.6, 40, 16, 1.0)
        self.custom_data_dict["custom_feature"] = custom_feature_values
        pen = pg.mkPen(color="g", width=2)
        item = self.plot_custom.plot([], [], pen=pen, name="Custom Feature")
        self.custom_plot_items.append(item)

        self.custom_data_dict["angle_change_lp"] = anglechange_lp
        pen = pg.mkPen(color="b", width=1)
        item = self.plot_custom.plot([], [], pen=pen, name="Angle Change LPF")
        self.custom_plot_items.append(item)

        self.custom_data_dict["acc_magnitude_dog"] = acc_mag_dog
        pen = pg.mkPen(color="c", width=1)
        item = self.plot_custom.plot([], [], pen=pen, name="Acc Magnitude DoG")
        self.custom_plot_items.append(item)

        self.custom_data_dict["angles"] = angles
        pen = pg.mkPen(color="m", width=1)
        item = self.plot_custom.plot([], [], pen=pen, name="Angles")
        self.custom_plot_items.append(item)

        if "custom_feature" in self.custom_data_dict:
            fall = self.custom_data_dict["custom_feature"] > 0.07
            self.custom_data_dict["fall_events"] = fall.astype(float)
            # Create plot item for fall events
            pen = pg.mkPen(color="r", width=5)
            item = self.plot_custom.plot([], [], pen=pen, name="Fall Events")
            self.custom_plot_items.append(item)
        

        # # Add angle_dog to acc_magnitude_dog for combined feature
        # if "angle_dog" in self.custom_data_dict and "acc_magnitude_dog" in self.custom_data_dict:
        #     angle_dog = self.custom_data_dict["angle_dog"]
        #     acc_mag_dog = self.custom_data_dict["acc_magnitude_dog"]
        #     combined = angle_dog + acc_mag_dog
        #     self.custom_data_dict["combined_feature"] = combined
            
        #     # Create plot item for combined feature
        #     pen = pg.mkPen(color="w", width=2)
        #     item = self.plot_custom.plot([], [], pen=pen, name="Combined Feature")
        #     self.custom_plot_items.append(item)

    # ------------------- Time mapping -------------------
    def frame_to_data_time(self, frame_idx):
        frac = frame_idx / max(1, self.frame_count - 1)
        data_ms = self.data_start_ms + frac * self.data_time_span
        return data_ms / 1000.0

    # ------------------- Frame control -------------------
    def set_frame(self, frame_idx):
        frame_idx = int(np.clip(frame_idx, 0, max(0, self.frame_count - 1)))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return
        self.current_frame = frame_idx
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(
            rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        scaled = qt_image.scaled(self.video_widget.size(), QtCore.Qt.KeepAspectRatio)
        self.video_widget.setPixmap(QtGui.QPixmap.fromImage(scaled))
        self.slider.blockSignals(True)
        self.slider.setValue(frame_idx)
        self.slider.blockSignals(False)

        cur_time = self.frame_to_data_time(frame_idx)
        self.update_plot_window(cur_time)

    def next_frame(self):
        if self.current_frame + 1 >= self.frame_count:
            self.play_btn.setChecked(False)
            self.timer.stop()
            return
        self.set_frame(self.current_frame + 1)

    def toggle_play(self, checked):
        if checked:
            self.play_btn.setText("Pause")
            self.timer.start()
        else:
            self.play_btn.setText("Play")
            self.timer.stop()

    def skip_time(self, delta_sec):
        new_time = self.frame_to_data_time(self.current_frame) + delta_sec
        new_frame = int(
            round(
                ((new_time * 1000 - self.data_start_ms) / self.data_time_span)
                * (self.frame_count - 1)
            )
        )
        self.set_frame(new_frame)

    # ------------------- Plot updates -------------------
    def update_plot_window(self, center_time):
        if self.data_t.size == 0:
            return
        half = self.window_halfwidth
        mask = (self.data_t >= center_time - half) & (self.data_t <= center_time + half)
        if not np.any(mask):
            return

        # Update raw data plot
        for i, (col_name, y) in enumerate(self.data_v_dict.items()):
            t_slice, v_slice = self.data_t[mask], y[mask]
            scale = columns[col_name].get("scale", 1.0)
            v_slice = v_slice * scale
            self.plot_items[i].setData(t_slice, v_slice)

        self.plot.setXRange(center_time - half, center_time + half)
        self.cur_line.setValue(center_time)
        
        # Update custom data plot
        for i, (col_name, y) in enumerate(self.custom_data_dict.items()):
            t_slice, v_slice = self.data_t[mask], y[mask]
            self.custom_plot_items[i].setData(t_slice, v_slice)
        
        self.plot_custom.setXRange(center_time - half, center_time + half)
        self.cur_line_custom.setValue(center_time)
        
        self.update_mark_button_states()

    # ------------------- Marks -------------------
    def add_mark(self):
        t = self.frame_to_data_time(self.current_frame)
        self.marks.append(t)
        self.marks = sorted(self.marks)
        self.update_marks_plot()
        self.update_mark_button_states()

    def delete_mark(self):
        if not self.marks:
            return
        t = self.frame_to_data_time(self.current_frame)
        self.marks = [m for m in self.marks if abs(m - t) > 0.1]
        self.update_marks_plot()
        self.update_mark_button_states()

    def next_mark(self):
        if not self.marks:
            return
        t = self.frame_to_data_time(self.current_frame)
        for m in self.marks:
            if m > t + 0.05:
                self.seek_to_time(m)
                break
        self.update_mark_button_states()

    def prev_mark(self):
        if not self.marks:
            return
        t = self.frame_to_data_time(self.current_frame)
        for m in reversed(self.marks):
            if m < t - 0.05:
                self.seek_to_time(m)
                break
        self.update_mark_button_states()

    def seek_to_time(self, t):
        new_frame = int(
            round(
                ((t * 1000 - self.data_start_ms) / self.data_time_span)
                * (self.frame_count - 1)
            )
        )
        self.set_frame(new_frame)

    def update_marks_plot(self):
        # Clear old mark lines from both plots
        for line in self.mark_lines:
            self.plot.removeItem(line)
        for line in self.mark_lines_custom:
            self.plot_custom.removeItem(line)
        
        self.mark_lines.clear()
        self.mark_lines_custom.clear()

        # Add vertical red lines for each mark on both plots
        for mark_time in self.marks:
            # Mark on raw data plot
            line = pg.InfiniteLine(
                pos=mark_time,
                angle=90,
                movable=False,
                pen=pg.mkPen(color="r", width=2, style=QtCore.Qt.DashLine),
            )
            self.plot.addItem(line)
            self.mark_lines.append(line)
            
            # Mark on custom data plot
            line_custom = pg.InfiniteLine(
                pos=mark_time,
                angle=90,
                movable=False,
                pen=pg.mkPen(color="r", width=2, style=QtCore.Qt.DashLine),
            )
            self.plot_custom.addItem(line_custom)
            self.mark_lines_custom.append(line_custom)

    def save_marks(self):
        if not self.marks:
            QtWidgets.QMessageBox.information(self, "No marks", "No marks to save.")
            return
        df = pd.DataFrame({"time_s": self.marks})
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save marks CSV", "marks.csv", "CSV files (*.csv)"
        )
        if path:
            df.to_csv(path, index=False)
            QtWidgets.QMessageBox.information(
                self, "Saved", f"Saved {len(self.marks)} marks to {path}"
            )

    # ------------------- UI events -------------------
    def on_slider_moved(self, value):
        if self.user_is_scrubbing:
            self.set_frame(value)

    def on_slider_pressed(self):
        self.user_is_scrubbing = True
        self.timer_was_running = self.timer.isActive()
        if self.timer_was_running:
            self.timer.stop()
            self.play_btn.setChecked(False)
            self.play_btn.setText("Play")

    def on_slider_released(self):
        self.user_is_scrubbing = False

    def on_plot_clicked(self, ev):
        pos = ev.scenePos()
        
        # Check if click is on raw data plot
        vb = self.plot.getViewBox()
        if vb.sceneBoundingRect().contains(pos):
            mouse_point = vb.mapSceneToView(pos)
            x = mouse_point.x()
            self.seek_to_time(x)
            return
        
        # Check if click is on custom data plot
        vb_custom = self.plot_custom.getViewBox()
        if vb_custom.sceneBoundingRect().contains(pos):
            mouse_point = vb_custom.mapSceneToView(pos)
            x = mouse_point.x()
            self.seek_to_time(x)

    def update_mark_button_states(self):
        current_t = self.frame_to_data_time(self.current_frame)
        has_mark = current_t in self.marks
        self.del_mark_btn.setEnabled(has_mark)
        self.next_mark_btn.setEnabled(any(m > current_t for m in self.marks))
        self.prev_mark_btn.setEnabled(any(m < current_t for m in self.marks))


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python video_scrubber.py video.mp4 [data.csv] [data_start_ms data_end_ms]"
        )
        sys.exit(1)
    video = sys.argv[1]
    data = sys.argv[2] if len(sys.argv) > 2 else None
    start_ms = sys.argv[3] if len(sys.argv) > 3 else None
    end_ms = sys.argv[4] if len(sys.argv) > 4 else None

    app = QtWidgets.QApplication([])
    w = DataAnnotater(video, data, start_ms, end_ms)
    w.resize(1200, 600)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
