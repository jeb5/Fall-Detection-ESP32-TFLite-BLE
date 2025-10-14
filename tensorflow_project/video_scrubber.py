# video_scrubber.py
import sys, os
import numpy as np
import pandas as pd
import cv2
from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg


class VideoScrubber(QtWidgets.QWidget):
    def __init__(self, video_path, data_path=None, data_start_ms=None, data_end_ms=None):
        super().__init__()
        self.setWindowTitle("Video + Data Scrubber")

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

        # --- UI elements ---
        self.video_label = QtWidgets.QLabel()
        self.video_label.setFixedSize(360, 640)

        self.play_btn = QtWidgets.QPushButton("Play")
        self.play_btn.setCheckable(True)
        self.play_btn.clicked.connect(self.toggle_play)

        self.back_btn = QtWidgets.QPushButton("-10s")
        self.back_btn.clicked.connect(lambda: self.skip_time(-10))
        self.fwd_btn = QtWidgets.QPushButton("+10s")
        self.fwd_btn.clicked.connect(lambda: self.skip_time(10))

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
        video_layout.addWidget(self.video_label)
        video_layout.addWidget(self.slider)
        video_layout.addLayout(play_controls)

        # --- Graph setup ---
        self.plot = pg.PlotWidget()
        self.plot.setLabel('bottom', 'Time (s)')
        self.plot.showGrid(x=True, y=True)
        self.cur_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(width=2))
        self.plot.addItem(self.cur_line)
        self.marks_scatter = pg.ScatterPlotItem()
        self.plot.addItem(self.marks_scatter)
        self.mark_lines = []

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
        for w in [self.mark_btn, self.del_mark_btn, self.prev_mark_btn, self.next_mark_btn, self.save_btn]:
            mark_controls.addWidget(w)

        graph_layout = QtWidgets.QVBoxLayout()
        graph_layout.addWidget(self.plot)
        graph_layout.addLayout(mark_controls)

        main_layout = QtWidgets.QHBoxLayout()
        main_layout.addLayout(video_layout, stretch=1)
        main_layout.addLayout(graph_layout, stretch=2)
        self.setLayout(main_layout)

        # --- Data loading ---
        self.data_t = np.array([])
        self.data_v_dict = {}
        self.plot_items = []
        if data_path and os.path.exists(data_path):
            self.load_data(data_path)

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

        # Initial frame
        self.set_frame(0)
        self.plot.scene().sigMouseClicked.connect(self.on_plot_clicked)

    # ------------------- Data -------------------
    def load_data(self, path):
        df = pd.read_csv(path)

        self.data_t = df["time_ms"].to_numpy(dtype=float) / 1000.0  # convert ms → s

        # Other numeric columns
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "time_ms"]
        colors = pg.intColor
        for i, c in enumerate(numeric_cols):
            y = df[c].to_numpy(dtype=float)
            pen = pg.mkPen(color=colors(i, hues=len(numeric_cols)), width=1)
            item = self.plot.plot([], [], pen=pen, name=c)
            self.data_v_dict[c] = y
            self.plot_items.append(item)

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
        qt_image = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        scaled = qt_image.scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio)
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(scaled))
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
        new_frame = int(round(((new_time * 1000 - self.data_start_ms) / self.data_time_span) * (self.frame_count - 1)))
        self.set_frame(new_frame)

    # ------------------- Plot updates -------------------
    def update_plot_window(self, center_time):
        if self.data_t.size == 0:
            return
        half = self.window_halfwidth
        mask = (self.data_t >= center_time - half) & (self.data_t <= center_time + half)
        if not np.any(mask):
            return

        for i, (col, y) in enumerate(self.data_v_dict.items()):
            t_slice, v_slice = self.data_t[mask], y[mask]
            self.plot_items[i].setData(t_slice, v_slice)

        self.plot.setXRange(center_time - half, center_time + half)
        self.cur_line.setValue(center_time)
        self.update_marks_plot()
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
        new_frame = int(round(((t * 1000 - self.data_start_ms) / self.data_time_span) * (self.frame_count - 1)))
        self.set_frame(new_frame)

    def update_marks_plot(self):
        # Clear old mark lines
        for line in self.mark_lines:
            self.plot.removeItem(line)
        self.mark_lines.clear()
        
        if not self.marks:
            self.marks_scatter.clear()
            return
        
        # Add vertical red lines for each mark
        for mark_time in self.marks:
            line = pg.InfiniteLine(
                pos=mark_time,
                angle=90,
                movable=False,
                pen=pg.mkPen(color='r', width=2, style=QtCore.Qt.DashLine)
            )
            self.plot.addItem(line)
            self.mark_lines.append(line)
        
        # Also show red dots for visibility
        xs = np.array(self.marks)
        y_range = self.plot.getViewBox().viewRange()[1]
        y_mid = (y_range[0] + y_range[1]) / 2
        ys = np.full_like(xs, y_mid)
        self.marks_scatter.setData(x=xs, y=ys, size=12, brush='r', symbol='o')

    def save_marks(self):
        if not self.marks:
            QtWidgets.QMessageBox.information(self, "No marks", "No marks to save.")
            return
        df = pd.DataFrame({'time_s': self.marks})
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save marks CSV", "marks.csv", "CSV files (*.csv)")
        if path:
            df.to_csv(path, index=False)
            QtWidgets.QMessageBox.information(self, "Saved", f"Saved {len(self.marks)} marks to {path}")

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
        vb = self.plot.getViewBox()
        if vb.sceneBoundingRect().contains(pos):
            mouse_point = vb.mapSceneToView(pos)
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
        print("Usage: python video_scrubber.py video.mp4 [data.csv] [data_start_ms data_end_ms]")
        sys.exit(1)
    video = sys.argv[1]
    data = sys.argv[2] if len(sys.argv) > 2 else None
    start_ms = sys.argv[3] if len(sys.argv) > 3 else None
    end_ms = sys.argv[4] if len(sys.argv) > 4 else None

    app = QtWidgets.QApplication([])
    w = VideoScrubber(video, data, start_ms, end_ms)
    w.resize(1200, 600)
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
