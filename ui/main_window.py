from __future__ import annotations

import re

import cv2
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from config import load_config
from detector import YOLOEngine
from ppe.compliance import check_compliance
from ppe.smoother import Smoother
from ppe.types import ComplianceStatus
from ui.annotator import visualize
from ui.result_panel import ResultPanel


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = load_config()
        ui_cfg = self.config["ui"]
        temporal_cfg = self.config["temporal"]
        self.filters_cfg = self.config.get("filters", {})
        self.association_cfg = self.config.get("association", {})

        self.setWindowTitle(ui_cfg.get("window_title", "工人 PPE 合規偵測"))
        self.setGeometry(100, 100, 1280, 720)

        self.engine = YOLOEngine(self.config)
        self.smoother = Smoother(
            window_size=temporal_cfg["window_size"],
            confirm_threshold=temporal_cfg["confirm_threshold"],
            ppe_confirm_threshold=temporal_cfg.get("ppe_confirm_threshold", 80),
            max_missed_frames=temporal_cfg.get("max_missed_frames", 300),
        )

        self.visualizer_cfg = self.config.get("visualizer", {})
        self.timer_interval = ui_cfg.get("timer_interval_ms", 30)
        self.frame_index = 0
        self.cap = None

        self._build_ui(ui_cfg, temporal_cfg)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def _build_ui(self, ui_cfg: dict, temporal_cfg: dict) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)

        url_layout = QHBoxLayout()
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("影片網址、Google Drive 分享連結或本地路徑")
        self.url_input.setText(
            "https://drive.google.com/file/d/1DhOkFkcPwGD_2dJJpwIj2ckZ5aeb6IOX/view?usp=drive_link"
        )
        self.btn_load = QPushButton("載入影片")
        self.btn_load.clicked.connect(self.load_video)
        url_layout.addWidget(self.url_input)
        url_layout.addWidget(self.btn_load)
        root_layout.addLayout(url_layout)

        content_layout = QHBoxLayout()

        video_column = QVBoxLayout()
        self.image_label = QLabel("請輸入連結並點擊「載入影片」")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet(
            "border: 1px solid #bdc3c7; background-color: #2c3e50; color: white; font-size: 16px;"
        )
        self.image_label.setMinimumSize(
            ui_cfg.get("display_min_width", 640),
            ui_cfg.get("display_min_height", 480),
        )
        video_column.addWidget(self.image_label)

        self.btn_start = QPushButton("開始偵測播放")
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.toggle_detection)
        video_column.addWidget(self.btn_start)
        content_layout.addLayout(video_column, stretch=3)

        self.result_panel = ResultPanel(
            window_size=temporal_cfg["window_size"],
            confirm_threshold=temporal_cfg["confirm_threshold"],
            ppe_confirm_threshold=temporal_cfg.get("ppe_confirm_threshold", 80),
        )
        content_layout.addWidget(self.result_panel, stretch=1)

        root_layout.addLayout(content_layout)
        self.setCentralWidget(root)

    @staticmethod
    def parse_google_drive_url(url: str) -> str:
        if "drive.google.com" in url:
            match = re.search(r"/d/([a-zA-Z0-9-_]+)", url)
            if match:
                file_id = match.group(1)
                return f"https://drive.google.com/uc?export=download&id={file_id}"
        return url

    def load_video(self) -> None:
        self.timer.stop()
        self.btn_start.setText("開始偵測播放")
        self.frame_index = 0
        self.engine.reset()
        self.smoother.reset()

        if self.cap:
            self.cap.release()

        raw_url = self.url_input.text().strip()
        if not raw_url:
            QMessageBox.warning(self, "警告", "請輸入有效的連結或路徑！")
            return

        video_source = self.parse_google_drive_url(raw_url)
        self.image_label.setText("正在載入影片，請稍候...")
        QApplication.processEvents()

        self.cap = cv2.VideoCapture(video_source)
        if self.cap.isOpened():
            self.btn_start.setEnabled(True)
            self.image_label.setText("影片載入成功！點擊下方按鈕開始偵測。")
        else:
            self.btn_start.setEnabled(False)
            self.image_label.setText("影片載入失敗，請檢查連結或權限。")
            QMessageBox.critical(
                self,
                "錯誤",
                "無法開啟影片來源！\nGoogle Drive 檔案需開啟「知道連結的任何人都可以檢視」。",
            )

    def toggle_detection(self) -> None:
        if not self.cap or not self.cap.isOpened():
            return

        if not self.timer.isActive():
            self.timer.start(self.timer_interval)
            self.btn_start.setText("暫停偵測")
        else:
            self.timer.stop()
            self.btn_start.setText("開始偵測播放")

    def update_frame(self) -> None:
        if not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.btn_start.setText("開始偵測播放")
            self.btn_start.setEnabled(False)
            self.image_label.setText("影片播放結束。")
            self.cap.release()
            return

        self.frame_index += 1

        detections = self.engine.predict(frame)
        frame_results = check_compliance(
            detections,
            frame_shape=frame.shape,
            filters_cfg=self.filters_cfg,
            association_cfg=self.association_cfg,
        )
        tracks = self.smoother.update(frame_results)

        annotated = visualize(
            frame, detections, tracks, frame_results, self.visualizer_cfg
        )
        self._show_frame(annotated)
        self.result_panel.update_results(tracks, self.frame_index)

        active_ids = {t.track_id for t in tracks}
        compliant = sum(
            1
            for t in tracks
            if t.confirmed_person and t.temporal_status == ComplianceStatus.COMPLIANT
        )
        violation = sum(
            1
            for t in tracks
            if t.confirmed_person
            and t.temporal_status != ComplianceStatus.COMPLIANT
            and t.temporal_status != ComplianceStatus.PERSON_ONLY
        )
        self.statusBar().showMessage(
            f"幀 {self.frame_index} | 活躍 Track {len(active_ids)} | "
            f"合規 {compliant} | 違規 {violation}"
        )

    def _show_frame(self, frame) -> None:
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(
            pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def closeEvent(self, event) -> None:
        if self.cap:
            self.cap.release()
        event.accept()
