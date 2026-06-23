from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ppe.temporal_tracker import TrackSnapshot
from ppe.types import ComplianceStatus, STATUS_LABELS


class ResultPanel(QWidget):
    """GUI 右側結果面板。"""

    def __init__(self, window_size: int = 100, confirm_threshold: int = 80):
        super().__init__()
        self.window_size = window_size
        self.confirm_threshold = confirm_threshold
        self.setMinimumWidth(280)
        self.setStyleSheet(
            "QGroupBox { font-weight: bold; margin-top: 8px; }"
            "QLabel { font-size: 13px; }"
        )

        layout = QVBoxLayout(self)

        summary_box = QGroupBox("整體統計")
        summary_layout = QVBoxLayout(summary_box)
        self.summary_label = QLabel("尚未開始偵測")
        self.summary_label.setWordWrap(True)
        summary_layout.addWidget(self.summary_label)
        layout.addWidget(summary_box)

        rule_box = QGroupBox("確認規則")
        rule_layout = QVBoxLayout(rule_box)
        rule_layout.addWidget(
            QLabel(
                f"滑動窗口 {window_size} 幀，至少 {confirm_threshold} 幀命中\n"
                "才視為「確認偵測到」"
            )
        )
        layout.addWidget(rule_box)

        tracks_box = QGroupBox("工人追蹤")
        tracks_layout = QVBoxLayout(tracks_box)
        self.tracks_container = QWidget()
        self.tracks_layout = QVBoxLayout(self.tracks_container)
        self.tracks_layout.setAlignment(Qt.AlignTop)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.tracks_container)
        tracks_layout.addWidget(scroll)
        layout.addWidget(tracks_box, stretch=1)

        violation_box = QGroupBox("違規 / 僅 person")
        violation_layout = QVBoxLayout(violation_box)
        self.violation_label = QLabel("—")
        self.violation_label.setWordWrap(True)
        self.violation_label.setStyleSheet("color: #c0392b;")
        violation_layout.addWidget(self.violation_label)
        layout.addWidget(violation_box)

    def update_results(self, tracks: list[TrackSnapshot], frame_index: int) -> None:
        confirmed = [t for t in tracks if t.confirmed_person]
        compliant = [t for t in confirmed if t.temporal_status == ComplianceStatus.COMPLIANT]
        person_only = [t for t in tracks if t.temporal_status == ComplianceStatus.PERSON_ONLY]
        violations = [
            t
            for t in confirmed
            if t.temporal_status
            not in (ComplianceStatus.COMPLIANT, ComplianceStatus.PERSON_ONLY)
        ]

        self.summary_label.setText(
            f"幀數：{frame_index}\n"
            f"追蹤中：{len(tracks)} 人\n"
            f"已確認 person：{len(confirmed)} 人\n"
            f"合規（確認）：{len(compliant)} 人\n"
            f"違規（確認）：{len(violations)} 人\n"
            f"僅 person（確認）：{len(person_only)} 人"
        )

        self._refresh_track_cards(tracks)
        self._refresh_violations(confirmed, person_only, violations)

    def _refresh_track_cards(self, tracks: list[TrackSnapshot]) -> None:
        while self.tracks_layout.count():
            item = self.tracks_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if not tracks:
            empty = QLabel("目前無追蹤中的 person")
            empty.setStyleSheet("color: #7f8c8d;")
            self.tracks_layout.addWidget(empty)
            return

        for track in tracks:
            card = QLabel(self._format_track(track))
            card.setWordWrap(True)
            card.setStyleSheet(
                "background: #ecf0f1; border-radius: 6px; padding: 8px; margin-bottom: 6px;"
            )
            self.tracks_layout.addWidget(card)

    def _format_track(self, track: TrackSnapshot) -> str:
        person_mark = "已確認" if track.confirmed_person else "累積中"
        helmet_mark = "已確認" if track.confirmed_helmet else "未達標"
        vest_mark = "已確認" if track.confirmed_vest else "未達標"

        return (
            f"<b>工人 #{track.track_id}</b><br>"
            f"本幀：{track.frame_status_label}<br>"
            f"時序判定：{track.status_label}<br>"
            f"person：{track.person_hits}/{track.window_size} ({person_mark})<br>"
            f"安全帽：{track.helmet_hits}/{track.window_size} ({helmet_mark})<br>"
            f"反光背心：{track.vest_hits}/{track.window_size} ({vest_mark})"
        )

    def _refresh_violations(
        self,
        confirmed: list[TrackSnapshot],
        person_only: list[TrackSnapshot],
        violations: list[TrackSnapshot],
    ) -> None:
        lines: list[str] = []

        for track in person_only:
            if track.confirmed_person:
                lines.append(
                    f"#{track.track_id} 僅偵測到 person，"
                    f"無 hard-hat / safety-vest（{track.helmet_hits}/{track.window_size}, "
                    f"{track.vest_hits}/{track.window_size}）"
                )
            else:
                lines.append(
                    f"#{track.track_id} 累積中：person {track.person_hits}/{track.confirm_threshold}"
                )

        for track in violations:
            missing = []
            if not track.confirmed_helmet:
                missing.append("hard-hat")
            if not track.confirmed_vest:
                missing.append("safety-vest")
            lines.append(
                f"#{track.track_id} {track.status_label}（缺少：{', '.join(missing)}）"
            )

        if not lines:
            if confirmed:
                self.violation_label.setText("所有已確認工人均合規。")
                self.violation_label.setStyleSheet("color: #27ae60;")
            else:
                self.violation_label.setText("尚無已確認的 person。")
                self.violation_label.setStyleSheet("color: #7f8c8d;")
            return

        self.violation_label.setText("\n".join(lines))
        self.violation_label.setStyleSheet("color: #c0392b;")
