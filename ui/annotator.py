from __future__ import annotations

import cv2
import numpy as np

from ppe.compliance import get_associated_ppe_detections
from ppe.smoother import TrackSnapshot
from ppe.types import ComplianceStatus, Detection, PersonCompliance, STATUS_LABELS


def visualize(
    frame: np.ndarray,
    detections: list[Detection],
    tracks: list[TrackSnapshot],
    compliance_results: list[PersonCompliance],
    visualizer_cfg: dict,
) -> np.ndarray:
    """
    繪製 Track ID、Head/Torso ROI、合規狀態與已關聯 PPE（CLAUDE.md 輸出要求）。
    """
    output = frame.copy()
    colors = visualizer_cfg.get("colors", {})
    thickness = visualizer_cfg.get("box_thickness", 2)
    text_scale = visualizer_cfg.get("text_scale", 0.6)

    head_roi_color = tuple(colors.get("head_roi", [255, 255, 0]))
    torso_roi_color = tuple(colors.get("torso_roi", [255, 0, 255]))
    ppe_color = tuple(colors.get("ppe", [255, 191, 0]))

    track_by_id = {t.track_id: t for t in tracks}

    # 繪製 Head / Torso ROI（參考區域）
    for result in compliance_results:
        if result.head_roi:
            _draw_roi(output, result.head_roi, head_roi_color, thickness)
        if result.torso_roi:
            _draw_roi(output, result.torso_roi, torso_roi_color, thickness)

    # 孤立裝備過濾：僅繪製已關聯到工人的 PPE
    for det in get_associated_ppe_detections(detections, compliance_results):
        _draw_box(output, det.bbox, ppe_color, det.class_name, thickness, text_scale)

    # 繪製 person 框 + Track ID + 最終確認狀態
    for det in detections:
        if det.class_name != "person":
            continue

        track = track_by_id.get(det.track_id) if det.track_id is not None else None
        color, label = _resolve_draw_style(track, det.track_id, colors)
        _draw_box(output, det.bbox, color, label, thickness, text_scale)

    return output


def draw_frame(
    frame: np.ndarray,
    detections: list[Detection],
    tracks: list[TrackSnapshot],
    visualizer_cfg: dict,
    compliance_results: list[PersonCompliance] | None = None,
) -> np.ndarray:
    """向後相容包裝，請優先使用 visualize()。"""
    if compliance_results is None:
        compliance_results = []
    return visualize(frame, detections, tracks, compliance_results, visualizer_cfg)


def _resolve_draw_style(
    track: TrackSnapshot | None,
    track_id: int | None,
    colors: dict,
) -> tuple[tuple, str]:
    compliant = tuple(colors.get("compliant", [0, 255, 0]))
    violation = tuple(colors.get("violation", [0, 0, 255]))
    person_only = tuple(colors.get("person_only", [0, 165, 255]))

    if track is None:
        prefix = f"#{track_id}" if track_id is not None else "#?"
        return person_only, f"{prefix} {STATUS_LABELS[ComplianceStatus.PERSON_ONLY]}"

    status = track.temporal_status if track.confirmed_person else track.frame_status
    label = f"#{track.track_id} {STATUS_LABELS[status]}"

    if status == ComplianceStatus.COMPLIANT:
        return compliant, label
    if status == ComplianceStatus.PERSON_ONLY:
        return person_only, label
    return violation, label


def _draw_roi(
    image: np.ndarray,
    roi: tuple,
    color: tuple,
    thickness: int,
) -> None:
    x1, y1, x2, y2 = map(int, roi)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, max(1, thickness - 1))


def _draw_box(
    image: np.ndarray,
    bbox: tuple,
    color: tuple,
    label: str,
    thickness: int,
    text_scale: float,
) -> None:
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        image,
        label,
        (x1, max(y1 - 8, 16)),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_scale,
        color,
        max(1, thickness - 1),
        cv2.LINE_AA,
    )
