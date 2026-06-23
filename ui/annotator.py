from __future__ import annotations

import cv2
import numpy as np

from ppe.temporal_tracker import TrackSnapshot
from ppe.types import ComplianceStatus, Detection, STATUS_LABELS


def draw_frame(
    frame: np.ndarray,
    detections: list[Detection],
    tracks: list[TrackSnapshot],
    visualizer_cfg: dict,
) -> np.ndarray:
    output = frame.copy()
    colors = visualizer_cfg.get("colors", {})
    thickness = visualizer_cfg.get("box_thickness", 2)
    text_scale = visualizer_cfg.get("text_scale", 0.6)

    ppe_color = tuple(colors.get("ppe", [255, 191, 0]))
    for det in detections:
        if det.class_name == "person":
            continue
        _draw_box(output, det.bbox, ppe_color, det.class_name, thickness, text_scale)

    track_by_bbox = {_bbox_key(t.person_bbox): t for t in tracks}

    for det in detections:
        if det.class_name != "person":
            continue
        track = _find_track(det.bbox, track_by_bbox)
        color, label = _resolve_draw_style(track, colors)
        _draw_box(output, det.bbox, color, label, thickness, text_scale)

    return output


def _find_track(bbox: tuple, track_by_bbox: dict) -> TrackSnapshot | None:
    key = _bbox_key(bbox)
    if key in track_by_bbox:
        return track_by_bbox[key]
    for track in track_by_bbox.values():
        if _bbox_close(bbox, track.person_bbox):
            return track
    return None


def _bbox_key(bbox: tuple) -> tuple:
    return tuple(round(v, 1) for v in bbox)


def _bbox_close(a: tuple, b: tuple, tol: float = 8.0) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def _resolve_draw_style(track: TrackSnapshot | None, colors: dict) -> tuple[tuple, str]:
    compliant = tuple(colors.get("compliant", [0, 255, 0]))
    violation = tuple(colors.get("violation", [0, 0, 255]))
    person_only = tuple(colors.get("person_only", [0, 165, 255]))

    if track is None:
        return person_only, STATUS_LABELS[ComplianceStatus.PERSON_ONLY]

    status = track.temporal_status if track.confirmed_person else track.frame_status
    label = f"#{track.track_id} {STATUS_LABELS[status]}"

    if status == ComplianceStatus.COMPLIANT:
        return compliant, label
    if status == ComplianceStatus.PERSON_ONLY:
        return person_only, label
    return violation, label


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
