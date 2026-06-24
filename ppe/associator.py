from __future__ import annotations

from ppe.types import Detection


def compute_iou(box_a: tuple, box_b: tuple) -> float:
    """保留供測試或備援。"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def box_center(bbox: tuple) -> tuple[float, float]:
    """計算邊界框中心點 (cx, cy)。"""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def is_center_inside(box: tuple, region: tuple) -> bool:
    """判定 box 中心點是否落在 region 矩形內（含邊界）。"""
    cx, cy = box_center(box)
    rx1, ry1, rx2, ry2 = region
    return rx1 <= cx <= rx2 and ry1 <= cy <= ry2


def split_person_roi(
    person_bbox: tuple,
    head_ratio: float = 1.0 / 3.0,
    torso_ratio: float = 0.5,
) -> tuple[tuple, tuple]:
    """
    依 person 高度垂直切割 ROI（供視覺化參考，CLAUDE.md §3）。

    h = y2 - y1：
      Head ROI  = (x1, y1, x2, y1 + h/3)           頂部 1/3
      Torso ROI = (x1, y1 + h/3, x2, y1 + h/3 + h/2)  中段 1/2
    """
    x1, y1, x2, y2 = person_bbox
    h = y2 - y1

    head_bottom = y1 + h * head_ratio
    head_roi = (x1, y1, x2, head_bottom)

    torso_top = head_bottom
    torso_bottom = torso_top + h * torso_ratio
    torso_roi = (x1, torso_top, x2, torso_bottom)

    return head_roi, torso_roi


def find_ppe_in_region(ppe_boxes: list[tuple], region: tuple) -> bool:
    """任一 PPE 框中心點落在 region 內即視為關聯成功。"""
    for box in ppe_boxes:
        if is_center_inside(box, region):
            return True
    return False


def filter_associated_ppe(
    detections: list[Detection],
    person_bboxes: list[tuple],
) -> list[Detection]:
    """孤立裝備過濾：僅保留中心點落在任一 person 框內的 PPE。"""
    if not person_bboxes:
        return []

    associated: list[Detection] = []
    for det in detections:
        if det.class_name not in ("helmet", "reflective_vest"):
            continue
        if any(is_center_inside(det.bbox, pb) for pb in person_bboxes):
            associated.append(det)
    return associated
