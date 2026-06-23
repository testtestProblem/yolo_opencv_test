from __future__ import annotations


def compute_iou(box_a: tuple, box_b: tuple) -> float:
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


def split_person_roi(person_bbox: tuple) -> tuple[tuple, tuple]:
    x1, y1, x2, y2 = person_bbox
    h = y2 - y1
    third = h / 3.0
    head_roi = (x1, y1, x2, y1 + third)
    torso_roi = (x1, y1 + third, x2, y1 + 2 * third)
    return head_roi, torso_roi


def find_best_match(
    ppe_boxes: list[tuple],
    roi: tuple,
    threshold: float,
) -> bool:
    for box in ppe_boxes:
        if compute_iou(box, roi) > threshold:
            return True
    return False
