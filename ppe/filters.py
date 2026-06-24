from __future__ import annotations


def is_valid_person(
    bbox: tuple[float, float, float, float],
    frame_shape: tuple,
    cfg: dict,
) -> bool:
    """幾何過濾：過小、過大、異常長寬比的 person 框直接丟棄。"""
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1

    if height < cfg.get("min_person_height", 30):
        return False

    frame_h, frame_w = frame_shape[:2]
    frame_area = frame_w * frame_h
    if frame_area <= 0:
        return False

    if (width * height) / frame_area > cfg.get("max_area_ratio", 0.7):
        return False

    if height > 0 and (width / height) > cfg.get("max_aspect_ratio", 1.5):
        return False

    return True
