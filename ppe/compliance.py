from __future__ import annotations

from ppe.associator import find_best_match, split_person_roi
from ppe.types import ComplianceStatus, Detection, PersonCompliance


def check_compliance(
    detections: list[Detection],
    helmet_iou_threshold: float = 0.3,
    vest_iou_threshold: float = 0.3,
    min_person_height: int = 30,
) -> list[PersonCompliance]:
    persons = [d for d in detections if d.class_name == "person"]
    helmets = [d.bbox for d in detections if d.class_name == "helmet"]
    vests = [d.bbox for d in detections if d.class_name == "reflective_vest"]

    results: list[PersonCompliance] = []

    for person in persons:
        x1, y1, x2, y2 = person.bbox
        height = y2 - y1
        if height < min_person_height:
            continue

        head_roi, torso_roi = split_person_roi(person.bbox)
        has_helmet = find_best_match(helmets, head_roi, helmet_iou_threshold)
        has_vest = find_best_match(vests, torso_roi, vest_iou_threshold)
        status = _resolve_status(has_helmet, has_vest)

        results.append(
            PersonCompliance(
                person_bbox=person.bbox,
                status=status,
                has_helmet=has_helmet,
                has_vest=has_vest,
                confidence=person.confidence,
            )
        )

    return results


def _resolve_status(has_helmet: bool, has_vest: bool) -> ComplianceStatus:
    if has_helmet and has_vest:
        return ComplianceStatus.COMPLIANT
    if not has_helmet and has_vest:
        return ComplianceStatus.NO_HELMET
    if has_helmet and not has_vest:
        return ComplianceStatus.NO_VEST
    return ComplianceStatus.NO_PPE
