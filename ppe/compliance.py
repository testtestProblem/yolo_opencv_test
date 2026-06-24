from __future__ import annotations

from ppe.associator import filter_associated_ppe, find_ppe_in_region, split_person_roi
from ppe.filters import is_valid_person
from ppe.types import ComplianceStatus, Detection, PersonCompliance


def check_compliance(
    detections: list[Detection],
    frame_shape: tuple,
    filters_cfg: dict,
    association_cfg: dict | None = None,
) -> list[PersonCompliance]:
    """
    逐幀合規判定（CLAUDE.md §3–§4）。

    - 幾何過濾後的 person 才進入判定
    - helmet / vest 中心點落在該工人 person 框內即視為關聯
    - Head / Torso ROI 僅供視覺化參考
    """
    association_cfg = association_cfg or {}
    head_ratio = association_cfg.get("head_ratio", 1.0 / 3.0)
    torso_ratio = association_cfg.get("torso_ratio", 0.5)

    persons = [d for d in detections if d.class_name == "person"]
    helmets = [d.bbox for d in detections if d.class_name == "helmet"]
    vests = [d.bbox for d in detections if d.class_name == "reflective_vest"]

    results: list[PersonCompliance] = []

    for person in persons:
        if not is_valid_person(person.bbox, frame_shape, filters_cfg):
            continue

        head_roi, torso_roi = split_person_roi(
            person.bbox,
            head_ratio=head_ratio,
            torso_ratio=torso_ratio,
        )

        # 判定條件：PPE 中心點落在該工人的 person 框內
        has_helmet = find_ppe_in_region(helmets, person.bbox)
        has_vest = find_ppe_in_region(vests, person.bbox)
        status = _resolve_status(has_helmet, has_vest)

        results.append(
            PersonCompliance(
                person_bbox=person.bbox,
                status=status,
                has_helmet=has_helmet,
                has_vest=has_vest,
                confidence=person.confidence,
                track_id=person.track_id,
                head_roi=head_roi,
                torso_roi=torso_roi,
            )
        )

    return results


def get_associated_ppe_detections(
    detections: list[Detection],
    compliance_results: list[PersonCompliance],
) -> list[Detection]:
    """回傳已關聯到至少一位工人的 PPE 偵測（孤立裝備過濾）。"""
    person_bboxes = [r.person_bbox for r in compliance_results]
    return filter_associated_ppe(detections, person_bboxes)


def _resolve_status(has_helmet: bool, has_vest: bool) -> ComplianceStatus:
    if has_helmet and has_vest:
        return ComplianceStatus.COMPLIANT
    if not has_helmet and has_vest:
        return ComplianceStatus.NO_HELMET
    if has_helmet and not has_vest:
        return ComplianceStatus.NO_VEST
    return ComplianceStatus.NO_PPE
