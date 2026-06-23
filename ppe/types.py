from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ComplianceStatus(Enum):
    COMPLIANT = "compliant"
    NO_HELMET = "no_helmet"
    NO_VEST = "no_vest"
    NO_PPE = "no_ppe"
    PERSON_ONLY = "person_only"


STATUS_LABELS = {
    ComplianceStatus.COMPLIANT: "合規",
    ComplianceStatus.NO_HELMET: "未戴安全帽",
    ComplianceStatus.NO_VEST: "未穿反光背心",
    ComplianceStatus.NO_PPE: "缺少安全帽與反光背心",
    ComplianceStatus.PERSON_ONLY: "僅偵測到 person",
}


@dataclass
class Detection:
    class_name: str
    bbox: tuple[float, float, float, float]
    confidence: float


@dataclass
class PersonCompliance:
    person_bbox: tuple[float, float, float, float]
    status: ComplianceStatus
    has_helmet: bool
    has_vest: bool
    confidence: float
