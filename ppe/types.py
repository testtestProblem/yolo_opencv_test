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
    ComplianceStatus.COMPLIANT: "Compliant",
    ComplianceStatus.NO_HELMET: "No Hard Hat",
    ComplianceStatus.NO_VEST: "No Safety Vest",
    ComplianceStatus.NO_PPE: "Missing Hard Hat and Safety Vest",
    ComplianceStatus.PERSON_ONLY: "Person Only",
}


@dataclass
class Detection:
    class_name: str
    bbox: tuple[float, float, float, float]
    confidence: float
    track_id: int | None = None


@dataclass
class PersonCompliance:
    person_bbox: tuple[float, float, float, float]
    status: ComplianceStatus
    has_helmet: bool
    has_vest: bool
    confidence: float
    track_id: int | None = None
    head_roi: tuple[float, float, float, float] | None = None
    torso_roi: tuple[float, float, float, float] | None = None
