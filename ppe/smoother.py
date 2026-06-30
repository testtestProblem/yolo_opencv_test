from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from ppe.types import ComplianceStatus, PersonCompliance, STATUS_LABELS


@dataclass
class FrameRecord:
    has_person: bool = True
    has_helmet: bool = False
    has_vest: bool = False
    frame_status: ComplianceStatus = ComplianceStatus.NO_PPE


@dataclass
class TrackSnapshot:
    track_id: int
    person_bbox: tuple[float, float, float, float]
    frame_status: ComplianceStatus
    confirmed_person: bool
    confirmed_helmet: bool
    confirmed_vest: bool
    person_hits: int
    helmet_hits: int
    vest_hits: int
    window_size: int
    confirm_threshold: int
    ppe_confirm_threshold: int
    temporal_status: ComplianceStatus

    @property
    def status_label(self) -> str:
        return STATUS_LABELS[self.temporal_status]

    @property
    def frame_status_label(self) -> str:
        return STATUS_LABELS[self.frame_status]


@dataclass
class _TrackState:
    track_id: int
    bbox: tuple[float, float, float, float]
    history: deque = field(default_factory=deque)
    missed_frames: int = 0

    def append(self, record: FrameRecord) -> None:
        self.history.append(record)
        self.missed_frames = 0

    def counts(self) -> tuple[int, int, int]:
        person_hits = sum(1 for r in self.history if r.has_person)
        helmet_hits = sum(1 for r in self.history if r.has_helmet)
        vest_hits = sum(1 for r in self.history if r.has_vest)
        return person_hits, helmet_hits, vest_hits

    def confirmed(
        self,
        person_threshold: int,
        ppe_threshold: int,
    ) -> tuple[bool, bool, bool]:
        person_hits, helmet_hits, vest_hits = self.counts()
        return (
            person_hits >= person_threshold,
            helmet_hits >= ppe_threshold,
            vest_hits >= ppe_threshold,
        )


class Smoother:
    """依 ByteTrack track_id 做時序平滑；不再做 IoU 重配對。"""

    def __init__(
        self,
        window_size: int = 120,
        confirm_threshold: int = 100,
        ppe_confirm_threshold: int = 80,
        max_missed_frames: int = 300,
    ):
        self.window_size = window_size
        self.confirm_threshold = confirm_threshold
        self.ppe_confirm_threshold = ppe_confirm_threshold
        self.max_missed_frames = max_missed_frames
        self._tracks: dict[int, _TrackState] = {}

    def reset(self) -> None:
        self._tracks.clear()

    def update(self, frame_results: list[PersonCompliance]) -> list[TrackSnapshot]:
        seen_ids: set[int] = set()

        for result in frame_results:
            if result.track_id is None:
                continue

            track_id = result.track_id
            seen_ids.add(track_id)

            if track_id not in self._tracks:
                self._tracks[track_id] = _TrackState(
                    track_id=track_id,
                    bbox=result.person_bbox,
                    history=deque(maxlen=self.window_size),
                )

            track = self._tracks[track_id]
            track.bbox = result.person_bbox
            track.append(
                FrameRecord(
                    has_helmet=result.has_helmet,
                    has_vest=result.has_vest,
                    frame_status=result.status,
                )
            )

        for track_id in list(self._tracks.keys()):
            if track_id not in seen_ids:
                self._tracks[track_id].missed_frames += 1
                if self._tracks[track_id].missed_frames > self.max_missed_frames:
                    del self._tracks[track_id]

        return [
            self._build_snapshot(track)
            for track in self._tracks.values()
            if track.track_id in seen_ids
        ]

    def _build_snapshot(self, track: _TrackState) -> TrackSnapshot:
        person_hits, helmet_hits, vest_hits = track.counts()
        confirmed_person, confirmed_helmet, confirmed_vest = track.confirmed(
            self.confirm_threshold,
            self.ppe_confirm_threshold,
        )

        latest = track.history[-1] if track.history else FrameRecord()
        temporal_status = _resolve_temporal_status(
            confirmed_person, confirmed_helmet, confirmed_vest
        )

        return TrackSnapshot(
            track_id=track.track_id,
            person_bbox=track.bbox,
            frame_status=latest.frame_status,
            confirmed_person=confirmed_person,
            confirmed_helmet=confirmed_helmet,
            confirmed_vest=confirmed_vest,
            person_hits=person_hits,
            helmet_hits=helmet_hits,
            vest_hits=vest_hits,
            window_size=self.window_size,
            confirm_threshold=self.confirm_threshold,
            ppe_confirm_threshold=self.ppe_confirm_threshold,
            temporal_status=temporal_status,
        )


def _resolve_temporal_status(
    confirmed_person: bool,
    confirmed_helmet: bool,
    confirmed_vest: bool,
) -> ComplianceStatus:
    if not confirmed_person:
        return ComplianceStatus.PERSON_ONLY

    if confirmed_helmet and confirmed_vest:
        return ComplianceStatus.COMPLIANT
    if not confirmed_helmet and confirmed_vest:
        return ComplianceStatus.NO_HELMET
    if confirmed_helmet and not confirmed_vest:
        return ComplianceStatus.NO_VEST
    if not confirmed_helmet and not confirmed_vest:
        return ComplianceStatus.PERSON_ONLY
    return ComplianceStatus.NO_PPE


# 向後相容別名
TemporalTracker = Smoother
