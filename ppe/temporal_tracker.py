from __future__ import annotations

from dataclasses import dataclass, field
from collections import deque

from ppe.associator import compute_iou
from ppe.types import ComplianceStatus, PersonCompliance, STATUS_LABELS


@dataclass
class FrameRecord:
    has_person: bool = True
    has_helmet: bool = False
    has_vest: bool = False


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
    temporal_status: ComplianceStatus

    @property
    def status_label(self) -> str:
        return STATUS_LABELS[self.temporal_status]

    @property
    def frame_status_label(self) -> str:
        return STATUS_LABELS[self.frame_status]


@dataclass
class _PersonTrack:
    track_id: int
    bbox: tuple[float, float, float, float]
    history: deque = field(default_factory=deque)
    missed_frames: int = 0

    def append(self, has_helmet: bool, has_vest: bool) -> None:
        self.history.append(FrameRecord(has_helmet=has_helmet, has_vest=has_vest))
        self.missed_frames = 0

    def counts(self) -> tuple[int, int, int]:
        person_hits = sum(1 for r in self.history if r.has_person)
        helmet_hits = sum(1 for r in self.history if r.has_helmet)
        vest_hits = sum(1 for r in self.history if r.has_vest)
        return person_hits, helmet_hits, vest_hits

    def confirmed(self, confirm_threshold: int) -> tuple[bool, bool, bool]:
        person_hits, helmet_hits, vest_hits = self.counts()
        return (
            person_hits >= confirm_threshold,
            helmet_hits >= confirm_threshold,
            vest_hits >= confirm_threshold,
        )


class TemporalTracker:
    """100 幀滑動窗口，80 幀命中才確認偵測結果。"""

    def __init__(
        self,
        window_size: int = 100,
        confirm_threshold: int = 80,
        track_iou_threshold: float = 0.3,
        max_missed_frames: int = 30,
    ):
        self.window_size = window_size
        self.confirm_threshold = confirm_threshold
        self.track_iou_threshold = track_iou_threshold
        self.max_missed_frames = max_missed_frames
        self._tracks: list[_PersonTrack] = []
        self._next_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1

    def update(self, frame_results: list[PersonCompliance]) -> list[TrackSnapshot]:
        matched_track_indices: set[int] = set()

        for result in frame_results:
            track_idx = self._match_track(result.person_bbox)
            if track_idx is not None:
                track = self._tracks[track_idx]
                track.bbox = result.person_bbox
                track.append(result.has_helmet, result.has_vest)
                matched_track_indices.add(track_idx)
            else:
                history: deque = deque(maxlen=self.window_size)
                history.append(
                    FrameRecord(has_helmet=result.has_helmet, has_vest=result.has_vest)
                )
                self._tracks.append(
                    _PersonTrack(
                        track_id=self._next_id,
                        bbox=result.person_bbox,
                        history=history,
                    )
                )
                self._next_id += 1
                matched_track_indices.add(len(self._tracks) - 1)

        for idx, track in enumerate(self._tracks):
            if idx not in matched_track_indices:
                track.missed_frames += 1

        self._tracks = [
            t for t in self._tracks if t.missed_frames <= self.max_missed_frames
        ]

        return [self._build_snapshot(track) for track in self._tracks]

    def _match_track(self, bbox: tuple) -> int | None:
        best_idx = None
        best_iou = self.track_iou_threshold
        for idx, track in enumerate(self._tracks):
            iou = compute_iou(bbox, track.bbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        return best_idx

    def _build_snapshot(self, track: _PersonTrack) -> TrackSnapshot:
        person_hits, helmet_hits, vest_hits = track.counts()
        confirmed_person, confirmed_helmet, confirmed_vest = track.confirmed(
            self.confirm_threshold
        )

        latest = track.history[-1] if track.history else FrameRecord()
        frame_status = _resolve_status(latest.has_helmet, latest.has_vest)
        temporal_status = _resolve_temporal_status(
            confirmed_person, confirmed_helmet, confirmed_vest
        )

        return TrackSnapshot(
            track_id=track.track_id,
            person_bbox=track.bbox,
            frame_status=frame_status,
            confirmed_person=confirmed_person,
            confirmed_helmet=confirmed_helmet,
            confirmed_vest=confirmed_vest,
            person_hits=person_hits,
            helmet_hits=helmet_hits,
            vest_hits=vest_hits,
            window_size=self.window_size,
            confirm_threshold=self.confirm_threshold,
            temporal_status=temporal_status,
        )


def _resolve_status(has_helmet: bool, has_vest: bool) -> ComplianceStatus:
    if has_helmet and has_vest:
        return ComplianceStatus.COMPLIANT
    if not has_helmet and has_vest:
        return ComplianceStatus.NO_HELMET
    if has_helmet and not has_vest:
        return ComplianceStatus.NO_VEST
    return ComplianceStatus.NO_PPE


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
