from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

from config import project_root
from ppe.types import Detection


class YOLOEngine:
    """雙模型推理：person 用 ByteTrack 取得穩定 track_id，PPE 用 predict。"""

    def __init__(self, config: dict):
        root = project_root()
        person_cfg = config["models"]["person"]
        ppe_cfg = config["models"]["ppe"]

        self.person_model = YOLO(str(root / person_cfg["path"]))
        self.ppe_model = YOLO(str(root / ppe_cfg["path"]))

        self.person_conf = person_cfg["conf_threshold"]
        self.person_iou = person_cfg["iou_threshold"]
        self.person_imgsz = person_cfg.get("imgsz", 640)
        self.person_class_map = person_cfg.get("class_map", {"person": "person"})
        self.person_target = person_cfg.get("target_class", "person")
        self.person_tracker = person_cfg.get("tracker", "bytetrack.yaml")

        self.ppe_conf = ppe_cfg["conf_threshold"]
        self.ppe_iou = ppe_cfg["iou_threshold"]
        self.ppe_imgsz = ppe_cfg.get("imgsz", 640)
        self.ppe_class_map = ppe_cfg.get("class_map", {})

    def reset(self) -> None:
        """載入新影片時重置 ByteTrack 內部狀態。"""
        predictor = getattr(self.person_model, "predictor", None)
        if predictor is not None:
            predictor.trackers = None

    def predict(self, frame) -> list[Detection]:
        person_detections = self._run_track(frame)
        ppe_detections = self._run_model(
            self.ppe_model,
            frame,
            self.ppe_conf,
            self.ppe_iou,
            self.ppe_imgsz,
            self.ppe_class_map,
        )
        return person_detections + ppe_detections

    def _run_track(self, frame) -> list[Detection]:
        results = self.person_model.track(
            frame,
            persist=True,
            tracker=self.person_tracker,
            conf=self.person_conf,
            iou=self.person_iou,
            imgsz=self.person_imgsz,
            verbose=False,
        )
        detections: list[Detection] = []
        result = results[0]
        if result.boxes is None:
            return detections

        names = self.person_model.names
        for box in result.boxes:
            cls_id = int(box.cls.item())
            raw_name = names.get(cls_id, str(cls_id))
            mapped_name = self.person_class_map.get(raw_name, raw_name)
            if mapped_name not in {self.person_target, "person"} and raw_name not in {
                self.person_target,
                "person",
            }:
                continue

            track_id = None
            if box.id is not None:
                track_id = int(box.id.item())

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    class_name=mapped_name,
                    bbox=(x1, y1, x2, y2),
                    confidence=float(box.conf.item()),
                    track_id=track_id,
                )
            )
        return detections

    def _run_model(
        self,
        model: YOLO,
        frame,
        conf: float,
        iou: float,
        imgsz: int,
        class_map: dict,
        allowed: set[str] | None = None,
    ) -> list[Detection]:
        results = model.predict(
            frame,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False,
        )
        detections: list[Detection] = []
        result = results[0]
        if result.boxes is None:
            return detections

        names = model.names
        for box in result.boxes:
            cls_id = int(box.cls.item())
            raw_name = names.get(cls_id, str(cls_id))
            mapped_name = class_map.get(raw_name, raw_name)
            if allowed and mapped_name not in allowed and raw_name not in allowed:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                Detection(
                    class_name=mapped_name,
                    bbox=(x1, y1, x2, y2),
                    confidence=float(box.conf.item()),
                    track_id=None,
                )
            )
        return detections
