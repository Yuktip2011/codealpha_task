"""
detector.py
-----------
YOLOv8-based object detection module.
Handles loading the pretrained model and running inference on video frames.
"""

import numpy as np
from ultralytics import YOLO

# ── COCO class names (80 classes) ──────────────────────────────────────────────
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

# Classes considered "vehicles"
VEHICLE_CLASSES = {"bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"}

# Color palette for bounding boxes (BGR)
COLORS = [
    (0, 114, 189), (217, 83, 25), (237, 177, 32), (126, 47, 142),
    (119, 172, 48), (77, 190, 238), (162, 20, 47), (76, 76, 76),
    (153, 153, 153), (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (0, 255, 255), (255, 0, 255),
]


class ObjectDetector:
    """
    Wraps YOLOv8 nano for object detection.

    Parameters
    ----------
    model_path : str
        Path or Ultralytics hub identifier (e.g. 'yolov8n.pt').
    confidence : float
        Minimum confidence score to keep a detection.
    device : str
        'cpu' or 'cuda' (falls back to cpu automatically).
    """

    def __init__(self, model_path: str = "yolov8n.pt",
                 confidence: float = 0.40,
                 device: str = "cpu"):
        self.confidence = confidence
        self.device = device
        self.model = self._load_model(model_path)

    # ── private ──────────────────────────────────────────────────────────────
    def _load_model(self, model_path: str) -> YOLO:
        """Load (and optionally download) the YOLOv8 model."""
        try:
            model = YOLO(model_path)
            return model
        except Exception as exc:
            raise RuntimeError(f"Failed to load YOLO model '{model_path}': {exc}") from exc

    # ── public ───────────────────────────────────────────────────────────────
    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run YOLOv8 inference on a single BGR frame.

        Returns
        -------
        list of dict with keys:
            bbox        : [x1, y1, x2, y2]  (ints)
            confidence  : float
            class_id    : int
            class_name  : str
        """
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )

        detections: list[dict] = []
        if not results:
            return detections

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = (
                    COCO_CLASSES[cls_id]
                    if cls_id < len(COCO_CLASSES)
                    else f"class_{cls_id}"
                )
                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": conf,
                        "class_id": cls_id,
                        "class_name": cls_name,
                    }
                )
        return detections

    def get_class_color(self, class_id: int) -> tuple[int, int, int]:
        """Return a consistent BGR color for a given class id."""
        return COLORS[class_id % len(COLORS)]

    @staticmethod
    def count_people(detections: list[dict]) -> int:
        return sum(1 for d in detections if d["class_name"] == "person")

    @staticmethod
    def count_vehicles(detections: list[dict]) -> int:
        return sum(1 for d in detections if d["class_name"] in VEHICLE_CLASSES)

