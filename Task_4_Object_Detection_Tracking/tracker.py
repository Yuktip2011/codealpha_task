"""
tracker.py
----------
DeepSORT-based multi-object tracking module.
Wraps deep_sort_realtime to assign stable tracking IDs across frames.

Python 3.14 Compatibility Note:
    setuptools 70+ removed pkg_resources. DeepSORT's built-in MobileNet
    embedder imports pkg_resources internally. We work around this by using
    embedder=None and supplying our own lightweight histogram embeddings,
    which keeps tracking stable without any network downloads.
"""

import numpy as np
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

# ── Embedding size for our custom histogram embedder ─────────────────────────
_EMB_DIM = 128   # 4 channels × 32 bins


def _make_embedding(frame: np.ndarray, bbox: list[int]) -> np.ndarray:
    """
    Compute a lightweight appearance embedding for a detected crop.

    Uses a normalised colour histogram (HSV + grayscale) concatenated into
    a fixed-length vector. This requires no model weights or internet access
    and runs fast on CPU.

    Parameters
    ----------
    frame : np.ndarray  BGR frame
    bbox  : [x1, y1, x2, y2]

    Returns
    -------
    np.ndarray  shape (128,)  L2-normalised embedding
    """
    x1, y1, x2, y2 = bbox
    # Clamp to frame bounds
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros(_EMB_DIM, dtype=np.float32)

    # Resize to fixed size for consistent histogram
    crop = cv2.resize(crop, (64, 128))
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    bins = 32
    # H, S, V histograms (32 bins each) + grayscale (32 bins) = 128 dims
    h_hist = cv2.calcHist([hsv],  [0], None, [bins], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv],  [1], None, [bins], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv],  [2], None, [bins], [0, 256]).flatten()
    g_hist = cv2.calcHist([gray], [0], None, [bins], [0, 256]).flatten()

    emb = np.concatenate([h_hist, s_hist, v_hist, g_hist]).astype(np.float32)

    # L2 normalise
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb /= norm
    return emb


class ObjectTracker:
    """
    Wraps DeepSORT to maintain object identities across video frames.

    Uses a custom colour-histogram embedder (no pretrained weights needed)
    for compatibility with Python 3.14 where pkg_resources is unavailable.

    Parameters
    ----------
    max_age : int
        Maximum number of frames a track is kept alive without a detection.
    n_init : int
        Consecutive detections before a track is confirmed.
    max_cosine_distance : float
        Re-ID threshold for cosine-distance metric.
    """

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_cosine_distance: float = 0.4,
    ):
        # embedder=None → we supply our own embeddings via embeds= argument
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            nn_budget=100,
            override_track_class=None,
            embedder=None,   # custom embedder – avoids pkg_resources dependency
            half=False,
            bgr=True,
        )
        self._active_ids: set[int] = set()

    # ── public ───────────────────────────────────────────────────────────────
    def update(
        self, detections: list[dict], frame: np.ndarray
    ) -> list[dict]:
        """
        Feed new detections into DeepSORT and return confirmed tracks.

        Parameters
        ----------
        detections : list[dict]
            Output of ObjectDetector.detect().
        frame : np.ndarray
            Current BGR video frame.

        Returns
        -------
        list of dict with keys:
            track_id    : int
            bbox        : [x1, y1, x2, y2]
            class_name  : str
            confidence  : float
        """
        if not detections:
            # Still call update so internal Kalman filters can age out tracks
            try:
                self.tracker.update_tracks([], frame=frame, embeds=[])
            except Exception:
                pass
            self._active_ids.clear()
            return []

        # Build DeepSORT input + compute appearance embeddings
        ds_input: list[tuple] = []
        embeds:   list[np.ndarray] = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            w, h = x2 - x1, y2 - y1
            ds_input.append(
                ([x1, y1, w, h], det["confidence"], det["class_name"])
            )
            embeds.append(_make_embedding(frame, det["bbox"]))

        raw_tracks = self.tracker.update_tracks(ds_input, frame=frame, embeds=embeds)

        confirmed: list[dict] = []
        self._active_ids = set()

        for track in raw_tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = (
                int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            )
            track_id  = int(track.track_id)
            cls_name  = track.get_det_class() or "unknown"
            conf      = track.get_det_conf() or 0.0

            self._active_ids.add(track_id)
            confirmed.append(
                {
                    "track_id":   track_id,
                    "bbox":       [x1, y1, x2, y2],
                    "class_name": cls_name,
                    "confidence": float(conf),
                }
            )
        return confirmed

    @property
    def active_ids(self) -> set[int]:
        """Currently active (confirmed) tracking IDs."""
        return self._active_ids

    @property
    def active_count(self) -> int:
        return len(self._active_ids)


