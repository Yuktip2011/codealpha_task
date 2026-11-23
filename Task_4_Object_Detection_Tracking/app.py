"""
app.py
------
Streamlit dashboard for the AI-Powered Object Detection & Tracking System.

Run with:
    streamlit run app.py
"""

import os
import sys
import time
import tempfile
import threading
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="AI Object Detection & Tracking",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject custom CSS ──────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── App background ── */
    .stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 100%); }

    /* ── Hide default header ── */
    header[data-testid="stHeader"] { background: transparent; }

    /* ── Hero banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #1a1f2e 0%, #0e4178 50%, #1a1f2e 100%);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 32px 40px;
        margin-bottom: 24px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute; top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(ellipse at center, rgba(0,114,255,0.08) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-title {
        font-size: 2.4rem; font-weight: 800;
        background: linear-gradient(90deg, #58a6ff, #79c0ff, #a5d6ff);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0 0 8px;
    }
    .hero-subtitle {
        color: #8b949e; font-size: 1.0rem; margin: 0;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: linear-gradient(135deg, #1c2128, #21262d);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        transition: border-color .2s;
    }
    .metric-card:hover { border-color: #58a6ff; }
    .metric-value {
        font-size: 2.0rem; font-weight: 700; color: #58a6ff;
    }
    .metric-label {
        font-size: 0.78rem; color: #8b949e;
        text-transform: uppercase; letter-spacing: .08em;
    }

    /* ── Status badge ── */
    .status-badge {
        display: inline-block;
        padding: 4px 12px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600;
        letter-spacing: .05em;
    }
    .status-running { background:#0e4c23; color:#3fb950; border:1px solid #238636; }
    .status-stopped { background:#3b1a1a; color:#f85149; border:1px solid #8b1a1a; }
    .status-ready   { background:#1b2a3b; color:#79c0ff; border:1px solid #1f4068; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #161b22 !important;
        border-right: 1px solid #30363d;
    }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label { color: #c9d1d9; }

    /* ── Buttons ── */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all .2s;
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,.4); }

    /* ── Tracking ID pills ── */
    .id-pill {
        display: inline-block;
        background: #0e4178; color: #79c0ff;
        border: 1px solid #1f4068;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.75rem; font-weight: 600;
        margin: 2px;
    }

    /* ── Info box ── */
    .info-box {
        background: #1b2a3b;
        border: 1px solid #1f4068;
        border-radius: 10px;
        padding: 14px 18px;
        color: #c9d1d9;
        font-size: 0.85rem;
        line-height: 1.6;
    }

    /* ── Video container ── */
    .video-container {
        border: 2px solid #30363d;
        border-radius: 12px;
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Lazy-import heavy modules (avoid crash on import errors) ───────────────────
@st.cache_resource(show_spinner="Loading YOLOv8 model…")
def load_detector(confidence: float):
    from detector import ObjectDetector
    return ObjectDetector(model_path="yolov8n.pt", confidence=confidence)


@st.cache_resource
def load_tracker():
    from tracker import ObjectTracker
    return ObjectTracker()


# ── Annotation helpers ─────────────────────────────────────────────────────────
def _box_color(class_id: int) -> tuple[int, int, int]:
    """BGR color based on class_id."""
    palette = [
        (0, 114, 189), (217, 83, 25), (237, 177, 32),
        (126, 47, 142), (119, 172, 48), (77, 190, 238),
        (162, 20, 47), (76, 76, 76), (77, 100, 141),
        (255, 90, 95), (90, 200, 250), (88, 166, 92),
    ]
    return palette[class_id % len(palette)]


def annotate_frame(frame: np.ndarray, tracks: list, detector) -> np.ndarray:
    """
    Draw bounding boxes, tracking IDs, class names and confidence
    on the given frame. Returns the annotated frame.
    """
    overlay = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = t["bbox"]
        cls_name = t["class_name"]
        tid = t.get("track_id", -1)
        conf = t.get("confidence", 0.0)

        # Determine colour
        try:
            from detector import COCO_CLASSES
            cls_id = COCO_CLASSES.index(cls_name) if cls_name in COCO_CLASSES else 0
        except Exception:
            cls_id = 0
        color = _box_color(cls_id)

        # Filled bounding box with transparency
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    alpha = 0.12
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    for t in tracks:
        x1, y1, x2, y2 = t["bbox"]
        cls_name = t["class_name"]
        tid = t.get("track_id", -1)
        conf = t.get("confidence", 0.0)

        try:
            from detector import COCO_CLASSES
            cls_id = COCO_CLASSES.index(cls_name) if cls_name in COCO_CLASSES else 0
        except Exception:
            cls_id = 0
        color = _box_color(cls_id)

        # Border rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Corner accents
        corner_len = max(12, min(20, (x2 - x1) // 5))
        for cx, cy, dx, dy in [
            (x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)
        ]:
            cv2.line(frame, (cx, cy), (cx + dx * corner_len, cy), color, 3)
            cv2.line(frame, (cx, cy), (cx, cy + dy * corner_len), color, 3)

        # Label background
        label = f"ID:{tid}  {cls_name}  {conf:.0%}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = y1 - 8 if y1 - 8 > th + 4 else y2 + th + 8
        cv2.rectangle(
            frame,
            (x1, label_y - th - 6),
            (x1 + tw + 10, label_y + baseline - 2),
            color, -1,
        )
        cv2.putText(
            frame, label,
            (x1 + 5, label_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
    return frame


def add_hud(frame: np.ndarray, stats: dict) -> np.ndarray:
    """Overlay a heads-up-display with live statistics."""
    h, w = frame.shape[:2]
    # Dark strip at top
    cv2.rectangle(frame, (0, 0), (w, 44), (13, 17, 23), -1)
    cv2.line(frame, (0, 44), (w, 44), (48, 54, 61), 1)

    # FPS / timestamp
    ts = datetime.now().strftime("%H:%M:%S")
    cv2.putText(frame, f"AI Object Tracker  |  {ts}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (88, 166, 255), 1, cv2.LINE_AA)

    # Stats on right
    info = (f"Objects: {stats.get('total', 0)}   "
            f"People: {stats.get('people', 0)}   "
            f"Vehicles: {stats.get('vehicles', 0)}   "
            f"Tracked: {stats.get('tracked', 0)}")
    (tw, _), _ = cv2.getTextSize(info, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    cv2.putText(frame, info, (w - tw - 10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
    return frame


# ── Video writer helper ────────────────────────────────────────────────────────
def create_video_writer(output_path: str, fps: float, width: int, height: int):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))


# ── Core processing loop ───────────────────────────────────────────────────────
def process_video(source, detector, tracker, output_path: str,
                  frame_placeholder, stats_placeholder, stop_flag: threading.Event):
    """
    Read frames from `source` (int for webcam, str for file),
    run detection + tracking, annotate and display them.
    Writes output to `output_path`.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        st.error("❌ Cannot open video source. Check webcam or file path.")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = create_video_writer(output_path, fps, width, height)

    frame_count = 0
    all_track_ids: set[int] = set()

    while cap.isOpened() and not stop_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            break  # end of file

        frame_count += 1

        # ── Detection ──────────────────────────────────────────────────────
        detections = detector.detect(frame)

        # ── Tracking ───────────────────────────────────────────────────────
        tracks = tracker.update(detections, frame)
        for t in tracks:
            all_track_ids.add(t["track_id"])

        # ── Annotation ─────────────────────────────────────────────────────
        # Merge detection info into tracks for display
        total_objects  = len(detections)
        num_people     = detector.count_people(detections)
        num_vehicles   = detector.count_vehicles(detections)
        num_tracked    = len(tracks)

        stats = {
            "total": total_objects,
            "people": num_people,
            "vehicles": num_vehicles,
            "tracked": num_tracked,
            "frame": frame_count,
            "ids": sorted(all_track_ids),
        }

        annotated = annotate_frame(frame, tracks, detector)
        annotated = add_hud(annotated, stats)

        writer.write(annotated)

        # ── Display (convert BGR → RGB for Streamlit) ──────────────────────
        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb, channels="RGB", use_container_width=True)

        # ── Stats sidebar update ───────────────────────────────────────────
        with stats_placeholder.container():
            _render_live_stats(stats)

        # small delay to keep UI responsive
        time.sleep(0.01)

    cap.release()
    writer.release()
    return True


def _render_live_stats(stats: dict):
    """Render the live statistics panel."""
    cols = st.columns(2)
    with cols[0]:
        st.markdown(
            f"""<div class="metric-card">
                    <div class="metric-value">{stats.get('total', 0)}</div>
                    <div class="metric-label">Total Objects</div>
                </div>""",
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f"""<div class="metric-card">
                    <div class="metric-value">{stats.get('tracked', 0)}</div>
                    <div class="metric-label">Active Tracks</div>
                </div>""",
            unsafe_allow_html=True,
        )
    cols2 = st.columns(2)
    with cols2[0]:
        st.markdown(
            f"""<div class="metric-card">
                    <div class="metric-value">{stats.get('people', 0)}</div>
                    <div class="metric-label">👤 People</div>
                </div>""",
            unsafe_allow_html=True,
        )
    with cols2[1]:
        st.markdown(
            f"""<div class="metric-card">
                    <div class="metric-value">{stats.get('vehicles', 0)}</div>
                    <div class="metric-label">🚗 Vehicles</div>
                </div>""",
            unsafe_allow_html=True,
        )
    st.markdown("---")
    st.markdown(f"**Frame:** `{stats.get('frame', 0)}`")
    ids = stats.get("ids", [])
    if ids:
        pills = "".join(f'<span class="id-pill">#{i}</span>' for i in ids[-20:])
        st.markdown(f"**All Track IDs:**<br>{pills}", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar() -> dict:
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown("---")

        source = st.selectbox(
            "📹 Input Source",
            options=["Webcam", "Video File"],
            index=0,
        )

        confidence = st.slider(
            "🎯 Detection Confidence",
            min_value=0.10, max_value=0.90,
            value=0.40, step=0.05,
            help="Minimum confidence score for a detection to be kept.",
        )

        st.markdown("---")
        st.markdown("### 📂 Upload Video")
        uploaded_file = None
        if source == "Video File":
            uploaded_file = st.file_uploader(
                "Choose a video file",
                type=["mp4", "avi", "mov", "mkv", "webm"],
                help="Upload a video file for detection & tracking.",
            )

        st.markdown("---")
        st.markdown(
            """
            <div class="info-box">
            <b>🤖 Model:</b> YOLOv8 Nano (COCO 80 classes)<br>
            <b>🔍 Tracker:</b> DeepSORT<br>
            <b>🖥️ Backend:</b> OpenCV + Streamlit
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown(
            "<p style='color:#8b949e;font-size:0.75rem;text-align:center;'>"
            "AI Object Detection & Tracking v1.0</p>",
            unsafe_allow_html=True,
        )

    return {
        "source": source,
        "confidence": confidence,
        "uploaded_file": uploaded_file,
    }


# ── Main App ───────────────────────────────────────────────────────────────────
def main():
    # ── Hero banner ───────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="hero-banner">
            <div class="hero-title">🎯 AI Object Detection & Tracking</div>
            <p class="hero-subtitle">
                Real-time detection with YOLOv8 &nbsp;•&nbsp;
                Multi-object tracking with DeepSORT &nbsp;•&nbsp;
                Streamlit dashboard
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar config ────────────────────────────────────────────────────
    cfg = render_sidebar()

    # ── Session state ─────────────────────────────────────────────────────
    if "running" not in st.session_state:
        st.session_state.running = False
    if "output_path" not in st.session_state:
        st.session_state.output_path = None

    # ── Load model ────────────────────────────────────────────────────────
    try:
        detector = load_detector(cfg["confidence"])
    except Exception as e:
        st.error(f"❌ Failed to load detector: {e}")
        st.info("Make sure `ultralytics` is installed: `pip install ultralytics`")
        return

    try:
        tracker = load_tracker()
    except Exception as e:
        st.error(f"❌ Failed to load tracker: {e}")
        st.info("Make sure `deep-sort-realtime` is installed: `pip install deep-sort-realtime`")
        return

    # ── Layout: video | stats ────────────────────────────────────────────
    left_col, right_col = st.columns([3, 1])

    with left_col:
        st.markdown("### 🖥️ Live Feed")
        status_placeholder = st.empty()
        frame_placeholder  = st.empty()

        if st.session_state.running:
            status_placeholder.markdown(
                '<span class="status-badge status-running">● RUNNING</span>',
                unsafe_allow_html=True,
            )
        else:
            status_placeholder.markdown(
                '<span class="status-badge status-ready">○ READY</span>',
                unsafe_allow_html=True,
            )

    with right_col:
        st.markdown("### 📊 Statistics")
        stats_placeholder = st.empty()

    # ── Control buttons ───────────────────────────────────────────────────
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])

    with btn_col1:
        start_clicked = st.button(
            "▶ Start Detection",
            type="primary",
            disabled=st.session_state.running,
            use_container_width=True,
        )
    with btn_col2:
        stop_clicked = st.button(
            "⏹ Stop",
            type="secondary",
            disabled=not st.session_state.running,
            use_container_width=True,
        )

    # ── Stop handling ─────────────────────────────────────────────────────
    if stop_clicked:
        st.session_state.running = False
        st.session_state._stop_flag_set = True
        st.rerun()

    # ── Start handling ────────────────────────────────────────────────────
    if start_clicked:
        # Validate inputs
        if cfg["source"] == "Video File" and cfg["uploaded_file"] is None:
            st.warning("⚠️ Please upload a video file first.")
            return

        # Prepare source
        if cfg["source"] == "Webcam":
            video_source = 0  # default webcam
        else:
            # Save uploaded file to a temp location
            suffix = Path(cfg["uploaded_file"].name).suffix
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(cfg["uploaded_file"].read())
            tmp.flush()
            tmp.close()
            video_source = tmp.name

        # Prepare output file
        out_tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=".mp4",
            dir=tempfile.gettempdir(),
        )
        out_tmp.close()
        output_path = out_tmp.name
        st.session_state.output_path = output_path

        # Update state & UI
        st.session_state.running = True
        st.session_state._stop_flag_set = False
        status_placeholder.markdown(
            '<span class="status-badge status-running">● RUNNING</span>',
            unsafe_allow_html=True,
        )

        # Stop flag (checked inside loop)
        stop_flag = threading.Event()

        # Reset tracker for a fresh run
        try:
            from tracker import ObjectTracker
            fresh_tracker = ObjectTracker()
        except Exception:
            fresh_tracker = tracker

        # ── Run processing (blocking – Streamlit's model) ──────────────
        success = process_video(
            source=video_source,
            detector=detector,
            tracker=fresh_tracker,
            output_path=output_path,
            frame_placeholder=frame_placeholder,
            stats_placeholder=stats_placeholder,
            stop_flag=stop_flag,
        )

        st.session_state.running = False
        status_placeholder.markdown(
            '<span class="status-badge status-stopped">■ STOPPED</span>',
            unsafe_allow_html=True,
        )

        if success:
            st.success("✅ Processing complete!")

    # ── Download processed video ──────────────────────────────────────────
    out_path = st.session_state.get("output_path")
    if out_path and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        st.markdown("---")
        st.markdown("### 💾 Download Processed Video")
        with open(out_path, "rb") as f:
            st.download_button(
                label="⬇️ Download MP4",
                data=f,
                file_name="detection_output.mp4",
                mime="video/mp4",
                use_container_width=False,
            )

    # ── About section ─────────────────────────────────────────────────────
    with st.expander("ℹ️ About this project", expanded=False):
        st.markdown(
            """
            This system combines **YOLOv8** (one of the fastest object detectors) with
            **DeepSORT** tracking to maintain unique IDs for each detected object across
            consecutive frames.

            **Supported objects (COCO 80 classes):** person, car, bus, bicycle, motorcycle,
            truck, dog, cat, bird, and 70 more categories.

            **Workflow:**
            1. Each frame is passed to YOLOv8 Nano for fast detection.
            2. Detections are fed to DeepSORT which uses a CNN appearance embedding
               and a Kalman filter to match objects across frames.
            3. Annotated frames are displayed in real-time and written to an MP4 file.
            4. Statistics (people, vehicles, tracking IDs) are updated per frame.
            """
        )


if __name__ == "__main__":
    main()

