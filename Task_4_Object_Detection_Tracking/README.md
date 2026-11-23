# 🎯 AI-Powered Object Detection & Tracking System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![DeepSORT](https://img.shields.io/badge/Tracking-DeepSORT-green)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![OpenCV](https://img.shields.io/badge/Vision-OpenCV-yellow)

---

## 📌 Introduction

This project is a **real-time AI object detection and tracking system** built with Python. It uses the state-of-the-art **YOLOv8 Nano** model for lightning-fast object detection across 80 COCO categories, combined with **DeepSORT** for multi-object tracking — assigning stable unique IDs to each detected entity across video frames.

The entire system is wrapped in a beautiful, professional **Streamlit dashboard** that supports both **live webcam** input and **uploaded video files**, with real-time statistics and the ability to **download the processed output video**.

---

## 🎯 Objectives

- Detect objects in real time from webcam or video files using YOLOv8.
- Track each detected object with a persistent ID using DeepSORT.
- Visualise annotated video with bounding boxes, labels, and confidence scores.
- Display live statistics: total objects, people, vehicles, active tracks.
- Save and allow download of the annotated output video (MP4).

---

## ✨ Features

| Feature | Detail |
|---|---|
| 🤖 YOLOv8 Nano model | Fast inference, pretrained on COCO 80 classes |
| 🔍 DeepSORT tracking | Stable per-object IDs across frames |
| 📹 Webcam support | Live detection from your default camera |
| 📂 Video upload | Supports MP4, AVI, MOV, MKV, WEBM |
| 🎨 Annotated output | Bounding boxes, corner accents, label overlays |
| 📊 Live statistics | Total, people, vehicles, tracked count per frame |
| 🏷️ Track ID pills | Visual list of all assigned tracking IDs |
| 💾 MP4 export | Download the processed video after each session |
| ⚙️ Confidence slider | Adjust detection threshold interactively |
| 🌙 Dark UI | Professional dark-mode Streamlit dashboard |

---

## 🛠️ Technologies

| Technology | Version | Role |
|---|---|---|
| Python | 3.10+ | Core language |
| [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) | ≥ 8.0 | Object detection |
| [DeepSORT Realtime](https://github.com/levan92/deep_sort_realtime) | ≥ 1.3 | Multi-object tracking |
| [OpenCV](https://opencv.org/) | ≥ 4.8 | Frame processing & annotation |
| [Streamlit](https://streamlit.io/) | ≥ 1.28 | Web dashboard |
| NumPy | ≥ 1.24 | Array operations |

---

## 📁 Project Structure

```
AI_Object_Detection_Tracking/
│
├── app.py           ← Streamlit dashboard (UI + processing loop)
├── detector.py      ← YOLOv8 detection module
├── tracker.py       ← DeepSORT tracking module
├── requirements.txt ← Python dependencies
├── README.md        ← This file
└── .gitignore       ← Git ignore rules
```

> **Note:** `yolov8n.pt` is downloaded automatically on first run (~6 MB).

---

## 🚀 Installation

### 1. Clone or copy the project

```bash
cd AI_Object_Detection_Tracking
```

### 2. Create and activate a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> On first run the YOLOv8 Nano weights (`yolov8n.pt`, ~6 MB) are downloaded
> automatically from the Ultralytics CDN.

---

## ▶️ Running the Application

```bash
streamlit run app.py
```

Streamlit will open the dashboard in your default browser at `http://localhost:8501`.

---

## 🖥️ Usage

1. **Choose input source** in the left sidebar:  
   - **Webcam** – uses your system's default camera (index 0).  
   - **Video File** – upload an MP4 / AVI / MOV / MKV file.

2. **Adjust the detection confidence** slider (default: 0.40).

3. Click **▶ Start Detection** to begin processing.

4. Watch the annotated live feed and real-time statistics.

5. Click **⏹ Stop** at any time to end processing.

6. Use the **⬇️ Download MP4** button to save the annotated output.

---

## 📊 Dashboard Overview

### Live Feed Panel
- Annotated video frames with:
  - Coloured bounding boxes (per class)
  - Corner accent markers
  - Label overlay: `ID:N  class_name  XX%`
  - HUD bar: timestamp, total objects, people, vehicles, tracked count

### Statistics Panel (right column)
- **Total Objects** detected in the current frame
- **Active Tracks** (confirmed DeepSORT tracks)
- **👤 People** count
- **🚗 Vehicles** count
- **Frame counter**
- **All Track IDs** as coloured pills

---

## 🎨 Output Description

The annotated output video includes:

- **Bounding boxes** drawn with class-specific colours.
- **Corner accents** (L-shaped) at each corner of the box.
- **Label strip** showing `ID:{track_id}  {class}  {confidence%}`.
- **HUD overlay** at the top of every frame with live stats.

---

## 🔧 Configuration

| Parameter | Location | Default | Description |
|---|---|---|---|
| `confidence` | Sidebar slider | `0.40` | YOLO confidence threshold |
| `max_age` | `tracker.py` | `30` | Frames to keep a lost track alive |
| `n_init` | `tracker.py` | `3` | Frames before a track is confirmed |
| `model_path` | `detector.py` | `yolov8n.pt` | YOLO model weights file |

---

## 🐛 Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: ultralytics` | `pip install ultralytics` |
| `ModuleNotFoundError: deep_sort_realtime` | `pip install deep-sort-realtime` |
| Webcam not found | Check camera index; try `video_source = 1` in `app.py` |
| Slow on CPU | Reduce frame resolution or use a GPU with CUDA |
| Download button missing | Ensure at least a few frames were processed |

---

## 📄 License

This project is open-source and free to use for educational purposes.

---

## 🙏 Acknowledgements

- [Ultralytics](https://ultralytics.com/) for YOLOv8
- [levan92](https://github.com/levan92/deep_sort_realtime) for DeepSORT Realtime
- [Streamlit](https://streamlit.io/) for the web framework
- COCO dataset team for the 80-class benchmark

