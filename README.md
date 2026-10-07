# RAIL-WATCH AI: Intelligent Crowd Monitoring & Anomaly Detection

A production-ready computer vision and mission-control surveillance system designed for railway station CCTV feeds. It provides real-time passenger detection, multi-object tracking, spatial crowd density calculation (`people/m²`), behavioral anomaly detection (overcrowding, sudden motion surges, chaotic panic dispersal, passenger falls, and restricted track incursions), privacy-preserving face blurring, structured event logging, and an interactive glassmorphic web dashboard.

---

## 1. Key Features

- **Detection**: Ultralytics YOLOv8n / YOLOv11n filtered for `person` class with NMS tuning for dense occlusions.
- **Tracking**: ByteTrack (`supervision.ByteTrack`) maintaining persistent IDs, centroid histories, velocity vectors, and aspect ratios.
- **Spatial Analytics**:
  - Per-ROI passenger count via ground-plane point-in-polygon assignment.
  - Crowd density calculation: $\text{Density} = \frac{\text{Passenger Count}}{\text{Physical Area } (m^2)}$.
  - Congestion tier classification per safety regulations:
    - `< 1.5 people/m²`: **Low (Normal Monitoring)**
    - `1.5 - 3.0 people/m²`: **Medium (Log & Observe)**
    - `3.0 - 4.5 people/m²`: **High (Alert Station Security)**
    - `> 4.5 people/m²`: **Critical (Immediate Action / Evacuation Warning)**
- **Behavioral Anomaly Detection**:
  - **Overcrowding**: Density breaches and sudden passenger surges ($Z > 2.5$).
  - **Sudden Crowd Movement**: Average flow velocity spike relative to moving baseline.
  - **Panic Dispersal**: Directional Shannon entropy spike combined with high speed variance.
  - **Passenger Fall**: Transition from standing ($H/W > 1.3$) to horizontal ($H/W \le 0.85$) + ground standstill persistence.
  - **Restricted Zone Intrusion**: Polygon breach detection into hazardous rail bed.
  - **Dynamic Baselines**: Rolling window Z-scores ($Z = \frac{X_t - \mu}{\sigma + \epsilon}$), avoiding rigid static thresholds.
- **Privacy by Design**:
  - Real-time Gaussian face blurring on upper body/head region before display or storage.
  - Persists only structured metadata (`.jsonl`), never storing raw surveillance footage unless explicitly configured.
- **Mission-Control Dashboard**:
  - High-performance FastAPI backend with WebSocket telemetry streaming (zero-polling).
  - Glassmorphic dark UI with live video stream, heatmaps, HUD gauges, bird's-eye 2D radar, and security incident alerts.

---

## 2. Directory Structure

```text
railway-crowd-monitoring/
├── config/
│   ├── config.yaml          # Camera sources, ROI polygons, thresholds, anomaly params
│   ├── loader.py            # Pydantic schema validation for application config
│   └── __init__.py
├── detection/
│   ├── detector.py          # YOLOv8n detector with adaptive synthetic benchmark fallback
│   └── __init__.py
├── tracking/
│   ├── tracker.py           # ByteTrack tracker with trajectory & velocity records
│   └── __init__.py
├── analytics/
│   ├── density.py           # Polygon spatial containment, density (p/m²), congestion tiers
│   ├── motion.py            # Motion vectors, mean flow speed, direction entropy
│   ├── heatmap.py           # Gaussian 2D crowd density accumulation with temporal decay
│   └── __init__.py
├── anomaly/
│   ├── statistical.py       # Rolling sliding window Z-score and EWMA baseline
│   ├── detectors.py         # Overcrowding, Fall, Panic, Surge, and Intrusion detectors
│   ├── engine.py            # Unified AnomalyEngine coordinating all detectors
│   └── __init__.py
├── alerting/
│   ├── alert_manager.py     # Cooldown suppression, structured JSONL logger, webhooks
│   └── __init__.py
├── api/
│   ├── app.py               # FastAPI server (REST endpoints, WebSocket, MJPEG video stream)
│   └── __init__.py
├── dashboard/
│   ├── index.html           # Mission-control dashboard layout
│   ├── styles.css           # Modern dark-mode glassmorphic design system
│   └── app.js               # WebSocket client, live gauges, alert feed, 2D radar canvas
├── tests/
│   ├── test_analytics.py    # Unit tests for density and motion logic
│   ├── test_anomaly.py      # Unit tests for statistical Z-score, fall, and panic detection
│   ├── generate_test_video.py # Synthetic CCTV clip generator with crowd scenarios
│   ├── test_milestone1.py   # Milestone 1 verification script
│   ├── test_milestone2.py   # Milestone 2 verification script
│   └── test_milestone3.py   # Milestone 3 verification script
├── logs/                    # Structured JSONL security alert records
├── data/                    # Video feeds / recorded test clips
├── pipeline.py              # Multi-threaded streaming vision pipeline
├── evaluate.py              # Benchmark evaluation and latency profiling script
├── requirements.txt         # Core Python dependencies
├── pyproject.toml           # Package metadata and build system
└── README.md                # Documentation and setup guide
```

---

## 3. Installation & Quickstart

### Prerequisites
- Python 3.10 to 3.14
- Optional: CUDA-capable NVIDIA GPU (automatically leveraged if available; CPU is fully supported)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Generate or Provide a Video Feed
Generate the synthetic CCTV test video with simulated crowd movement, falls, panic, and rail intrusions:
```bash
python tests/generate_test_video.py
```
*Or point `config/config.yaml` to any local video file or RTSP stream URL:*
```yaml
camera:
  source: "rtsp://admin:pass@192.168.1.100:554/stream1"  # Or local file path
  camera_id: "CAM-P1-CENTRAL"
  station_name: "Central Railway Station - Platform 1"
```

### Step 3: Run the Live Dashboard & Streaming Server
```bash
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```
Open your browser and navigate to:
👉 **`http://localhost:8000`**

---

## 4. Running Tests & Benchmarks

### Unit Test Suite
Execute the unit tests specifically verifying density calculation, threshold boundaries, motion vectors, rolling Z-score baselines, and passenger fall logic:
```bash
python -m pytest tests/test_analytics.py tests/test_anomaly.py -v
```

### Evaluation Benchmark
Profile end-to-end throughput (FPS), component latencies, tracking stability, and anomaly detection rates:
```bash
python evaluate.py --video data/sample_cctv.mp4
```
A complete Markdown evaluation report will be generated at `evaluation_report.md`.

---

## 5. Configuration Reference (`config/config.yaml`)

| Section | Parameter | Default | Description |
| :--- | :--- | :--- | :--- |
| `detector` | `model_path` | `yolov8n.pt` | Ultralytics model weights |
| `detector` | `confidence` | `0.35` | Minimum detection confidence threshold |
| `tracker` | `tracker_type` | `bytetrack` | Supervision ByteTrack algorithm |
| `rois` | `physical_area_m2` | Float | Calibrated real-world surface area in square meters |
| `rois` | `type` | `monitored` / `restricted` | Monitored concourses vs dangerous exclusion zones |
| `congestion_tiers` | `<1.5`, `1.5-3.0`, `3.0-4.5`, `>4.5` | Low/Med/High/Crit | Station congestion tier thresholds |
| `anomaly` | `zscore_window` | `60` frames | Rolling baseline window size |
| `anomaly.fall` | `aspect_ratio_threshold` | `0.85` | Person H/W aspect ratio indicating horizontal posture |
| `privacy` | `anonymize_faces` | `true` | Blurs face/head region before display and saving |
| `alerting` | `cooldown_seconds` | `8` seconds | Suppression window to prevent alert fatigue |

---

## 6. Privacy & Compliance

This system adheres strictly to privacy-by-design standards:
1. **Automated Anonymization**: The `PrivacyAnonymizer` automatically derives face/head coordinates from detected bounding boxes and applies dynamic Gaussian blurring.
2. **Metadata-Only Retention**: Only anonymized numerical telemetry (counts, densities, velocity vectors) and structured anomaly events are written to `logs/station_anomalies.jsonl`. Raw surveillance frames are never persisted to disk.
