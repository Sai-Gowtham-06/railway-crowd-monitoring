import os
from typing import Dict, List, Optional
import yaml
from pydantic import BaseModel, Field

class CameraConfig(BaseModel):
    source: str = "data/station_entry_live.mp4"
    camera_id: str = "CAM-P1-CENTRAL"
    station_name: str = "Central Railway Station"
    width: int = 1280
    height: int = 720
    fps: int = 25

class DetectorConfig(BaseModel):
    model_path: str = "yolov8n.pt"
    confidence: float = 0.35
    iou: float = 0.45
    device: str = "auto"
    classes: List[int] = Field(default_factory=lambda: [0])

class TrackerConfig(BaseModel):
    tracker_type: str = "bytetrack"
    track_activation_threshold: float = 0.30
    lost_track_buffer: int = 35
    minimum_matching_threshold: float = 0.70
    frame_rate: int = 25

class ROIConfig(BaseModel):
    name: str
    polygon: List[List[int]]
    physical_area_m2: float
    type: str = "monitored"  # "monitored" or "restricted"

class CongestionTier(BaseModel):
    max_density: Optional[float] = None
    min_density: Optional[float] = None
    label: str
    action: str
    color_bgr: List[int]

class PanicConfig(BaseModel):
    min_crowd_size: int = 4
    direction_entropy_threshold: float = 1.4
    speed_variance_multiplier: float = 2.2

class FallConfig(BaseModel):
    aspect_ratio_threshold: float = 0.85
    drop_speed_threshold: float = 30.0
    standstill_seconds: float = 1.2

class IntrusionConfig(BaseModel):
    restricted_zone_keys: List[str] = Field(default_factory=lambda: ["rail_bed_danger_zone"])

class AnomalyConfig(BaseModel):
    zscore_window: int = 60
    min_samples_for_zscore: int = 15
    density_zscore_threshold: float = 2.5
    speed_spike_zscore_threshold: float = 2.8
    panic: PanicConfig = Field(default_factory=PanicConfig)
    fall: FallConfig = Field(default_factory=FallConfig)
    intrusion: IntrusionConfig = Field(default_factory=IntrusionConfig)

class AlertingConfig(BaseModel):
    webhook_url: str = "http://127.0.0.1:8000/api/mock_webhook"
    webhook_enabled: bool = False
    cooldown_seconds: int = 8
    log_file: str = "logs/station_anomalies.jsonl"

class PrivacyConfig(BaseModel):
    anonymize_faces: bool = True
    blur_kernel_size: int = 25
    face_ratio_top: float = 0.25
    save_raw_frames: bool = False

class DashboardConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    stream_fps: int = 25

class AppConfig(BaseModel):
    camera: CameraConfig = Field(default_factory=CameraConfig)
    detector: DetectorConfig = Field(default_factory=DetectorConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    rois: Dict[str, ROIConfig] = Field(default_factory=dict)
    congestion_tiers: Dict[str, CongestionTier] = Field(default_factory=dict)
    anomaly: AnomalyConfig = Field(default_factory=AnomalyConfig)
    alerting: AlertingConfig = Field(default_factory=AlertingConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)


def load_config(config_path: str = "config/config.yaml") -> AppConfig:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return AppConfig(**data)
