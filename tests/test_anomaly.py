import os
import sys
import time
import pytest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anomaly.statistical import RollingBaseline
from anomaly.detectors import (
    OvercrowdingDetector,
    SuddenMovementDetector,
    PanicMotionDetector,
    PassengerFallDetector,
    IntrusionDetector
)
from alerting.alert_manager import AlertManager
from config.loader import AlertingConfig
from analytics.density import ROIMetrics
from analytics.motion import MotionMetrics
from tracking.tracker import TrackHistoryItem

def test_rolling_baseline_zscore():
    baseline = RollingBaseline(window_size=20, min_samples=10)
    
    # Feed steady state values around 10.0
    for _ in range(15):
        z = baseline.update(10.0 + np.random.normal(0, 0.1))
        
    assert baseline.is_warmed_up()
    assert pytest.approx(baseline.mean, abs=0.5) == 10.0
    
    # A massive spike to 30.0 should produce a high positive Z-score > 5
    spike_z = baseline.update(30.0)
    assert spike_z is not None
    assert spike_z > 5.0

def test_passenger_fall_detector():
    detector = PassengerFallDetector(
        aspect_ratio_threshold=0.85,
        standstill_seconds=1.0,
        frame_rate=25
    )
    
    track_id = 42
    rec = TrackHistoryItem()
    
    # First: standing person with aspect ratio ~ 2.4
    for f in range(10):
        rec.boxes.append((100, 100, 140, 196)) # w=40, h=96 => ar = 2.4
        rec.aspect_ratio = 2.4
        rec.speed = 15.0
        
    records = {track_id: rec}
    events = detector.check("CAM-1", records, frame_idx=10)
    assert len(events) == 0 # No fall yet
    
    # Fall occurs: aspect ratio drops to 0.5 (lying horizontal, w=80, h=30)
    # and stays stationary on ground
    for f in range(11, 45): # Over 25 frames (1 sec)
        rec.boxes.append((100, 160, 180, 190)) # w=80, h=30 => ar = 0.375
        rec.aspect_ratio = 0.375
        rec.speed = 2.0 # Immobile on floor
        events = detector.check("CAM-1", records, frame_idx=f)
        if f >= 36: # Exceeded 1 second standstill
            assert len(events) >= 1
            assert events[0].anomaly_type == "passenger_fall"
            assert events[0].severity == "CRITICAL"
            break

def test_panic_motion_detector():
    detector = PanicMotionDetector(min_crowd_size=4, entropy_threshold=1.2)
    
    motion_metrics = MotionMetrics()
    motion_metrics.active_tracks_count = 6
    motion_metrics.mean_speed = 50.0
    motion_metrics.direction_entropy = 1.8 # High dispersal
    
    events = detector.check("CAM-1", motion_metrics)
    assert len(events) == 1
    assert events[0].anomaly_type == "panic_motion"
    assert events[0].severity == "CRITICAL"

def test_restricted_zone_intrusion():
    detector = IntrusionDetector(restricted_keys=["track_danger_zone"])
    
    metrics = {
        "platform_zone": ROIMetrics("platform_zone", "Platform", 20.0, "monitored"),
        "track_danger_zone": ROIMetrics("track_danger_zone", "Track Bed", 5.0, "restricted")
    }
    metrics["track_danger_zone"].person_count = 1
    metrics["track_danger_zone"].tracked_ids = [99]
    
    events = detector.check("CAM-1", metrics)
    assert len(events) == 1
    assert events[0].anomaly_type == "zone_intrusion"
    assert events[0].track_id == 99
    assert events[0].severity == "CRITICAL"

def test_alert_manager_cooldown_and_logging(tmp_path):
    log_file = str(tmp_path / "test_alerts.jsonl")
    cfg = AlertingConfig(cooldown_seconds=3, log_file=log_file, webhook_enabled=False)
    manager = AlertManager(cfg)
    
    from anomaly.detectors import AnomalyEvent
    event = AnomalyEvent(
        timestamp=time.time(),
        camera_id="CAM-1",
        anomaly_type="overcrowding",
        severity="HIGH",
        zone_key="zone_a",
        track_id=None,
        message="Test alert",
        metrics={"density": 3.8}
    )
    
    # First dispatch should succeed
    dispatched = manager.dispatch([event])
    assert len(dispatched) == 1
    assert os.path.exists(log_file)
    
    # Immediate second dispatch of the same event should be suppressed by cooldown
    dispatched_suppressed = manager.dispatch([event])
    assert len(dispatched_suppressed) == 0
