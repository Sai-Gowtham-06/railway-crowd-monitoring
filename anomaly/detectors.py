import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
from tracking.tracker import TrackHistoryItem
from analytics.density import ROIMetrics
from analytics.motion import MotionMetrics
from .statistical import RollingBaseline

@dataclass
class AnomalyEvent:
    timestamp: float
    camera_id: str
    anomaly_type: str  # "overcrowding" | "sudden_movement" | "panic_motion" | "passenger_fall" | "zone_intrusion"
    severity: str      # "LOW" | "MEDIUM" | "HIGH" | "CRITICAL"
    zone_key: Optional[str]
    track_id: Optional[int]
    message: str
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class OvercrowdingDetector:
    """
    Detects abnormal crowd density using both absolute congestion tier thresholds
    and statistical rolling Z-score spikes.
    """
    def __init__(self, zscore_threshold: float = 2.5, window_size: int = 60, min_samples: int = 15):
        self.zscore_threshold = zscore_threshold
        self.baselines: Dict[str, RollingBaseline] = {}
        self.window_size = window_size
        self.min_samples = min_samples

    def check(self, camera_id: str, roi_metrics: Dict[str, ROIMetrics]) -> List[AnomalyEvent]:
        events = []
        now = time.time()
        
        for key, m in roi_metrics.items():
            if m.zone_type == "restricted":
                continue # Handled by intrusion detector
                
            if key not in self.baselines:
                self.baselines[key] = RollingBaseline(self.window_size, self.min_samples)
                
            z_score = self.baselines[key].update(m.density_people_per_m2)
            
            # Condition 1: Density in Critical or High tier
            if m.congestion_tier == "Critical":
                events.append(AnomalyEvent(
                    timestamp=now,
                    camera_id=camera_id,
                    anomaly_type="overcrowding",
                    severity="CRITICAL",
                    zone_key=key,
                    track_id=None,
                    message=f"CRITICAL Overcrowding in {m.name}: {m.density_people_per_m2:.2f} p/m² (Capacity breach)",
                    metrics={"density": m.density_people_per_m2, "person_count": m.person_count, "tier": m.congestion_tier, "z_score": z_score}
                ))
            elif m.congestion_tier == "High":
                events.append(AnomalyEvent(
                    timestamp=now,
                    camera_id=camera_id,
                    anomaly_type="overcrowding",
                    severity="HIGH",
                    zone_key=key,
                    track_id=None,
                    message=f"High Crowd Density in {m.name}: {m.density_people_per_m2:.2f} p/m²",
                    metrics={"density": m.density_people_per_m2, "person_count": m.person_count, "tier": m.congestion_tier, "z_score": z_score}
                ))
            # Condition 2: Statistical surge Z-score > threshold even before absolute max
            elif z_score is not None and z_score > self.zscore_threshold and m.person_count >= 5:
                events.append(AnomalyEvent(
                    timestamp=now,
                    camera_id=camera_id,
                    anomaly_type="overcrowding",
                    severity="MEDIUM",
                    zone_key=key,
                    track_id=None,
                    message=f"Rapid crowd influx detected in {m.name} (Z-Score: +{z_score:.2f})",
                    metrics={"density": m.density_people_per_m2, "person_count": m.person_count, "z_score": z_score}
                ))
                
        return events

class SuddenMovementDetector:
    """
    Detects sudden spikes in overall crowd speed (surge / stampede precursor)
    relative to moving average baseline.
    """
    def __init__(self, speed_zscore_threshold: float = 2.8, window_size: int = 60, min_samples: int = 15):
        self.threshold = speed_zscore_threshold
        self.baseline = RollingBaseline(window_size, min_samples)

    def check(self, camera_id: str, motion_metrics: MotionMetrics) -> List[AnomalyEvent]:
        events = []
        if motion_metrics.active_tracks_count < 3:
            return events
            
        z_score = self.baseline.update(motion_metrics.mean_speed)
        if z_score is not None and z_score > self.threshold and motion_metrics.mean_speed > 25.0:
            events.append(AnomalyEvent(
                timestamp=time.time(),
                camera_id=camera_id,
                anomaly_type="sudden_movement",
                severity="HIGH",
                zone_key=None,
                track_id=None,
                message=f"Sudden crowd speed surge: {motion_metrics.mean_speed:.1f} px/s (Z-score: +{z_score:.2f})",
                metrics={
                    "mean_speed": motion_metrics.mean_speed,
                    "baseline_mean": self.baseline.mean,
                    "speed_z_score": z_score,
                    "active_tracks": motion_metrics.active_tracks_count
                }
            ))
        return events

class PanicMotionDetector:
    """
    Detects chaotic / panic scattering by analyzing multi-directional entropy
    combined with elevated speed variance.
    """
    def __init__(self, min_crowd_size: int = 4, entropy_threshold: float = 1.4, variance_multiplier: float = 2.2):
        self.min_crowd_size = min_crowd_size
        self.entropy_threshold = entropy_threshold
        self.variance_multiplier = variance_multiplier

    def check(self, camera_id: str, motion_metrics: MotionMetrics) -> List[AnomalyEvent]:
        events = []
        if motion_metrics.active_tracks_count < self.min_crowd_size:
            return events
            
        # Panic signature: High directional entropy (scattering) + High speed variance
        if motion_metrics.direction_entropy > self.entropy_threshold and motion_metrics.mean_speed > 35.0:
            events.append(AnomalyEvent(
                timestamp=time.time(),
                camera_id=camera_id,
                anomaly_type="panic_motion",
                severity="CRITICAL",
                zone_key=None,
                track_id=None,
                message=f"Chaotic panic motion / dispersal pattern detected (Direction entropy: {motion_metrics.direction_entropy:.2f})",
                metrics={
                    "direction_entropy": motion_metrics.direction_entropy,
                    "mean_speed": motion_metrics.mean_speed,
                    "speed_variance": motion_metrics.speed_variance,
                    "active_tracks": motion_metrics.active_tracks_count
                }
            ))
        return events

class PassengerFallDetector:
    """
    Detects passenger falls based on:
    1. Transition from standing (aspect ratio > 1.3) to horizontal (aspect ratio <= 0.85).
    2. Subsequent motion cessation / standstill persistence (immobile on ground).
    """
    def __init__(
        self,
        aspect_ratio_threshold: float = 0.85,
        drop_speed_threshold: float = 30.0,
        standstill_seconds: float = 1.2,
        frame_rate: int = 25
    ):
        self.aspect_ratio_threshold = aspect_ratio_threshold
        self.drop_speed_threshold = drop_speed_threshold
        self.min_standstill_frames = int(standstill_seconds * frame_rate)
        
        # Track fallen candidates: track_id -> {'start_frame': int, 'alerted': bool, 'standing_ar': float}
        self.fall_state: Dict[int, Dict[str, Any]] = {}

    def check(self, camera_id: str, track_records: Dict[int, TrackHistoryItem], frame_idx: int) -> List[AnomalyEvent]:
        events = []
        now = time.time()
        
        for tid, record in track_records.items():
            if len(record.boxes) < 2:
                continue
                
            ar = record.aspect_ratio
            speed = record.speed
            
            # If currently tracked as in a potential fall state
            if tid in self.fall_state:
                state = self.fall_state[tid]
                # If recovered and standing again (ar > 1.25) or moving fast, reset state
                if ar > 1.25 or speed > 25.0:
                    del self.fall_state[tid]
                    continue
                    
                duration_frames = frame_idx - state["start_frame"]
                if duration_frames >= self.min_standstill_frames and not state["alerted"]:
                    state["alerted"] = True
                    events.append(AnomalyEvent(
                        timestamp=now,
                        camera_id=camera_id,
                        anomaly_type="passenger_fall",
                        severity="CRITICAL",
                        zone_key=None,
                        track_id=tid,
                        message=f"Passenger Fall Detected (Track #{tid}): Person stationary on ground for {duration_frames/self.min_standstill_frames * (self.min_standstill_frames/25.0):.1f}s",
                        metrics={
                            "track_id": tid,
                            "aspect_ratio": round(ar, 2),
                            "prior_standing_aspect_ratio": round(state["standing_ar"], 2),
                            "duration_frames": duration_frames,
                            "current_speed": round(speed, 1)
                        }
                    ))
            else:
                # Check for initial fall transition:
                # Was standing in past history (any box with aspect ratio >= 1.3)
                # and is now horizontal (ar <= aspect_ratio_threshold)
                if ar <= self.aspect_ratio_threshold:
                    was_standing = False
                    prior_ar = 1.8
                    for b in record.boxes:
                        past_h = b[3] - b[1]
                        past_w = max(1.0, b[2] - b[0])
                        past_ar_val = past_h / past_w
                        if past_ar_val >= 1.3:
                            was_standing = True
                            prior_ar = past_ar_val
                            break
                            
                    if was_standing:
                        self.fall_state[tid] = {
                            "start_frame": frame_idx,
                            "alerted": False,
                            "standing_ar": prior_ar
                        }
                    
        return events

class IntrusionDetector:
    """
    Detects unauthorized or dangerous entry into designated restricted zones
    (such as railway track beds or hazardous edge zones).
    """
    def __init__(self, restricted_keys: List[str]):
        self.restricted_keys = set(restricted_keys)

    def check(self, camera_id: str, roi_metrics: Dict[str, ROIMetrics]) -> List[AnomalyEvent]:
        events = []
        now = time.time()
        
        for key, m in roi_metrics.items():
            if key in self.restricted_keys or m.zone_type == "restricted":
                if m.person_count > 0:
                    for tid in m.tracked_ids:
                        events.append(AnomalyEvent(
                            timestamp=now,
                            camera_id=camera_id,
                            anomaly_type="zone_intrusion",
                            severity="CRITICAL",
                            zone_key=key,
                            track_id=tid,
                            message=f"RESTRICTED ZONE INTRUSION: Track #{tid} inside '{m.name}'",
                            metrics={
                                "zone_key": key,
                                "zone_name": m.name,
                                "track_id": tid,
                                "total_intruders": m.person_count
                            }
                        ))
        return events
