from typing import Dict, List
from collections import deque
from config.loader import AnomalyConfig
from tracking.tracker import TrackHistoryItem
from analytics.density import ROIMetrics
from analytics.motion import MotionMetrics
from .detectors import (
    AnomalyEvent,
    OvercrowdingDetector,
    SuddenMovementDetector,
    PanicMotionDetector,
    PassengerFallDetector,
    IntrusionDetector
)

class AnomalyEngine:
    """
    Central orchestration engine for all CCTV behavioral and spatial anomalies.
    Runs detectors sequentially and collects anomalies with rolling histories.
    """
    def __init__(self, config: AnomalyConfig, camera_id: str = "CAM-1", frame_rate: int = 25):
        self.camera_id = camera_id
        self.frame_rate = frame_rate
        
        self.overcrowding_detector = OvercrowdingDetector(
            zscore_threshold=config.density_zscore_threshold,
            window_size=config.zscore_window,
            min_samples=config.min_samples_for_zscore
        )
        self.sudden_movement_detector = SuddenMovementDetector(
            speed_zscore_threshold=config.speed_spike_zscore_threshold,
            window_size=config.zscore_window,
            min_samples=config.min_samples_for_zscore
        )
        self.panic_detector = PanicMotionDetector(
            min_crowd_size=config.panic.min_crowd_size,
            entropy_threshold=config.panic.direction_entropy_threshold,
            variance_multiplier=config.panic.speed_variance_multiplier
        )
        self.fall_detector = PassengerFallDetector(
            aspect_ratio_threshold=config.fall.aspect_ratio_threshold,
            drop_speed_threshold=config.fall.drop_speed_threshold,
            standstill_seconds=config.fall.standstill_seconds,
            frame_rate=frame_rate
        )
        self.intrusion_detector = IntrusionDetector(
            restricted_keys=config.intrusion.restricted_zone_keys
        )
        
        self.recent_events: deque = deque(maxlen=200)

    def process_frame(
        self,
        frame_idx: int,
        roi_metrics: Dict[str, ROIMetrics],
        motion_metrics: MotionMetrics,
        track_records: Dict[int, TrackHistoryItem]
    ) -> List[AnomalyEvent]:
        """
        Evaluate all anomaly detectors for the current video frame.
        """
        current_events: List[AnomalyEvent] = []
        
        # 1. Check overcrowding
        current_events.extend(self.overcrowding_detector.check(self.camera_id, roi_metrics))
        
        # 2. Check sudden crowd speed surges
        current_events.extend(self.sudden_movement_detector.check(self.camera_id, motion_metrics))
        
        # 3. Check chaotic panic motion
        current_events.extend(self.panic_detector.check(self.camera_id, motion_metrics))
        
        # 4. Check passenger falls
        current_events.extend(self.fall_detector.check(self.camera_id, track_records, frame_idx))
        
        # 5. Check restricted intrusions
        current_events.extend(self.intrusion_detector.check(self.camera_id, roi_metrics))
        
        for ev in current_events:
            self.recent_events.append(ev)
            
        return current_events

    def get_recent_events(self, limit: int = 50) -> List[Dict]:
        items = list(self.recent_events)
        return [e.to_dict() for e in items[-limit:]]
