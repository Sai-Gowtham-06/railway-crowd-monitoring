import math
from collections import deque
from typing import Dict, List, Tuple, Optional
import numpy as np
import supervision as sv

class TrackHistoryItem:
    def __init__(self, maxlen: int = 40):
        self.centroids: deque = deque(maxlen=maxlen) # deque of (x, y, frame_idx)
        self.boxes: deque = deque(maxlen=maxlen)     # deque of (x1, y1, x2, y2)
        self.vx: float = 0.0  # pixels / second
        self.vy: float = 0.0  # pixels / second
        self.speed: float = 0.0
        self.direction: float = 0.0  # radians (-pi to pi)
        self.aspect_ratio: float = 1.0 # height / width
        self.last_frame_idx: int = 0
        self.total_frames: int = 0
        self.is_stationary: bool = False
        self.stationary_frames: int = 0

class CrowdTracker:
    """
    ByteTrack-based multi-object tracker for crowd streams.
    Maintains persistent track histories, motion velocities, and aspect ratios.
    """
    def __init__(
        self,
        track_activation_threshold: float = 0.30,
        lost_track_buffer: int = 35,
        minimum_matching_threshold: float = 0.70,
        frame_rate: int = 25,
        max_history_len: int = 45
    ):
        self.frame_rate = frame_rate
        self.max_history_len = max_history_len
        self.lost_track_buffer = lost_track_buffer
        
        # Initialize ByteTrack from supervision
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate
        )
        
        self.track_records: Dict[int, TrackHistoryItem] = {}
        self.frame_count: int = 0

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Update tracker with new detections from CrowdDetector.
        Returns:
            sv.Detections with assigned tracker_id.
        """
        self.frame_count += 1
        tracked_detections = self.tracker.update_with_detections(detections)
        
        active_ids = set()
        
        if tracked_detections.tracker_id is not None:
            for i, track_id in enumerate(tracked_detections.tracker_id):
                if track_id is None:
                    continue
                track_id = int(track_id)
                active_ids.add(track_id)
                
                box = tracked_detections.xyxy[i]
                x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                w = max(1.0, x2 - x1)
                h = max(1.0, y2 - y1)
                aspect_ratio = h / w
                
                if track_id not in self.track_records:
                    self.track_records[track_id] = TrackHistoryItem(maxlen=self.max_history_len)
                    
                record = self.track_records[track_id]
                record.total_frames += 1
                record.last_frame_idx = self.frame_count
                record.aspect_ratio = aspect_ratio
                record.boxes.append((x1, y1, x2, y2))
                
                # Compute velocity from previous centroid
                if len(record.centroids) > 0:
                    prev_cx, prev_cy, prev_f = record.centroids[-1]
                    delta_f = max(1, self.frame_count - prev_f)
                    dt = delta_f / float(self.frame_rate)
                    
                    vx = (cx - prev_cx) / dt
                    vy = (cy - prev_cy) / dt
                    speed = math.hypot(vx, vy)
                    
                    # Smooth velocity with exponential moving average
                    alpha = 0.4
                    record.vx = alpha * vx + (1 - alpha) * record.vx
                    record.vy = alpha * vy + (1 - alpha) * record.vy
                    record.speed = alpha * speed + (1 - alpha) * record.speed
                    record.direction = math.atan2(record.vy, record.vx)
                    
                    # Check if stationary (e.g. speed < 5 pixels/sec)
                    if record.speed < 8.0:
                        record.stationary_frames += delta_f
                        if record.stationary_frames > (self.frame_rate * 0.8):
                            record.is_stationary = True
                    else:
                        record.stationary_frames = 0
                        record.is_stationary = False
                
                record.centroids.append((cx, cy, self.frame_count))
                
        # Clean up stale tracks from memory
        stale_threshold = self.frame_count - (self.lost_track_buffer * 2)
        to_delete = [
            tid for tid, rec in self.track_records.items()
            if rec.last_frame_idx < stale_threshold
        ]
        for tid in to_delete:
            del self.track_records[tid]
            
        return tracked_detections

    def get_track_history(self, track_id: int) -> Optional[TrackHistoryItem]:
        return self.track_records.get(track_id)

    def get_all_active_tracks(self) -> Dict[int, TrackHistoryItem]:
        return self.track_records
