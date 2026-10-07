import math
from typing import Dict, List, Any
import numpy as np
from tracking.tracker import TrackHistoryItem

class MotionMetrics:
    def __init__(self):
        self.active_tracks_count: int = 0
        self.mean_speed: float = 0.0 # px/s
        self.mean_vx: float = 0.0
        self.mean_vy: float = 0.0
        self.dominant_direction_deg: float = 0.0
        self.speed_variance: float = 0.0
        self.direction_entropy: float = 0.0 # higher value = chaotic/omnidirectional
        self.individual_vectors: List[Dict[str, Any]] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_tracks_count": self.active_tracks_count,
            "mean_speed": round(self.mean_speed, 2),
            "mean_vx": round(self.mean_vx, 2),
            "mean_vy": round(self.mean_vy, 2),
            "dominant_direction_deg": round(self.dominant_direction_deg, 1),
            "speed_variance": round(self.speed_variance, 2),
            "direction_entropy": round(self.direction_entropy, 3),
            "individual_vectors": self.individual_vectors
        }

class MotionAnalyzer:
    """
    Analyzes track deltas and motion fields to determine mean flow velocity,
    direction vectors, and turbulence/entropy.
    """
    def __init__(self, num_direction_bins: int = 8):
        self.num_direction_bins = num_direction_bins

    def analyze(self, track_records: Dict[int, TrackHistoryItem]) -> MotionMetrics:
        metrics = MotionMetrics()
        if not track_records:
            return metrics
            
        speeds = []
        vxs = []
        vys = []
        angles = []
        vectors = []
        
        for tid, record in track_records.items():
            if len(record.centroids) < 2:
                continue
                
            speeds.append(record.speed)
            vxs.append(record.vx)
            vys.append(record.vy)
            angles.append(record.direction)
            
            latest_c = record.centroids[-1]
            vectors.append({
                "track_id": tid,
                "x": round(latest_c[0], 1),
                "y": round(latest_c[1], 1),
                "vx": round(record.vx, 1),
                "vy": round(record.vy, 1),
                "speed": round(record.speed, 1),
                "is_stationary": record.is_stationary
            })
            
        if not speeds:
            return metrics
            
        metrics.active_tracks_count = len(speeds)
        metrics.mean_speed = float(np.mean(speeds))
        metrics.mean_vx = float(np.mean(vxs))
        metrics.mean_vy = float(np.mean(vys))
        metrics.speed_variance = float(np.var(speeds))
        metrics.individual_vectors = vectors
        
        # Net direction
        net_angle = math.atan2(metrics.mean_vy, metrics.mean_vx)
        metrics.dominant_direction_deg = math.degrees(net_angle)
        
        # Compute Shannon entropy across direction bins to detect chaotic/panic motion
        # Bins from -pi to pi
        bin_counts, _ = np.histogram(angles, bins=self.num_direction_bins, range=(-math.pi, math.pi))
        total_moving = np.sum(bin_counts)
        if total_moving > 0:
            probs = bin_counts / total_moving
            # Filter non-zero for log
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log2(probs))
            metrics.direction_entropy = float(entropy)
            
        return metrics
