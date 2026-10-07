import pytest
import numpy as np
from config.loader import ROIConfig, CongestionTier
from analytics.density import DensityAnalyzer
from analytics.motion import MotionAnalyzer
from tracking.tracker import TrackHistoryItem

@pytest.fixture
def sample_density_analyzer():
    rois = {
        "zone_a": ROIConfig(
            name="Zone A",
            polygon=[[0, 0], [100, 0], [100, 100], [0, 100]],
            physical_area_m2=10.0,
            type="monitored"
        ),
        "zone_b": ROIConfig(
            name="Zone B (Restricted)",
            polygon=[[200, 0], [300, 0], [300, 100], [200, 100]],
            physical_area_m2=5.0,
            type="restricted"
        )
    }
    tiers = {
        "low": CongestionTier(max_density=1.5, label="Low", action="Monitor", color_bgr=[0, 255, 0]),
        "medium": CongestionTier(max_density=3.0, label="Medium", action="Log", color_bgr=[0, 255, 255]),
        "high": CongestionTier(max_density=4.5, label="High", action="Alert", color_bgr=[0, 165, 255]),
        "critical": CongestionTier(min_density=4.5, label="Critical", action="Evacuate", color_bgr=[0, 0, 255])
    }
    return DensityAnalyzer(rois=rois, congestion_tiers=tiers)

def test_tier_classification_thresholds(sample_density_analyzer):
    analyzer = sample_density_analyzer
    
    # Boundary tests for <1.5, 1.5-3.0, 3.0-4.5, >4.5
    tier, action, _ = analyzer.classify_tier(0.0)
    assert tier == "Low"
    assert action == "Monitor"
    
    tier, action, _ = analyzer.classify_tier(1.49)
    assert tier == "Low"
    
    tier, action, _ = analyzer.classify_tier(1.5)
    assert tier == "Medium"
    
    tier, action, _ = analyzer.classify_tier(3.0)
    assert tier == "Medium"
    
    tier, action, _ = analyzer.classify_tier(3.01)
    assert tier == "High"
    
    tier, action, _ = analyzer.classify_tier(4.5)
    assert tier == "High"
    
    tier, action, _ = analyzer.classify_tier(4.51)
    assert tier == "Critical"
    assert action == "Evacuate"

def test_roi_spatial_containment(sample_density_analyzer):
    analyzer = sample_density_analyzer
    
    # Person 1 inside Zone A: box bottom-center is at (50, 80)
    # Person 2 inside Zone A: box bottom-center is at (80, 90)
    # Person 3 outside any zone: box bottom-center is at (150, 50)
    # Person 4 inside Zone B: box bottom-center is at (250, 60)
    boxes = np.array([
        [40, 20, 60, 80],
        [70, 30, 90, 90],
        [140, 10, 160, 50],
        [240, 10, 260, 60]
    ], dtype=np.float32)
    tids = np.array([101, 102, 103, 104])
    
    metrics = analyzer.analyze(boxes, tids)
    
    assert metrics["zone_a"].person_count == 2
    assert metrics["zone_a"].density_people_per_m2 == 2 / 10.0 # 0.2
    assert metrics["zone_a"].congestion_tier == "Low"
    assert 101 in metrics["zone_a"].tracked_ids
    assert 102 in metrics["zone_a"].tracked_ids
    
    assert metrics["zone_b"].person_count == 1
    assert 104 in metrics["zone_b"].tracked_ids

def test_motion_analyzer_calculations():
    motion_analyzer = MotionAnalyzer()
    
    records = {}
    # Create 3 track records moving eastward (+x)
    for i in range(3):
        rec = TrackHistoryItem()
        rec.centroids.append((10.0, 50.0, 1))
        rec.centroids.append((20.0, 50.0, 2))
        rec.vx = 25.0
        rec.vy = 0.0
        rec.speed = 25.0
        rec.direction = 0.0
        records[i] = rec
        
    metrics = motion_analyzer.analyze(records)
    assert metrics.active_tracks_count == 3
    assert pytest.approx(metrics.mean_speed, 0.1) == 25.0
    assert pytest.approx(metrics.mean_vx, 0.1) == 25.0
    assert pytest.approx(metrics.mean_vy, 0.1) == 0.0
    assert pytest.approx(metrics.dominant_direction_deg, 0.1) == 0.0
    assert metrics.direction_entropy == 0.0 # All moving identically
