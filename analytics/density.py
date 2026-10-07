from typing import Dict, List, Tuple, Any
import numpy as np
import cv2
from shapely.geometry import Point, Polygon
from config.loader import ROIConfig, CongestionTier

class ROIMetrics:
    def __init__(self, key: str, name: str, area_m2: float, zone_type: str):
        self.key = key
        self.name = name
        self.area_m2 = area_m2
        self.zone_type = zone_type  # "monitored" or "restricted"
        self.person_count: int = 0
        self.density_people_per_m2: float = 0.0
        self.congestion_tier: str = "Low"
        self.tier_action: str = "Monitor"
        self.tier_color_bgr: Tuple[int, int, int] = (46, 204, 113) # Green
        self.tracked_ids: List[int] = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "area_m2": round(self.area_m2, 2),
            "zone_type": self.zone_type,
            "person_count": self.person_count,
            "density": round(self.density_people_per_m2, 2),
            "tier": self.congestion_tier,
            "action": self.tier_action,
            "color_bgr": list(self.tier_color_bgr),
            "tracked_ids": self.tracked_ids
        }

class DensityAnalyzer:
    """
    Computes per-ROI passenger counts, densities (people/m²), and congestion tiers.
    Uses bottom-center coordinates (ground-plane contact point) for polygon spatial assignment.
    """
    def __init__(
        self,
        rois: Dict[str, ROIConfig],
        congestion_tiers: Dict[str, CongestionTier]
    ):
        self.rois = rois
        self.congestion_tiers = congestion_tiers
        
        # Prepare Shapely polygons and OpenCV contours
        self._polygons: Dict[str, Polygon] = {}
        self._contours: Dict[str, np.ndarray] = {}
        
        for key, roi in rois.items():
            pts = [(p[0], p[1]) for p in roi.polygon]
            self._polygons[key] = Polygon(pts)
            self._contours[key] = np.array(roi.polygon, dtype=np.int32).reshape((-1, 1, 2))

    def classify_tier(self, density: float) -> Tuple[str, str, Tuple[int, int, int]]:
        """
        Classify density into Low / Medium / High / Critical per specifications:
        <1.5 = Low | 1.5-3.0 = Medium | 3.0-4.5 = High | >4.5 = Critical
        """
        if density < 1.5:
            tier = self.congestion_tiers.get("low")
            return "Low", tier.action if tier else "Monitor", tuple(tier.color_bgr) if tier else (46, 204, 113)
        elif density <= 3.0:
            tier = self.congestion_tiers.get("medium")
            return "Medium", tier.action if tier else "Log & Observe", tuple(tier.color_bgr) if tier else (241, 196, 15)
        elif density <= 4.5:
            tier = self.congestion_tiers.get("high")
            return "High", tier.action if tier else "Alert Security", tuple(tier.color_bgr) if tier else (230, 126, 34)
        else:
            tier = self.congestion_tiers.get("critical")
            return "Critical", tier.action if tier else "Immediate Action", tuple(tier.color_bgr) if tier else (231, 76, 60)

    def analyze(
        self,
        tracked_xyxy: np.ndarray,
        track_ids: Optional[np.ndarray] = None
    ) -> Dict[str, ROIMetrics]:
        """
        Assign each detected person to an ROI and compute metrics.
        """
        metrics: Dict[str, ROIMetrics] = {}
        for key, roi in self.rois.items():
            metrics[key] = ROIMetrics(
                key=key,
                name=roi.name,
                area_m2=roi.physical_area_m2,
                zone_type=roi.type
            )
            
        if tracked_xyxy is None or len(tracked_xyxy) == 0:
            for key, m in metrics.items():
                m.tier, m.action, m.tier_color_bgr = self.classify_tier(0.0)
            return metrics
            
        num_persons = len(tracked_xyxy)
        for i in range(num_persons):
            box = tracked_xyxy[i]
            x1, y1, x2, y2 = box
            # Ground-plane reference: bottom-center of bounding box
            foot_x = float((x1 + x2) / 2.0)
            foot_y = float(y2)
            pt = Point(foot_x, foot_y)
            tid = int(track_ids[i]) if track_ids is not None and track_ids[i] is not None else i
            
            for key, poly in self._polygons.items():
                # Point-in-polygon test
                if poly.contains(pt):
                    metrics[key].person_count += 1
                    metrics[key].tracked_ids.append(tid)
                    
        # Compute densities and tiers
        for key, m in metrics.items():
            if m.area_m2 > 0:
                m.density_people_per_m2 = m.person_count / m.area_m2
            else:
                m.density_people_per_m2 = float(m.person_count)
            m.congestion_tier, m.tier_action, m.tier_color_bgr = self.classify_tier(m.density_people_per_m2)
            
        return metrics

    def get_roi_polygons(self) -> Dict[str, List[List[int]]]:
        return {key: roi.polygon for key, roi in self.rois.items()}
