import cv2
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.loader import load_config
from detection.detector import CrowdDetector
from tracking.tracker import CrowdTracker
from privacy.anonymizer import PrivacyAnonymizer
from analytics.density import DensityAnalyzer
from analytics.motion import MotionAnalyzer
from analytics.heatmap import HeatmapGenerator

def run_milestone2_verification():
    print("[Milestone 2] Loading configuration...")
    config = load_config("config/config.yaml")
    
    detector = CrowdDetector(
        model_path=config.detector.model_path,
        confidence=0.20,
        iou=config.detector.iou,
        device=config.detector.device,
        classes=config.detector.classes
    )
    
    tracker = CrowdTracker(
        track_activation_threshold=0.20,
        lost_track_buffer=config.tracker.lost_track_buffer,
        minimum_matching_threshold=config.tracker.minimum_matching_threshold,
        frame_rate=config.tracker.frame_rate
    )
    
    anonymizer = PrivacyAnonymizer(
        enabled=config.privacy.anonymize_faces,
        blur_kernel_size=config.privacy.blur_kernel_size,
        face_ratio_top=config.privacy.face_ratio_top
    )
    
    density_analyzer = DensityAnalyzer(
        rois=config.rois,
        congestion_tiers=config.congestion_tiers
    )
    
    motion_analyzer = MotionAnalyzer()
    
    heatmap_gen = HeatmapGenerator(
        width=config.camera.width,
        height=config.camera.height
    )
    
    cap = cv2.VideoCapture(config.camera.source)
    if not cap.isOpened():
        print(f"Error: Could not open {config.camera.source}")
        return False
        
    frame_idx = 0
    t0 = time.time()
    
    while cap.isOpened() and frame_idx < 60:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        
        # 1. Detect & Track
        detections = detector.detect(frame)
        tracked = tracker.update(detections)
        
        # 2. Extract feet positions
        feet_pts = []
        if len(tracked) > 0 and tracked.xyxy is not None:
            for box in tracked.xyxy:
                feet_pts.append(((box[0] + box[2]) / 2.0, box[3]))
                
        # 3. Update Heatmap
        heatmap_gen.update(feet_pts)
        
        # 4. Density Analytics
        roi_metrics = density_analyzer.analyze(
            tracked.xyxy if len(tracked) > 0 else None,
            tracked.tracker_id if len(tracked) > 0 else None
        )
        
        # 5. Motion Analytics
        active_tracks = tracker.get_all_active_tracks()
        motion_metrics = motion_analyzer.analyze(active_tracks)
        
        # 6. Anonymize
        anonymized = anonymizer.anonymize_frame(frame, detections)
        
        # 7. Render visualization
        vis_frame = heatmap_gen.overlay_on_frame(anonymized, alpha=0.35)
        
        # Draw ROI Polygons with congestion tier colors
        for key, m in roi_metrics.items():
            pts = [(int(p[0]), int(p[1])) for p in config.rois[key].polygon]
            pts_np = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            
            # Draw polygon boundary
            cv2.polylines(vis_frame, [pts_np], isClosed=True, color=m.tier_color_bgr, thickness=2)
            
            # ROI label badge
            badge_text = f"{m.name}: {m.person_count}p ({m.density_people_per_m2:.2f} p/m2) [{m.congestion_tier}]"
            cv2.putText(vis_frame, badge_text, (pts[0][0], max(30, pts[0][1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, m.tier_color_bgr, 2)
            
        # Draw HUD Box in corner
        cv2.rectangle(vis_frame, (20, 50), (380, 160), (20, 20, 25), -1)
        cv2.rectangle(vis_frame, (20, 50), (380, 160), (80, 80, 90), 1)
        cv2.putText(vis_frame, "REAL-TIME STATION ANALYTICS", (35, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(vis_frame, f"Active Tracked Passengers: {motion_metrics.active_tracks_count}", (35, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(vis_frame, f"Mean Speed: {motion_metrics.mean_speed:.1f} px/s", (35, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(vis_frame, f"Motion Direction Entropy: {motion_metrics.direction_entropy:.3f}", (35, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        if frame_idx == 45:
            snapshot_path = "artifacts/milestone2_snapshot.jpg"
            cv2.imwrite(snapshot_path, vis_frame)
            print(f"[Milestone 2] Saved verification snapshot to {snapshot_path}")
            
    cap.release()
    elapsed = time.time() - t0
    print(f"[Milestone 2 Verification Complete] Processed {frame_idx} frames at {frame_idx/elapsed:.2f} FPS")
    return True

if __name__ == "__main__":
    import numpy as np
    run_milestone2_verification()
