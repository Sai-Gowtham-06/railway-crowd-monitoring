import cv2
import os
import sys
import time
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.loader import load_config
from detection.detector import CrowdDetector
from tracking.tracker import CrowdTracker
from privacy.anonymizer import PrivacyAnonymizer
from analytics.density import DensityAnalyzer
from analytics.motion import MotionAnalyzer
from analytics.heatmap import HeatmapGenerator
from anomaly.engine import AnomalyEngine
from alerting.alert_manager import AlertManager

def run_milestone3_verification():
    print("[Milestone 3] Loading configuration...")
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
    
    anomaly_engine = AnomalyEngine(
        config=config.anomaly,
        camera_id=config.camera.camera_id,
        frame_rate=config.camera.fps
    )
    
    alert_manager = AlertManager(config.alerting)
    
    cap = cv2.VideoCapture(config.camera.source)
    if not cap.isOpened():
        print(f"Error: Could not open {config.camera.source}")
        return False
        
    frame_idx = 0
    total_anomalies_detected = 0
    dispatched_alerts_count = 0
    
    snapshot_saved = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        
        # 1. Detection & Tracking
        detections = detector.detect(frame)
        tracked = tracker.update(detections)
        active_tracks = tracker.get_all_active_tracks()
        
        # 2. Analytics
        roi_metrics = density_analyzer.analyze(
            tracked.xyxy if len(tracked) > 0 else None,
            tracked.tracker_id if len(tracked) > 0 else None
        )
        motion_metrics = motion_analyzer.analyze(active_tracks)
        
        # 3. Anomaly Evaluation
        current_anomalies = anomaly_engine.process_frame(
            frame_idx=frame_idx,
            roi_metrics=roi_metrics,
            motion_metrics=motion_metrics,
            track_records=active_tracks
        )
        total_anomalies_detected += len(current_anomalies)
        
        # 4. Alert Dispatching
        dispatched = alert_manager.dispatch(current_anomalies)
        dispatched_alerts_count += len(dispatched)
        
        # 5. Visual Anomaly Rendering & Snapshot
        if current_anomalies and not snapshot_saved and frame_idx > 120:
            anonymized = anonymizer.anonymize_frame(frame, detections)
            vis = anonymized.copy()
            
            # Draw Alert Banner across the top
            alert = current_anomalies[0]
            severity_color = (0, 0, 255) if alert.severity == "CRITICAL" else (0, 140, 255)
            cv2.rectangle(vis, (0, 0), (config.camera.width, 65), severity_color, -1)
            cv2.putText(vis, f"ALERT [{alert.severity}]: {alert.message}", (25, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            
            # Save snapshot
            snapshot_path = "artifacts/milestone3_snapshot.jpg"
            cv2.imwrite(snapshot_path, vis)
            print(f"[Milestone 3] Saved anomaly alert snapshot to {snapshot_path}")
            snapshot_saved = True
            
    cap.release()
    print(f"[Milestone 3 Complete] Processed {frame_idx} frames.")
    print(f"Total anomaly frame occurrences: {total_anomalies_detected}")
    print(f"Total dispatched unique alerts: {dispatched_alerts_count}")
    print(f"Structured log entries written to: {config.alerting.log_file}")
    
    return True

if __name__ == "__main__":
    run_milestone3_verification()
