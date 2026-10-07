import cv2
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import supervision as sv
from config.loader import load_config
from detection.detector import CrowdDetector
from tracking.tracker import CrowdTracker
from privacy.anonymizer import PrivacyAnonymizer

def run_milestone1_verification():
    print("[Milestone 1] Loading configuration...")
    config = load_config("config/config.yaml")
    
    # Initialize components
    # We test with confidence threshold = 0.20 to be flexible
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
    
    cap = cv2.VideoCapture(config.camera.source)
    if not cap.isOpened():
        print(f"Error: Could not open video file {config.camera.source}")
        return False
        
    os.makedirs("artifacts", exist_ok=True)
    frame_idx = 0
    total_detections = 0
    total_tracks = 0
    
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=2)
    trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=30)
    
    start_time = time.time()
    
    while cap.isOpened() and frame_idx < 60:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        
        # 1. Detect
        detections = detector.detect(frame)
        total_detections += len(detections)
        
        # 2. Track
        tracked_detections = tracker.update(detections)
        if tracked_detections.tracker_id is not None:
            total_tracks += len(tracked_detections.tracker_id)
            
        # 3. Privacy face blur
        anonymized_frame = anonymizer.anonymize_frame(frame, detections)
        
        # 4. Annotate
        annotated_frame = trace_annotator.annotate(scene=anonymized_frame.copy(), detections=tracked_detections)
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=tracked_detections)
        
        if tracked_detections.tracker_id is not None:
            labels = [f"ID #{tid}" for tid in tracked_detections.tracker_id]
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=tracked_detections, labels=labels)
            
        # Save snapshot at frame 30
        if frame_idx == 30:
            snapshot_path = "artifacts/milestone1_snapshot.jpg"
            cv2.imwrite(snapshot_path, annotated_frame)
            print(f"[Milestone 1] Saved verification snapshot to {snapshot_path}")
            
    cap.release()
    elapsed = time.time() - start_time
    fps = frame_idx / elapsed if elapsed > 0 else 0
    
    print(f"[Milestone 1 Verification Complete]")
    print(f"Frames processed: {frame_idx}")
    print(f"Average FPS: {fps:.2f}")
    print(f"Total detections: {total_detections}")
    print(f"Total tracked instances: {total_tracks}")
    
    return True

if __name__ == "__main__":
    run_milestone1_verification()
