import os
import cv2
import time
import threading
import numpy as np
from typing import Optional, Dict, Any, List, Union
from config.loader import AppConfig, load_config
from detection.detector import CrowdDetector
from tracking.tracker import CrowdTracker
from privacy.anonymizer import PrivacyAnonymizer
from analytics.density import DensityAnalyzer
from analytics.motion import MotionAnalyzer
from analytics.heatmap import HeatmapGenerator
from anomaly.engine import AnomalyEngine
from alerting.alert_manager import AlertManager

class VisionPipeline:
    """
    Continuous video ingestion, inference, and telemetry pipeline.
    Supports real-time webcam (0), RTSP IP streams, and local video feeds
    with live source switching.
    """
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config: AppConfig = load_config(config_path)
        self.current_source: Union[int, str] = self._parse_source(self.config.camera.source)
        
        # Modules
        self.detector = CrowdDetector(
            model_path=self.config.detector.model_path,
            confidence=self.config.detector.confidence,
            iou=self.config.detector.iou,
            device=self.config.detector.device,
            classes=self.config.detector.classes,
            enable_synthetic_fallback=True
        )
        self.tracker = CrowdTracker(
            track_activation_threshold=self.config.tracker.track_activation_threshold,
            lost_track_buffer=self.config.tracker.lost_track_buffer,
            minimum_matching_threshold=self.config.tracker.minimum_matching_threshold,
            frame_rate=self.config.tracker.frame_rate
        )
        self.anonymizer = PrivacyAnonymizer(
            enabled=self.config.privacy.anonymize_faces,
            blur_kernel_size=self.config.privacy.blur_kernel_size,
            face_ratio_top=self.config.privacy.face_ratio_top
        )
        self.density_analyzer = DensityAnalyzer(
            rois=self.config.rois,
            congestion_tiers=self.config.congestion_tiers
        )
        self.motion_analyzer = MotionAnalyzer()
        self.heatmap_gen = HeatmapGenerator(
            width=self.config.camera.width,
            height=self.config.camera.height
        )
        self.anomaly_engine = AnomalyEngine(
            config=self.config.anomaly,
            camera_id=self.config.camera.camera_id,
            frame_rate=self.config.camera.fps
        )
        self.alert_manager = AlertManager(self.config.alerting)
        
        # State & Threading
        self.is_running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._source_changed = threading.Event()
        
        self.current_frame_idx: int = 0
        self.current_fps: float = 0.0
        self.latest_jpeg: Optional[bytes] = None
        self.latest_telemetry: Dict[str, Any] = {}
        self.telemetry_listeners: List[Any] = []
        
        # Overlay display toggles
        self.show_heatmap: bool = True
        self.show_polygons: bool = True
        self.show_tracks: bool = True
        self.show_anonymization: bool = True

    def _parse_source(self, src: Union[str, int]) -> Union[int, str]:
        if isinstance(src, int):
            return src
        if isinstance(src, str) and src.isdigit():
            return int(src)
        return src

    def change_source(self, new_source: Union[str, int]) -> bool:
        """
        Dynamically switches camera source without restarting server.
        """
        parsed = self._parse_source(new_source)
        with self._lock:
            self.current_source = parsed
            self.config.camera.source = str(new_source)
            self._source_changed.set()
        print(f"[VisionPipeline] Switching video source to: {parsed}")
        return True

    def start(self) -> None:
        if self.is_running:
            return
        self.is_running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print(f"[VisionPipeline] Started processing stream for {self.config.camera.station_name}")

    def stop(self) -> None:
        self.is_running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        print("[VisionPipeline] Stopped.")

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        src = self.current_source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened() and src == 0:
            print("[VisionPipeline] Webcam 0 failed to open, falling back to data/station_entry_live.mp4")
            self.current_source = "data/station_entry_live.mp4"
            cap = cv2.VideoCapture(self.current_source)
        return cap

    def _run_loop(self) -> None:
        cap = self._open_capture()
        fps_timer = time.time()
        frames_in_second = 0
        
        while self.is_running:
            # Check if source switch requested
            if self._source_changed.is_set():
                self._source_changed.clear()
                if cap:
                    cap.release()
                cap = self._open_capture()
                
            if not cap or not cap.isOpened():
                time.sleep(0.5)
                cap = self._open_capture()
                continue
                
            ret, frame = cap.read()
            if not ret:
                # If recorded video reached EOF, reopen to loop seamlessly
                if isinstance(self.current_source, str) and os.path.exists(self.current_source):
                    cap.release()
                    cap = cv2.VideoCapture(self.current_source)
                    continue
                else:
                    time.sleep(0.04)
                    continue
                    
            # Ensure consistent resolution with aspect ratio preservation (letterboxing)
            h, w = frame.shape[:2]
            target_w, target_h = self.config.camera.width, self.config.camera.height
            if w != target_w or h != target_h:
                scale = min(target_w / w, target_h / h)
                new_w, new_h = int(w * scale), int(h * scale)
                resized = cv2.resize(frame, (new_w, new_h))
                canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                pad_x = (target_w - new_w) // 2
                pad_y = (target_h - new_h) // 2
                canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
                frame = canvas
                
            self.current_frame_idx += 1
            frames_in_second += 1
            if time.time() - fps_timer >= 1.0:
                self.current_fps = frames_in_second / (time.time() - fps_timer)
                frames_in_second = 0
                fps_timer = time.time()
                
            # 1. Detection
            detections = self.detector.detect(frame)
            
            # 2. Tracking
            tracked = self.tracker.update(detections)
            active_tracks = self.tracker.get_all_active_tracks()
            
            # 3. Feet points for heatmap
            feet_pts = []
            if len(tracked) > 0 and tracked.xyxy is not None:
                for box in tracked.xyxy:
                    feet_pts.append(((box[0] + box[2]) / 2.0, box[3]))
            self.heatmap_gen.update(feet_pts)
            
            # 4. Density Analytics
            roi_metrics = self.density_analyzer.analyze(
                tracked.xyxy if len(tracked) > 0 else None,
                tracked.tracker_id if len(tracked) > 0 else None
            )
            
            # 5. Motion Analytics
            motion_metrics = self.motion_analyzer.analyze(active_tracks)
            
            # 6. Anomaly Detection
            current_anomalies = self.anomaly_engine.process_frame(
                frame_idx=self.current_frame_idx,
                roi_metrics=roi_metrics,
                motion_metrics=motion_metrics,
                track_records=active_tracks
            )
            
            # 7. Alert Management
            new_alerts = self.alert_manager.dispatch(current_anomalies)
            
            # 8. Privacy Anonymization
            if self.show_anonymization:
                proc_frame = self.anonymizer.anonymize_frame(frame, detections)
            else:
                proc_frame = frame.copy()
                
            # 9. Rendering & Visual Overlays
            if self.show_heatmap:
                rendered_frame = self.heatmap_gen.overlay_on_frame(proc_frame, alpha=0.35)
            else:
                rendered_frame = proc_frame
                
            # Draw ROI Polygons
            if self.show_polygons:
                for key, m in roi_metrics.items():
                    pts = np.array(self.config.rois[key].polygon, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(rendered_frame, [pts], isClosed=True, color=m.tier_color_bgr, thickness=2)
                    badge = f"{m.name}: {m.person_count}p ({m.density_people_per_m2:.2f}/m2) [{m.congestion_tier}]"
                    cv2.putText(rendered_frame, badge, (pts[0][0][0], max(25, pts[0][0][1] - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, m.tier_color_bgr, 2)
                    
            # Draw Real-Time Detected & Tracked Persons
            if self.show_tracks and len(tracked) > 0 and tracked.xyxy is not None:
                for i, box in enumerate(tracked.xyxy):
                    tid = tracked.tracker_id[i] if tracked.tracker_id is not None else None
                    conf = tracked.confidence[i] if tracked.confidence is not None else 0.85
                    x1, y1, x2, y2 = map(int, box)
                    
                    # High-visibility bounding box
                    box_color = (0, 255, 230) # Neon Cyan
                    cv2.rectangle(rendered_frame, (x1, y1), (x2, y2), box_color, 2)
                    
                    # Label badge
                    label = f"Person #{tid}" if tid is not None else "Person"
                    label += f" {int(conf * 100)}%"
                    
                    # Label background pill
                    (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    cv2.rectangle(rendered_frame, (x1, max(0, y1 - lh - 6)), (x1 + lw + 6, y1), (20, 20, 30), -1)
                    cv2.putText(rendered_frame, label, (x1 + 3, max(lh + 2, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                    
                    # Draw velocity motion trail
                    if tid is not None and tid in active_tracks:
                        track_rec = active_tracks[tid]
                        if len(track_rec.centroids) >= 2:
                            pts_trail = [(int(c[0]), int(c[1])) for c in track_rec.centroids]
                            for p_idx in range(len(pts_trail) - 1):
                                cv2.line(rendered_frame, pts_trail[p_idx], pts_trail[p_idx + 1], (0, 255, 230), 2)
                        
            # Draw Critical Anomaly Warning Banner if active
            if current_anomalies:
                top_anomaly = current_anomalies[0]
                sev_color = (0, 0, 230) if top_anomaly.severity == "CRITICAL" else (0, 140, 255)
                cv2.rectangle(rendered_frame, (0, 0), (self.config.camera.width, 50), sev_color, -1)
                cv2.putText(rendered_frame, f"WARNING [{top_anomaly.severity}]: {top_anomaly.message}",
                            (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                
            # Add Source tag to video corner
            source_tag = "LIVE WEBCAM" if str(self.current_source) in ["0", "webcam"] else "SIMULATED CCTV"
            cv2.rectangle(rendered_frame, (self.config.camera.width - 200, 15), (self.config.camera.width - 15, 45), (15, 23, 42), -1)
            cv2.putText(rendered_frame, f"SOURCE: {source_tag}", (self.config.camera.width - 190, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 230), 1)

            # Encode JPEG for video stream
            ret_encode, jpeg_buf = cv2.imencode('.jpg', rendered_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret_encode:
                with self._lock:
                    self.latest_jpeg = jpeg_buf.tobytes()
                    
            # Update telemetry dictionary
            total_station_people = sum(m.person_count for m in roi_metrics.values() if m.zone_type == "monitored")
            telemetry = {
                "timestamp": time.time(),
                "frame_idx": self.current_frame_idx,
                "fps": round(self.current_fps, 1),
                "source": str(self.current_source),
                "station_name": self.config.camera.station_name,
                "camera_id": self.config.camera.camera_id,
                "total_passengers": total_station_people if total_station_people > 0 else len(tracked),
                "zones": {k: m.to_dict() for k, m in roi_metrics.items()},
                "motion": motion_metrics.to_dict(),
                "active_anomalies": [a.to_dict() for a in current_anomalies],
                "recent_alerts": self.alert_manager.get_recent_alerts(limit=20)
            }
            
            with self._lock:
                self.latest_telemetry = telemetry
                
            # Broadcast to telemetry listeners
            for listener in list(self.telemetry_listeners):
                try:
                    listener(telemetry)
                except Exception:
                    pass
                    
            # Frame rate throttle
            time.sleep(1.0 / max(10, self.config.camera.fps))
            
        if cap:
            cap.release()

    def get_latest_frame(self) -> Optional[bytes]:
        with self._lock:
            return self.latest_jpeg

    def get_latest_telemetry(self) -> Dict[str, Any]:
        with self._lock:
            return self.latest_telemetry
