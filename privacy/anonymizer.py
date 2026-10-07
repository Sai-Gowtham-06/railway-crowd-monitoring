import cv2
import numpy as np
import supervision as sv
from typing import Optional

class PrivacyAnonymizer:
    """
    Privacy-preserving anonymizer for public surveillance CCTV feeds.
    Anonymizes detected individuals by applying Gaussian blur to face/head regions
    before display, broadcast, or storage.
    """
    def __init__(
        self,
        enabled: bool = True,
        blur_kernel_size: int = 25,
        face_ratio_top: float = 0.25
    ):
        self.enabled = enabled
        # Kernel size must be an odd positive integer
        self.ksize = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
        self.face_ratio_top = face_ratio_top

    def anonymize_frame(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """
        Applies face blurring to the frame based on current detections.
        Modifies and returns the frame with blurred face regions.
        """
        if not self.enabled or detections is None or len(detections) == 0:
            return frame
            
        h_frame, w_frame = frame.shape[:2]
        out_frame = frame.copy()
        
        for box in detections.xyxy:
            x1 = max(0, int(box[0]))
            y1 = max(0, int(box[1]))
            x2 = min(w_frame, int(box[2]))
            y2 = min(h_frame, int(box[3]))
            
            box_h = y2 - y1
            box_w = x2 - x1
            if box_h <= 5 or box_w <= 5:
                continue
                
            # Head/face is estimated as the top fraction of the person box
            # For typical surveillance camera angles, top 25-30% captures head and neck
            face_h = max(4, int(box_h * self.face_ratio_top))
            face_y2 = min(h_frame, y1 + face_h)
            
            # Extract ROI
            face_roi = out_frame[y1:face_y2, x1:x2]
            if face_roi.size > 0:
                # Dynamic blur based on face size
                dynamic_k = max(7, (min(face_h, box_w) // 2) * 2 + 1)
                blurred = cv2.GaussianBlur(face_roi, (dynamic_k, dynamic_k), 30)
                out_frame[y1:face_y2, x1:x2] = blurred
                
        return out_frame
