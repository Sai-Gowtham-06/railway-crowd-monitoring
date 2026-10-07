import cv2
import numpy as np

class HeatmapGenerator:
    """
    Generates dynamic rolling crowd density heatmaps using 2D Gaussian accumulation
    with temporal decay. Supports overlay on video frames and standalone bird's-eye views.
    """
    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        decay: float = 0.93,
        kernel_size: int = 45,
        colormap: int = cv2.COLORMAP_JET
    ):
        self.width = width
        self.height = height
        self.decay = decay
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.colormap = colormap
        
        # Internal floating-point accumulation buffer (downscaled 2x for high performance)
        self.scale = 0.5
        self.grid_w = int(width * self.scale)
        self.grid_h = int(height * self.scale)
        self.density_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        
        # Pre-compute 2D Gaussian kernel
        self.gaussian_kernel = cv2.getGaussianKernel(self.kernel_size, sigma=self.kernel_size / 3.0)
        self.gaussian_kernel = self.gaussian_kernel @ self.gaussian_kernel.T
        self.gaussian_kernel = (self.gaussian_kernel / self.gaussian_kernel.max()).astype(np.float32)

    def update(self, feet_positions: list) -> None:
        """
        Add passenger positions and decay existing heat.
        """
        # Apply temporal decay
        self.density_grid *= self.decay
        
        kh, kw = self.gaussian_kernel.shape
        half_kh = kh // 2
        half_kw = kw // 2
        
        for (x, y) in feet_positions:
            gx = int(x * self.scale)
            gy = int(y * self.scale)
            
            if 0 <= gx < self.grid_w and 0 <= gy < self.grid_h:
                # Kernel clipping coordinates
                x1 = max(0, gx - half_kw)
                y1 = max(0, gy - half_kh)
                x2 = min(self.grid_w, gx + half_kw + 1)
                y2 = min(self.grid_h, gy + half_kh + 1)
                
                kx1 = max(0, half_kw - gx)
                ky1 = max(0, half_kh - gy)
                kx2 = kx1 + (x2 - x1)
                ky2 = ky1 + (y2 - y1)
                
                if (x2 > x1) and (y2 > y1) and (kx2 > kx1) and (ky2 > ky1):
                    self.density_grid[y1:y2, x1:x2] += self.gaussian_kernel[ky1:ky2, kx1:kx2] * 2.0

    def get_colored_heatmap(self) -> np.ndarray:
        """
        Returns an RGB colormapped heatmap resized to original resolution.
        """
        # Normalize to 0-255 range
        max_val = np.max(self.density_grid)
        if max_val > 0.05:
            norm_grid = np.clip(self.density_grid / max_val * 255.0, 0, 255).astype(np.uint8)
        else:
            norm_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.uint8)
            
        colored = cv2.applyColorMap(norm_grid, self.colormap)
        # Resize to full frame
        return cv2.resize(colored, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

    def overlay_on_frame(self, frame: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        """
        Blends heatmap onto CCTV frame where heat is present.
        """
        colored_heatmap = self.get_colored_heatmap()
        
        # Mask where heat is non-negligible
        max_val = np.max(self.density_grid)
        if max_val <= 0.05:
            return frame
            
        norm_grid = np.clip(self.density_grid / max_val * 255.0, 0, 255).astype(np.uint8)
        mask = cv2.resize(norm_grid, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        
        # Only blend where density > threshold
        heat_mask = (mask > 20).astype(np.uint8)
        
        blended = frame.copy()
        overlay = cv2.addWeighted(frame, 1.0 - alpha, colored_heatmap, alpha, 0)
        
        # Apply overlay selectively to heated areas
        blended[heat_mask > 0] = overlay[heat_mask > 0]
        return blended
