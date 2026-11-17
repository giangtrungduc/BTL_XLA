"""
Module core/histogram.py
Biểu đồ tần suất Histogram và Cân bằng Histogram (Chương 2, slide 58-78).
Cài đặt cả 2 phương pháp cân bằng lược đồ xám có trong bài giảng.
Toàn bộ thuật toán được cài đặt trực tiếp bằng NumPy (Code from scratch).
"""
from typing import Tuple
import numpy as np


def compute_histogram(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tính biểu đồ tần suất Histogram mức xám (Chương 2, slide 58-62).
    
    Tham số:
        image: Mảng 2D (ảnh xám) hoặc 3D.
        
    Trả về:
        h: Histogram chưa chuẩn hóa (số lượng điểm ảnh ở mỗi mức 0..255).
        p: Histogram chuẩn hóa p(r_k) = n_k / (M * N).
    """
    if image.ndim == 2:
        gray = image
    else:
        # Độ chói chuẩn ITU-R BT.601: 0.299R + 0.587G + 0.114B (Chương 3)
        gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        
    gray_clean = np.clip(np.round(gray), 0, 255).astype(np.uint8)
    h = np.bincount(gray_clean.ravel(), minlength=256)
    total_pixels = float(gray_clean.size)
    p = h / total_pixels
    return h, p


def compute_stats(image: np.ndarray) -> Tuple[float, float]:
    """Tính trung bình (mean) và độ lệch chuẩn (std) của mức xám ảnh (Chương 2, slide 61-62)."""
    if image.ndim == 2:
        gray = image.astype(np.float64)
    else:
        gray = (0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]).astype(np.float64)
    return float(np.mean(gray)), float(np.std(gray))


def equalize_histogram(image: np.ndarray, method: str = "stretch_to_zero") -> np.ndarray:
    """
    Cân bằng Lược đồ xám (Histogram Equalization) - Chương 2, slide 65-78.
    
    Tham số:
        image: Mảng ảnh mức xám (2D) hoặc mảng 3D.
        method:
            - 'textbook' (Lý thuyết giáo trình slide 70):
                s_k = round((L - 1) * sum_{j=0}^k p_r(r_j))
            - 'stretch_to_zero' (Thực tế tối ưu kéo giãn đáy về 0, slide 72-73, khớp cv2.equalizeHist):
                s_k = round( 255 * (CDF_k - CDF_min) / (Total - CDF_min) )
    """
    def _equalize_channel(ch: np.ndarray) -> np.ndarray:
        h, _ = compute_histogram(ch)
        cdf = np.cumsum(h)
        total_pixels = ch.size
        
        if method == "stretch_to_zero":
            non_zeros = cdf[cdf > 0]
            if len(non_zeros) == 0 or total_pixels == non_zeros[0]:
                return ch.copy()
            min_cdf = non_zeros[0]
            lut = np.round((cdf - min_cdf) * 255.0 / (total_pixels - min_cdf))
        else:
            # Phương pháp giáo trình chuẩn (Slide tr.70)
            lut = np.round(255.0 * cdf / float(total_pixels))
            
        lut = np.clip(lut, 0, 255).astype(np.uint8)
        return lut[ch]

    if image.ndim == 2:
        return _equalize_channel(image)
    else:
        channels = [_equalize_channel(image[..., c]) for c in range(image.shape[2])]
        return np.stack(channels, axis=-1)
