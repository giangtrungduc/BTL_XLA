"""
Module core/intensity.py
Các phép biến đổi độ sáng và tương phản điểm ảnh (Chương 2, slide 43-56).
Sử dụng bảng tra cứu (Look-Up Table - LUT) để tối ưu hóa tốc độ tính toán.
Toàn bộ thuật toán được cài đặt trực tiếp bằng NumPy (Code from scratch).
"""
import numpy as np


def gamma_transform(image: np.ndarray, gamma: float = 1.0, c: float = 1.0) -> np.ndarray:
    """
    Biến đổi hàm lũy thừa / Gamma (Power-law Transformation) - Chương 2, slide 51-54.
    Công thức: s = c * r^gamma (với r được chuẩn hóa [0, 1]).
    - gamma < 1: Tăng độ sáng vùng tối (làm rõ chi tiết trong bóng râm).
    - gamma > 1: Giảm độ sáng vùng chói (làm đậm các vùng bạc màu).
    """
    r = np.arange(256, dtype=np.float32) / 255.0
    lut = np.clip(np.round(c * np.power(r, gamma) * 255.0), 0, 255).astype(np.uint8)
    return lut[image]


def log_transform(image: np.ndarray, c: float = None) -> np.ndarray:
    """
    Biến đổi Logarit (Log Transformation) - Chương 2, slide 46-47.
    Công thức: s = c * log(1 + r).
    Kéo giãn dải mức xám thấp (tối) và nén dải mức xám cao (sáng).
    """
    if c is None:
        c = 255.0 / np.log(1.0 + 255.0)
    r = np.arange(256, dtype=np.float32)
    lut = np.clip(np.round(c * np.log(1.0 + r)), 0, 255).astype(np.uint8)
    return lut[image]


def negative_transform(image: np.ndarray) -> np.ndarray:
    """
    Biến đổi âm bản (Negative Transformation) - Chương 2, slide 45.
    Công thức: s = (L - 1) - r = 255 - r.
    Đảo ngược sáng tối (rất hữu ích khi phục hồi phim âm bản).
    """
    return (255 - image).astype(np.uint8)


def piecewise_linear_transform(
    image: np.ndarray, 
    r1: int = 50, 
    s1: int = 20, 
    r2: int = 200, 
    s2: int = 230
) -> np.ndarray:
    """
    Biến đổi tuyến tính từng đoạn / Kéo giãn tương phản (Contrast Stretching) - Chương 2, slide 47-50.
    Ánh xạ khoảng [r1, r2] thành khoảng [s1, s2] để tăng tương phản cho vùng quan tâm.
    """
    r1 = max(0, min(r1, 254))
    r2 = max(r1 + 1, min(r2, 255))
    s1 = max(0, min(s1, 255))
    s2 = max(0, min(s2, 255))
    
    r = np.arange(256, dtype=np.float32)
    lut = np.zeros(256, dtype=np.float32)
    
    # Đoạn 1: [0, r1]
    if r1 > 0:
        lut[:r1 + 1] = (s1 / float(r1)) * r[:r1 + 1]
    else:
        lut[0] = s1
        
    # Đoạn 2: [r1, r2]
    lut[r1:r2 + 1] = s1 + ((s2 - s1) / float(r2 - r1)) * (r[r1:r2 + 1] - r1)
    
    # Đoạn 3: [r2, 255]
    if r2 < 255:
        lut[r2:] = s2 + ((255.0 - s2) / float(255 - r2)) * (r[r2:] - r2)
    else:
        lut[255] = s2
        
    lut_uint8 = np.clip(np.round(lut), 0, 255).astype(np.uint8)
    return lut_uint8[image]
