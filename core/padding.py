"""
Module core/padding.py
Xử lý biên cho ảnh trước khi áp dụng bộ lọc không gian (Chương 2, slide 91-96).
Toàn bộ thuật toán được cài đặt trực tiếp bằng NumPy (Code from scratch).
KHÔNG import cv2 trong core.
"""
import numpy as np


def pad_image(image: np.ndarray, pad_h: int, pad_w: int, mode: str = "mirror") -> np.ndarray:
    """
    Xử lý biên ảnh số (Chương 2, slide 91-96).
    
    Tham số:
        image: Mảng numpy 2D (ảnh xám) hoặc 3D (ảnh màu RGB/HSI).
        pad_h: Số lượng hàng đệm ở phía trên và phía dưới.
        pad_w: Số lượng cột đệm ở bên trái và bên phải.
        mode: Phương pháp xử lý biên:
            - 'mirror' / 'reflect': Phản chiếu gương f(-s, y) = f(s, y) (Slide tr.94)
            - 'replicate' / 'edge': Lặp lại pixel ở rìa ảnh (Slide tr.93)
            - 'zero' / 'constant': Chèn thêm số 0 vào rìa ảnh (Slide tr.92)
            
    Trả về:
        Mảng numpy đã được đệm viền.
    """
    if pad_h == 0 and pad_w == 0:
        return image.copy()
        
    is_color = (image.ndim == 3)
    if is_color:
        pad_width = ((pad_h, pad_h), (pad_w, pad_w), (0, 0))
    else:
        pad_width = ((pad_h, pad_h), (pad_w, pad_w))
        
    mode_lower = mode.lower()
    if mode_lower in ("mirror", "reflect"):
        return np.pad(image, pad_width, mode="reflect")
    elif mode_lower in ("replicate", "edge", "clamp"):
        return np.pad(image, pad_width, mode="edge")
    elif mode_lower in ("zero", "constant"):
        return np.pad(image, pad_width, mode="constant", constant_values=0)
    else:
        # Mặc định sử dụng mirror
        return np.pad(image, pad_width, mode="reflect")
