"""
Module core/color.py
Chuyển đổi không gian màu RGB <-> HSI và xử lý ảnh màu trên kênh cường độ I (Chương 3, slide 32-33, 53).
Toàn bộ thuật toán được cài đặt trực tiếp bằng NumPy (Code from scratch).

Theo bài giảng Chương 3 tr.53:
Khi tăng cường hoặc cân bằng histogram ảnh màu, KHÔNG ĐƯỢC xử lý độc lập trên từng kênh R, G, B
vì sẽ làm biến đổi tỷ lệ màu gốc gây lệch sắc độ nghiêm trọng.
Phương pháp chuẩn xác là chuyển sang không gian HSI, chỉ xử lý trên kênh I (Intensity),
giữ nguyên H (Hue) và S (Saturation), sau đó chuyển ngược về RGB.
"""
from typing import Tuple, Callable
import numpy as np


def rgb_to_hsi(rgb_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Chuyển đổi ảnh màu từ không gian RGB sang HSI (Chương 3, slide 32-33).
    
    Tham số:
        rgb_image: Mảng numpy uint8 (H, W, 3) có giá trị trong [0, 255].
        
    Trả về:
        H: Sắc độ (Hue) trong khoảng [0, 2*pi].
        S: Độ bão hòa (Saturation) trong khoảng [0, 1].
        I: Cường độ sáng (Intensity) trong khoảng [0, 1].
    """
    img_f = rgb_image.astype(np.float64) / 255.0
    r = img_f[..., 0]
    g = img_f[..., 1]
    b = img_f[..., 2]
    
    # 1. Intensity I = (R + G + B) / 3
    intensity = (r + g + b) / 3.0
    
    # 2. Saturation S = 1 - 3 * min(R, G, B) / (R + G + B)
    sum_rgb = r + g + b
    min_rgb = np.minimum(np.minimum(r, g), b)
    saturation = np.zeros_like(intensity)
    non_zero = sum_rgb > 1e-6
    saturation[non_zero] = 1.0 - (3.0 / sum_rgb[non_zero]) * min_rgb[non_zero]
    saturation = np.clip(saturation, 0.0, 1.0)
    
    # 3. Hue H
    eps = 1e-6
    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt(np.square(r - g) + (r - b) * (g - b)) + eps
    theta = np.arccos(np.clip(num / den, -1.0, 1.0))
    
    hue = np.zeros_like(intensity)
    # H = theta nếu B <= G, ngược lại H = 2*pi - theta
    mask = (b <= g)
    hue[mask] = theta[mask]
    hue[~mask] = 2.0 * np.pi - theta[~mask]
    
    # Nếu S = 0 (ảnh xám/không bão hòa), H không xác định -> gán 0
    hue[saturation < 1e-5] = 0.0
    
    return hue, saturation, intensity


def hsi_to_rgb(hue: np.ndarray, saturation: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    """
    Chuyển đổi ngược từ không gian HSI sang RGB (Chương 3, slide 33).
    
    Tham số:
        H: [0, 2*pi]
        S: [0, 1]
        I: [0, 1]
        
    Trả về:
        Mảng numpy uint8 (H, W, 3) trong [0, 255].
    """
    h_deg = np.rad2deg(hue) % 360.0
    s = np.clip(saturation, 0.0, 1.0)
    i = np.clip(intensity, 0.0, 1.0)
    
    r = np.zeros_like(i)
    g = np.zeros_like(i)
    b = np.zeros_like(i)
    
    # Sector 1: 0 <= H < 120 (RG sector)
    m1 = (h_deg >= 0.0) & (h_deg < 120.0)
    if np.any(m1):
        h_rad = np.deg2rad(h_deg[m1])
        b[m1] = i[m1] * (1.0 - s[m1])
        r[m1] = i[m1] * (1.0 + (s[m1] * np.cos(h_rad)) / (np.cos(np.deg2rad(60.0) - h_rad) + 1e-8))
        g[m1] = 3.0 * i[m1] - (r[m1] + b[m1])
        
    # Sector 2: 120 <= H < 240 (GB sector)
    m2 = (h_deg >= 120.0) & (h_deg < 240.0)
    if np.any(m2):
        h_rad = np.deg2rad(h_deg[m2] - 120.0)
        r[m2] = i[m2] * (1.0 - s[m2])
        g[m2] = i[m2] * (1.0 + (s[m2] * np.cos(h_rad)) / (np.cos(np.deg2rad(60.0) - h_rad) + 1e-8))
        b[m2] = 3.0 * i[m2] - (r[m2] + g[m2])
        
    # Sector 3: 240 <= H < 360 (BR sector)
    m3 = (h_deg >= 240.0) & (h_deg < 360.0)
    if np.any(m3):
        h_rad = np.deg2rad(h_deg[m3] - 240.0)
        g[m3] = i[m3] * (1.0 - s[m3])
        b[m3] = i[m3] * (1.0 + (s[m3] * np.cos(h_rad)) / (np.cos(np.deg2rad(60.0) - h_rad) + 1e-8))
        r[m3] = 3.0 * i[m3] - (g[m3] + b[m3])
        
    rgb = np.stack([r, g, b], axis=-1)
    return np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)


def process_color_on_intensity(rgb_image: np.ndarray, intensity_fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """
    Quy trình chuẩn hóa xử lý ảnh màu (Chương 3, slide 53):
    1. Chuyển RGB sang HSI
    2. Áp dụng hàm xử lý lên kênh cường độ I (ở định dạng uint8 [0, 255])
    3. Ghép lại với H và S gốc, chuyển ngược về RGB
    Đảm bảo 100% giữ nguyên sắc độ màu của ảnh gốc!
    """
    if rgb_image.ndim == 2:
        return intensity_fn(rgb_image)
        
    h, s, i = rgb_to_hsi(rgb_image)
    i_uint8 = np.clip(np.round(i * 255.0), 0, 255).astype(np.uint8)
    
    # Xử lý kênh cường độ
    i_processed = intensity_fn(i_uint8)
    i_new = i_processed.astype(np.float64) / 255.0
    
    return hsi_to_rgb(h, s, i_new)
