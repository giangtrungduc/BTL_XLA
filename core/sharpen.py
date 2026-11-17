"""
Module core/sharpen.py
Làm sắc nét ảnh bằng đạo hàm không gian (Chương 2, slide 119-123; Chương 6, slide 20-28, 58-60).
Toàn bộ thuật toán được cài đặt trực tiếp bằng NumPy (Code from scratch), không dùng OpenCV.

Bao gồm:
  1. Laplacian 4 lân cận & 8 lân cận (Đạo hàm bậc hai)
  2. Sobel Gradient (Đạo hàm bậc nhất)
  3. Chế độ kết hợp nâng cao bài giảng (Laplacian + Sobel + Smooth + Mask + Gamma) - Ch.6 tr.58-60
"""
import numpy as np
from core.window import get_sliding_windows
from core.denoise import arithmetic_mean_filter
from core.intensity import gamma_transform


# Các ma trận kernel chuẩn theo giáo trình
LAPLACIAN_4_NEIGHBOR = np.array([
    [0,  1, 0],
    [1, -4, 1],
    [0,  1, 0]
], dtype=np.float32)

LAPLACIAN_8_NEIGHBOR = np.array([
    [1,  1, 1],
    [1, -8, 1],
    [1,  1, 1]
], dtype=np.float32)

SOBEL_X = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

SOBEL_Y = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float32)


def compute_laplacian(image: np.ndarray, mode: str = "8-neighbor", pad_mode: str = "mirror") -> np.ndarray:
    """
    Tính đạo hàm bậc hai Laplacian nabla^2 f (Chương 2, slide 120; Chương 6, slide 25).
    
    Tham số:
        image: Mảng 2D (hoặc 3D).
        mode: '4-neighbor' hoặc '8-neighbor'.
        
    Trả về:
        Mảng float32 chứa giá trị đạo hàm bậc 2 (có thể âm hoặc dương).
    """
    kernel = LAPLACIAN_8_NEIGHBOR if "8" in mode else LAPLACIAN_4_NEIGHBOR

    def _op(ch):
        win = get_sliding_windows(ch.astype(np.float32), 3, pad_mode)
        return np.sum(win * kernel, axis=(-2, -1))

    if image.ndim == 2:
        return _op(image)
    else:
        channels = [_op(image[..., c]) for c in range(image.shape[2])]
        return np.stack(channels, axis=-1)


def laplacian_sharpen(
    image: np.ndarray, 
    k: float = 0.4, 
    mode: str = "8-neighbor", 
    pad_mode: str = "mirror"
) -> np.ndarray:
    """
    Làm sắc nét ảnh bằng toán tử Laplacian (Chương 2, slide 120; Chương 6, slide 27).
    Công thức: g(x, y) = f(x, y) - k * nabla^2 f(x, y)
    (Trừ vì hệ số tâm ma trận là số âm: -4 hoặc -8).
    """
    lap = compute_laplacian(image, mode=mode, pad_mode=pad_mode)
    img_f = image.astype(np.float32)
    sharpened = img_f - k * lap
    return np.clip(np.round(sharpened), 0, 255).astype(np.uint8)


def sobel_filter(image: np.ndarray, pad_mode: str = "mirror") -> np.ndarray:
    """
    Độ lớn Gradient Sobel (Chương 6, slide 20-24).
    M = |Gx| + |Gy|.
    """
    def _op(ch):
        win = get_sliding_windows(ch.astype(np.float32), 3, pad_mode)
        gx = np.sum(win * SOBEL_X, axis=(-2, -1))
        gy = np.sum(win * SOBEL_Y, axis=(-2, -1))
        mag = np.abs(gx) + np.abs(gy)
        return mag

    if image.ndim == 2:
        mag = _op(image)
    else:
        channels = [_op(image[..., c]) for c in range(image.shape[2])]
        mag = np.stack(channels, axis=-1)
        
    return np.clip(np.round(mag), 0, 255).astype(np.uint8)


def combined_sharpen(
    image: np.ndarray, 
    k_lap: float = 0.4, 
    gamma_val: float = 0.85
) -> np.ndarray:
    """
    Quy trình làm sắc nét kết hợp kinh điển (Chương 6, slide 58-60):
    1. Ảnh gốc f
    2. Laplacian của f -> f_lap = f - k * nabla^2 f
    3. Gradient Sobel của f -> M
    4. Làm mịn Sobel bằng bộ lọc trung bình 5x5 -> M_smooth
    5. Tạo mặt nạ: mask = (f_lap * M_smooth) chuẩn hóa
    6. Cộng với ảnh gốc: f_sharpened = f + mask
    7. Áp dụng biến đổi Gamma để cân bằng dải xám
    """
    img_f = image.astype(np.float32)
    
    # 1. Laplacian
    lap = compute_laplacian(image, mode="8-neighbor")
    f_lap = img_f - k_lap * lap
    
    # 2. Sobel
    sobel_mag = sobel_filter(image).astype(np.float32)
    
    # 3. Làm mịn Sobel 5x5
    sobel_smooth = arithmetic_mean_filter(np.clip(sobel_mag, 0, 255).astype(np.uint8), kernel_size=5).astype(np.float32)
    
    # 4. Nhân mặt nạ (chuẩn hóa tỷ lệ [0, 1])
    norm_sobel = sobel_smooth / (np.max(sobel_smooth) + 1e-6)
    mask = (f_lap - img_f) * norm_sobel
    
    # 5. Cộng ảnh gốc
    f_sharp = np.clip(np.round(img_f + mask), 0, 255).astype(np.uint8)
    
    # 6. Biến đổi Gamma
    if gamma_val != 1.0:
        f_sharp = gamma_transform(f_sharp, gamma=gamma_val)
        
    return f_sharp
