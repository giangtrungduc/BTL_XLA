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
from core.window import convolve2d
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
        return convolve2d(ch, kernel, pad_mode)

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
        gx = convolve2d(ch, SOBEL_X, pad_mode)
        gy = convolve2d(ch, SOBEL_Y, pad_mode)
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
    k_lap: float = 0.6,
    gamma_val: float = 1.0
) -> np.ndarray:
    """
    Làm nét kết hợp Laplacian + Sobel (ý tưởng của Chương 6, slide 58-60):
      1. Tính Laplacian L = nabla^2 f (chi tiết bậc hai: viền và cả nhiễu).
      2. Tính độ lớn gradient Sobel M = |Gx| + |Gy| (mạnh ở biên, yếu ở vùng phẳng).
      3. Làm mịn M bằng lọc trung bình 5x5 rồi chuẩn hóa về [0, 1] -> mặt nạ W.
         W gần 1 tại biên thật, gần 0 tại vùng phẳng có nhiễu hạt.
      4. g = f - k * (W * L): chỉ làm nét ở nơi có biên, không khuếch đại nhiễu vùng phẳng.
      5. (Tùy chọn) biến đổi Gamma để chỉnh lại dải xám.

    So với Laplacian thuần (g = f - k * L), cách này ít làm nổi hạt nhiễu hơn,
    phù hợp với ảnh cũ vốn còn sót nhiễu sau bước khử nhiễu.
    """
    img_f = image.astype(np.float32)
    lap = compute_laplacian(image, mode="8-neighbor")

    sobel_mag = sobel_filter(image)  # uint8, đã cắt về [0, 255]
    sobel_smooth = arithmetic_mean_filter(sobel_mag, kernel_size=5).astype(np.float32)
    if sobel_smooth.ndim == 3:
        # Dùng chung một mặt nạ cho cả 3 kênh để không lệch màu
        sobel_smooth = np.mean(sobel_smooth, axis=-1, keepdims=True)
    weight = sobel_smooth / (np.max(sobel_smooth) + 1e-6)

    sharpened = img_f - k_lap * weight * lap
    out = np.clip(np.round(sharpened), 0, 255).astype(np.uint8)

    if gamma_val != 1.0:
        out = gamma_transform(out, gamma=gamma_val)
    return out
