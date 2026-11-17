"""
Module core/noise.py
Mô hình hóa các loại nhiễu toán học theo giáo trình Xử lý ảnh (Chương 2, slide 128-132).
Công thức tổng quát: g(x, y) = f(x, y) + eta(x, y).
Toàn bộ thuật toán được cài đặt trực tiếp bằng NumPy (Code from scratch).
"""
import numpy as np


def add_gaussian_noise(image: np.ndarray, mean: float = 0.0, std: float = 20.0, seed: int = None) -> np.ndarray:
    """
    Nhiễu Gauss (Gaussian Noise) - Chương 2, slide 129.
    Hàm mật độ xác suất phân phối chuẩn: p(z) = (1 / (sqrt(2*pi)*sigma)) * exp(-(z-mu)^2 / (2*sigma^2)).
    
    Tham số:
        image: Mảng uint8 (2D hoặc 3D).
        mean: Kỳ vọng mu.
        std: Độ lệch chuẩn sigma.
        seed: Hạt giống ngẫu nhiên (tùy chọn để tái lập).
    """
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.normal(mean, std, image.shape)
    noisy = image.astype(np.float32) + noise
    return np.clip(np.round(noisy), 0, 255).astype(np.uint8)


def add_uniform_noise(image: np.ndarray, low: float = -25.0, high: float = 25.0, seed: int = None) -> np.ndarray:
    """
    Nhiễu đều (Uniform Noise) - Chương 2, slide 130.
    Hàm mật độ xác suất phân phối đều: p(z) = 1 / (b - a) với a <= z <= b.
    """
    if seed is not None:
        np.random.seed(seed)
    noise = np.random.uniform(low, high, image.shape)
    noisy = image.astype(np.float32) + noise
    return np.clip(np.round(noisy), 0, 255).astype(np.uint8)


def add_salt_and_pepper_noise(
    image: np.ndarray, 
    salt_prob: float = 0.02, 
    pepper_prob: float = 0.02, 
    seed: int = None
) -> np.ndarray:
    """
    Nhiễu xung / Muối - Tiêu (Salt & Pepper Noise) - Chương 2, slide 131.
    Pixel nhận giá trị cực đại 255 (muối) với xác suất P_s,
    hoặc nhận giá trị cực tiểu 0 (tiêu) với xác suất P_p.
    """
    if seed is not None:
        np.random.seed(seed)
        
    noisy = image.copy()
    rnd = np.random.rand(*image.shape[:2])
    
    # Nhiễu tiêu (0: màu đen)
    noisy[rnd < pepper_prob] = 0
    # Nhiễu muối (255: màu trắng)
    noisy[rnd > (1.0 - salt_prob)] = 255
    
    return noisy
