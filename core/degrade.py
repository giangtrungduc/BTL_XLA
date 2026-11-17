"""
Module core/degrade.py
Mô phỏng ảnh cũ từ ảnh sạch (Chương 2, slide 128-132; Chương 0, slide 8).
Tạo ảnh hư tổn nhân tạo để kiểm thử và đối chiếu định lượng (Ground Truth vs Restored).

Bao gồm:
  1. Thêm nhiễu ngẫu nhiên: g = f + eta (Gauss, Muối - Tiêu)
  2. Giảm độ tương phản / bạc màu hóa bằng hàm Gamma
  3. Mô phỏng vết xước và bụi phim (Film Scratches)
"""
from typing import Tuple
import numpy as np

from core.noise import add_gaussian_noise, add_salt_and_pepper_noise
from core.intensity import gamma_transform


def simulate_old_photo(
    clean_image: np.ndarray,
    noise_type: str = "salt_pepper",
    noise_level: float = 0.02,
    gamma_val: float = 1.25,
    num_scratches: int = 5,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tạo ảnh cũ giả lập từ ảnh gốc sạch.
    
    Tham số:
        clean_image: Ảnh sạch ban đầu.
        noise_type: 'salt_pepper', 'gaussian', hoặc 'both'.
        noise_level: Cường độ nhiễu.
        gamma_val: Hệ số làm bạc màu / giảm tương phản (gamma > 1).
        num_scratches: Số lượng vết xước mô phỏng.
        seed: Hạt giống ngẫu nhiên.
        
    Trả về:
        degraded: Ảnh sau khi làm cũ.
        scratch_mask: Mặt nạ nhị phân của các vết xước nhân tạo (dùng để nghiệm thu).
    """
    if seed is not None:
        np.random.seed(seed)
        
    h, w = clean_image.shape[:2]
    img = clean_image.copy()
    
    # 1. Giảm tương phản / bạc màu hóa bằng Gamma
    if gamma_val != 1.0:
        img = gamma_transform(img, gamma=gamma_val)
        
    # 2. Thêm nhiễu
    if noise_type in ("gaussian", "both"):
        img = add_gaussian_noise(img, mean=0, std=noise_level * 255.0, seed=seed)
    if noise_type in ("salt_pepper", "both"):
        img = add_salt_and_pepper_noise(img, salt_prob=noise_level, pepper_prob=noise_level, seed=seed)
        
    # 3. Vẽ vết xước phim (Scratches)
    scratch_mask = np.zeros((h, w), dtype=np.uint8)
    for _ in range(num_scratches):
        # Chọn điểm bắt đầu và kết thúc
        start_x = np.random.randint(10, w - 10)
        start_y = np.random.randint(10, h // 3)
        end_x = int(start_x + np.random.normal(0, 15))
        end_y = np.random.randint(h // 2, h - 10)
        
        # Tạo đường đi hơi cong/gãy khúc
        num_pts = max(10, int(np.hypot(end_x - start_x, end_y - start_y) // 2))
        ys = np.linspace(start_y, end_y, num_pts).astype(int)
        xs = np.linspace(start_x, end_x, num_pts) + np.random.normal(0, 1.2, num_pts)
        xs = np.clip(np.round(xs), 0, w - 1).astype(int)
        
        for y, x in zip(ys, xs):
            scratch_mask[y, x] = 255
            # Vết xước có độ rộng 1-2 pixel
            if x + 1 < w:
                scratch_mask[y, x + 1] = 255
                
    # Gán màu sáng trắng cho vết xước
    if img.ndim == 2:
        img[scratch_mask > 0] = 245
    else:
        img[scratch_mask > 0] = [245, 245, 245]
        
    return img, scratch_mask
