"""
Module core/scratch.py
Giảm vết xước và vết nứt trên ảnh cổ theo phương pháp ghép từ các bộ lọc (Chương 6, Chương 5, Chương 2).
Toàn bộ thuật toán được cài đặt trực tiếp 100% bằng NumPy (Code from scratch), không dùng OpenCV.

Quy trình:
  1. Tìm vết xước:
     - Lấy ảnh trừ ảnh đã lọc Median hoặc dùng trị tuyệt đối |Laplacian|.
     - Phân ngưỡng tự động bằng thuật toán Otsu (Chương 6, slide 71).
  2. Làm sạch mặt nạ vết xước:
     - Áp dụng các phép toán hình thái học (Đóng, Giãn, Mở) - Chương 5.
     - Lọc thành phần liên thông để giữ lại các vệt dài/mảnh.
  3. Điền vết xước (Inpainting thay thế):
     - Tại các điểm thuộc mặt nạ vết xước (mask > 0), CHỈ lấy các điểm lành lặn ngoài mặt nạ (mask == 0)
       trong cửa sổ lân cận để tính Median hoặc Trung bình k lân cận gần nhất (Chương 2, slide 116).
     - Giữ nguyên 100% các điểm ảnh lành lặn.

Lưu ý học thuật: Phương pháp này xử lý rất tốt các vết xước mảnh và vết nứt sợi,
chưa xử lý được các mảng rách thủng hay vết ố loang lổ quá lớn (cần mô hình học sâu chuyên dụng).
"""
from typing import Tuple
import numpy as np

from core.denoise import median_filter
from core.sharpen import compute_laplacian
from core.morphology import morph_close, dilate, morph_open, connected_components
from core.padding import pad_image


def otsu_threshold(image: np.ndarray) -> int:
    """
    Thuật toán phân ngưỡng Otsu cực đại hóa phương sai giữa các lớp (Chương 6, slide 71).
    Cài đặt thuần NumPy:
        sigma_B^2(t) = omega_0(t) * omega_1(t) * [mu_0(t) - mu_1(t)]^2.
    """
    if image.ndim == 3:
        gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    else:
        gray = image
        
    gray_uint8 = np.clip(np.round(gray), 0, 255).astype(np.uint8)
    hist = np.bincount(gray_uint8.ravel(), minlength=256).astype(np.float64)
    total = float(gray_uint8.size)
    if total == 0:
        return 128
        
    p = hist / total
    # Xác suất tích lũy omega(t)
    omega = np.cumsum(p)
    # Kỳ vọng tích lũy mu(t)
    bins = np.arange(256, dtype=np.float64)
    mu = np.cumsum(p * bins)
    mu_global = mu[-1]
    
    # Tính phương sai giữa các lớp sigma_B^2 cho mọi ngưỡng t từ 0 đến 254
    # Tránh chia cho 0 khi omega = 0 hoặc omega = 1
    valid = (omega > 1e-6) & (omega < (1.0 - 1e-6))
    if not np.any(valid):
        return 128
        
    # sigma_B^2(t) = (mu_global * omega(t) - mu(t))^2 / (omega(t) * (1 - omega(t)))
    numerator = np.square(mu_global * omega - mu)
    denominator = omega * (1.0 - omega)
    
    sigma_b_squared = np.zeros(256, dtype=np.float64)
    sigma_b_squared[valid] = numerator[valid] / denominator[valid]
    
    best_threshold = int(np.argmax(sigma_b_squared))
    return best_threshold


def detect_scratch_mask(
    image: np.ndarray, 
    sensitivity: float = 0.5, 
    method: str = "median_diff"
) -> np.ndarray:
    """
    Phát hiện mặt nạ vết xước và nứt sáng màu (Chương 6, slide 64-65, 71).
    
    Tham số:
        image: Ảnh xám hoặc RGB uint8.
        sensitivity: Độ nhạy (0.0 đến 1.0, mặc định 0.5).
                     Càng cao thì bắt càng nhiều vết xước nhỏ.
        method: 'median_diff' (ảnh trừ median) hoặc 'laplacian' (|Laplacian|).
        
    Trả về:
        Mặt nạ nhị phân uint8 (255: vết xước, 0: nền lành lặn).
    """
    if image.ndim == 3:
        gray = (0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]).astype(np.uint8)
    else:
        gray = image.copy()
        
    if method == "laplacian":
        lap = compute_laplacian(gray, mode="8-neighbor")
        feature_map = np.clip(np.abs(lap), 0, 255).astype(np.uint8)
    else:
        # Mặc định: Lấy ảnh trừ ảnh đã lọc Median 5x5
        med5 = median_filter(gray, kernel_size=5)
        diff = gray.astype(np.float32) - med5.astype(np.float32)
        feature_map = np.clip(diff, 0, 255).astype(np.uint8)
        
    # Phân ngưỡng Otsu
    otsu_th = otsu_threshold(feature_map)
    # Điều chỉnh ngưỡng theo độ nhạy sensitivity:
    # sensitivity càng cao -> ngưỡng càng hạ thấp để bắt thêm vết xước
    th = int(otsu_th * (1.5 - sensitivity))
    th = max(15, min(th, 230))
    
    raw_mask = (feature_map > th).astype(np.uint8) * 255
    
    # Làm sạch mặt nạ:
    # 1. Đóng để nối các vết xước đứt khúc
    closed = morph_close(raw_mask, kernel_size=3)
    # 2. Giãn nhẹ để bao phủ trọn vẹn mép vết nứt
    dilated = dilate(closed, kernel_size=3)
    # 3. Lọc thành phần liên thông (bỏ các chấm nhiễu đơn lẻ < 4 pixel)
    filtered_mask, _ = connected_components(dilated, min_size=4)
    
    return filtered_mask


def fill_masked(
    image: np.ndarray, 
    mask: np.ndarray, 
    method: str = "median", 
    kernel_size: int = 7, 
    k: int = 5
) -> np.ndarray:
    """
    Điền vết xước bằng các bộ lọc Chương 2 (Median hoặc K-Nearest Neighbor Mean).
    ĐẶC ĐIỂM CỐT LÕI: Tại các điểm mask > 0, CHỈ LẤY các điểm lành lặn ngoài mặt nạ (mask == 0)
    trong cửa sổ lân cận để nội suy thay thế. Giữ nguyên 100% các điểm mask == 0.
    
    Tham số:
        image: Ảnh 2D hoặc 3D uint8.
        mask: Mặt nạ nhị phân 2D uint8 (255: điểm cần điền, 0: giữ nguyên).
        method: 'median' hoặc 'k_nearest'.
        kernel_size: Kích thước cửa sổ nội suy lân cận (7x7 hoặc 9x9).
        k: Số lân cận gần nhất nếu dùng k_nearest.
        
    Trả về:
        Ảnh sau khi đã điền và làm mờ các vết xước.
    """
    pad_len = kernel_size // 2
    is_color = (image.ndim == 3)
    
    # Tọa độ các điểm cần điền
    scratch_y, scratch_x = np.where(mask > 0)
    if len(scratch_y) == 0:
        return image.copy()
        
    result = image.copy()
    
    # Đệm ảnh và mặt nạ
    padded_mask = pad_image(mask, pad_len, pad_len, mode="replicate")
    
    if not is_color:
        padded_img = pad_image(image, pad_len, pad_len, mode="mirror")
        
        for y, x in zip(scratch_y, scratch_x):
            py, px = y + pad_len, x + pad_len
            win_img = padded_img[py - pad_len : py + pad_len + 1, px - pad_len : px + pad_len + 1]
            win_mask = padded_mask[py - pad_len : py + pad_len + 1, px - pad_len : px + pad_len + 1]
            
            # Chỉ lấy các điểm lành lặn ngoài mặt nạ
            valid_pixels = win_img[win_mask == 0]
            if len(valid_pixels) > 0:
                if method == "k_nearest" and len(valid_pixels) >= k:
                    # Lấy k điểm gần với trung vị của các điểm hợp lệ
                    local_med = np.median(valid_pixels)
                    diffs = np.abs(valid_pixels.astype(float) - local_med)
                    sorted_indices = np.argsort(diffs)[:k]
                    val = np.mean(valid_pixels[sorted_indices])
                else:
                    val = np.median(valid_pixels)
                result[y, x] = np.clip(np.round(val), 0, 255).astype(np.uint8)
                
    else:
        # Xử lý trên từng kênh màu
        padded_img = pad_image(image, pad_len, pad_len, mode="mirror")
        for c in range(3):
            for y, x in zip(scratch_y, scratch_x):
                py, px = y + pad_len, x + pad_len
                win_img = padded_img[py - pad_len : py + pad_len + 1, px - pad_len : px + pad_len + 1, c]
                win_mask = padded_mask[py - pad_len : py + pad_len + 1, px - pad_len : px + pad_len + 1]
                
                valid_pixels = win_img[win_mask == 0]
                if len(valid_pixels) > 0:
                    val = np.median(valid_pixels)
                    result[y, x, c] = np.clip(np.round(val), 0, 255).astype(np.uint8)
                    
    return result
