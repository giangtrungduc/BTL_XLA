"""
Module core/denoise.py
Các bộ lọc không gian khử nhiễu đầy đủ theo bài giảng (Chương 2, slide 91-117, 133-141).
Toàn bộ thuật toán được cài đặt trực tiếp bằng NumPy (Code from scratch), không dùng OpenCV.

Bao gồm:
  1. Median Filter (Lọc trung vị) - Slide tr.109, 137
  2. Arithmetic Mean Filter (Trung bình số học) - Slide tr.98, 134
  3. Geometric Mean Filter (Trung bình hình học) - Slide tr.134
  4. Harmonic Mean Filter (Trung bình điều hòa) - Slide tr.135
  5. Contraharmonic Mean Filter (Trung bình điều hòa ngược Q) - Slide tr.135
  6. Max Filter (Lọc cực đại) - Slide tr.137
  7. Min Filter (Lọc cực tiểu) - Slide tr.137
  8. Midpoint Filter (Lọc điểm giữa) - Slide tr.138
  9. Alpha-trimmed Mean Filter (Lọc trung bình cắt tỉa alpha) - Slide tr.139
 10. K-Nearest Neighbor Mean Filter (Trung bình k lân cận gần nhất) - Slide tr.116
 11. Gaussian Filter (Lọc thông thấp Gauss) - Slide tr.106
"""
import numpy as np
from core.window import get_sliding_windows, convolve2d


def _apply_op(image: np.ndarray, op_2d_fn) -> np.ndarray:
    """Hàm phụ trợ hỗ trợ xử lý cả ảnh mức xám (2D) và ảnh màu (3D)."""
    if image.ndim == 2:
        return op_2d_fn(image)
    else:
        channels = [op_2d_fn(image[..., c]) for c in range(image.shape[2])]
        return np.stack(channels, axis=-1)


# 1. Median Filter (Ch.2 tr.109, 137)
def median_filter(image: np.ndarray, kernel_size: int = 3, pad_mode: str = "mirror") -> np.ndarray:
    """
    Bộ lọc trung vị: f_hat(x, y) = median(S_xy).
    Đặc biệt hiệu quả trong việc loại bỏ nhiễu xung (muối tiêu).
    """
    def _op(ch):
        win = get_sliding_windows(ch, kernel_size, pad_mode)
        med = np.median(win, axis=(-2, -1))
        return np.clip(np.round(med), 0, 255).astype(np.uint8)
        
    return _apply_op(image, _op)


# 2. Arithmetic Mean Filter (Ch.2 tr.98, 134)
def arithmetic_mean_filter(image: np.ndarray, kernel_size: int = 3, pad_mode: str = "mirror") -> np.ndarray:
    """
    Bộ lọc trung bình số học: f_hat(x, y) = (1 / mn) * sum_{(s,t) in S_xy} f(s, t).
    Làm mịn đều tổng thể ảnh, khử nhiễu Gauss và nhiễu đều.
    """
    box = np.full((kernel_size, kernel_size), 1.0 / (kernel_size * kernel_size), dtype=np.float32)

    def _op(ch):
        mean = convolve2d(ch, box, pad_mode)
        return np.clip(np.round(mean), 0, 255).astype(np.uint8)
        
    return _apply_op(image, _op)


# 3. Geometric Mean Filter (Ch.2 tr.134)
def geometric_mean_filter(image: np.ndarray, kernel_size: int = 3, pad_mode: str = "mirror") -> np.ndarray:
    """
    Bộ lọc trung bình hình học: f_hat(x, y) = [ prod_{(s,t) in S_xy} f(s, t) ]^(1 / mn).
    Dùng phép cộng logarit để tránh tràn số: exp( (1 / mn) * sum( ln(f + eps) ) ).
    Giữ chi tiết biên tốt hơn trung bình số học.
    """
    eps = 1e-5
    mn = float(kernel_size * kernel_size)

    ones = np.ones((kernel_size, kernel_size), dtype=np.float32)

    def _op(ch):
        log_sum = convolve2d(np.log(ch.astype(np.float32) + eps), ones, pad_mode)
        geo_mean = np.exp(log_sum / mn)
        return np.clip(np.round(geo_mean), 0, 255).astype(np.uint8)

    return _apply_op(image, _op)


# 4. Harmonic Mean Filter (Ch.2 tr.135)
def harmonic_mean_filter(image: np.ndarray, kernel_size: int = 3, pad_mode: str = "mirror") -> np.ndarray:
    """
    Bộ lọc trung bình điều hòa: f_hat(x, y) = mn / sum_{(s,t) in S_xy} (1 / (f(s, t) + eps)).
    Rất tốt cho nhiễu muối và nhiễu Gauss, không dùng cho nhiễu tiêu.
    """
    eps = 1e-4
    mn = float(kernel_size * kernel_size)

    ones = np.ones((kernel_size, kernel_size), dtype=np.float32)

    def _op(ch):
        inv_sum = convolve2d(1.0 / (ch.astype(np.float32) + eps), ones, pad_mode)
        har_mean = mn / (inv_sum + 1e-8)
        return np.clip(np.round(har_mean), 0, 255).astype(np.uint8)

    return _apply_op(image, _op)


# 5. Contraharmonic Mean Filter (Ch.2 tr.135)
def contraharmonic_mean_filter(
    image: np.ndarray, 
    kernel_size: int = 3, 
    Q: float = 1.5, 
    pad_mode: str = "mirror"
) -> np.ndarray:
    """
    Bộ lọc trung bình điều hòa ngược: f_hat = sum(f^(Q+1)) / sum(f^Q).
    - Q > 0: Khử nhiễu tiêu (pepper noise).
    - Q < 0: Khử nhiễu muối (salt noise).
    - Q = 0: Trở thành Arithmetic Mean.
    - Q = -1: Trở thành Harmonic Mean.
    """
    eps = 1e-4

    ones = np.ones((kernel_size, kernel_size), dtype=np.float32)

    def _op(ch):
        # Chuẩn hóa về [0, 1] trước khi lũy thừa để tránh tràn số khi Q lớn
        f = ch.astype(np.float64) / 255.0 + eps
        num = convolve2d(np.power(f, Q + 1.0), ones, pad_mode).astype(np.float64)
        denom = convolve2d(np.power(f, Q), ones, pad_mode).astype(np.float64) + 1e-12
        res = num / denom * 255.0
        return np.clip(np.round(res), 0, 255).astype(np.uint8)

    return _apply_op(image, _op)


# 6. Max Filter (Ch.2 tr.137)
def max_filter(image: np.ndarray, kernel_size: int = 3, pad_mode: str = "mirror") -> np.ndarray:
    """Bộ lọc cực đại: f_hat(x, y) = max(S_xy). Dùng để tìm điểm sáng hoặc khử nhiễu tiêu."""
    def _op(ch):
        win = get_sliding_windows(ch, kernel_size, pad_mode)
        res = np.max(win, axis=(-2, -1))
        return res.astype(np.uint8)

    return _apply_op(image, _op)


# 7. Min Filter (Ch.2 tr.137)
def min_filter(image: np.ndarray, kernel_size: int = 3, pad_mode: str = "mirror") -> np.ndarray:
    """Bộ lọc cực tiểu: f_hat(x, y) = min(S_xy). Dùng để tìm điểm tối hoặc khử nhiễu muối."""
    def _op(ch):
        win = get_sliding_windows(ch, kernel_size, pad_mode)
        res = np.min(win, axis=(-2, -1))
        return res.astype(np.uint8)

    return _apply_op(image, _op)


# 8. Midpoint Filter (Ch.2 tr.138)
def midpoint_filter(image: np.ndarray, kernel_size: int = 3, pad_mode: str = "mirror") -> np.ndarray:
    """
    Bộ lọc điểm giữa: f_hat(x, y) = (max(S_xy) + min(S_xy)) / 2.
    Kết hợp giữa thống kê thứ tự và lấy trung bình.
    """
    def _op(ch):
        win = get_sliding_windows(ch.astype(np.float32), kernel_size, pad_mode)
        res = 0.5 * (np.max(win, axis=(-2, -1)) + np.min(win, axis=(-2, -1)))
        return np.clip(np.round(res), 0, 255).astype(np.uint8)

    return _apply_op(image, _op)


# 9. Alpha-trimmed Mean Filter (Ch.2 tr.139)
def alpha_trimmed_mean_filter(
    image: np.ndarray, 
    kernel_size: int = 3, 
    d: int = 2, 
    pad_mode: str = "mirror"
) -> np.ndarray:
    """
    Bộ lọc trung bình cắt tỉa alpha:
    Sắp xếp các điểm trong cửa sổ, bỏ đi d/2 điểm nhỏ nhất và d/2 điểm lớn nhất,
    sau đó tính trung bình cộng mn - d điểm còn lại.
    Đặc biệt tốt khi ảnh vừa có nhiễu Gauss vừa có nhiễu muối tiêu.
    """
    mn = kernel_size * kernel_size
    assert 0 <= d < mn, f"Tham số d ({d}) phải nhỏ hơn kích thước cửa sổ ({mn})"
    
    half_d = d // 2

    def _op(ch):
        win = get_sliding_windows(ch.astype(np.float32), kernel_size, pad_mode)
        # Sắp xếp các điểm trong cửa sổ: shape (H, W, mn)
        flat_sorted = np.sort(win.reshape(ch.shape[0], ch.shape[1], -1), axis=-1)
        if half_d > 0:
            trimmed = flat_sorted[:, :, half_d:(mn - half_d)]
        else:
            trimmed = flat_sorted
        res = np.mean(trimmed, axis=-1)
        return np.clip(np.round(res), 0, 255).astype(np.uint8)

    return _apply_op(image, _op)


# 10. K-Nearest Neighbor Mean Filter (Ch.2 tr.116)
def k_nearest_mean_filter(
    image: np.ndarray, 
    kernel_size: int = 3, 
    k: int = 5, 
    pad_mode: str = "mirror"
) -> np.ndarray:
    """
    Bộ lọc trung bình k lân cận gần nhất (K-Nearest Neighbor Averaging Filter):
    Trong cửa sổ kích thước k_size x k_size, chọn ra k điểm ảnh có mức xám gần nhất
    với mức xám của điểm tâm (x, y), sau đó tính trung bình cộng k điểm này.
    Làm mịn ảnh nhưng bảo toàn viền cạnh sắc nét rất tốt.
    """
    mn = kernel_size * kernel_size
    k = max(1, min(k, mn))

    def _op(ch):
        ch_f = ch.astype(np.float32)
        win = get_sliding_windows(ch_f, kernel_size, pad_mode)
        # Điểm tâm: ch_f shape (H, W, 1)
        center = ch_f[:, :, np.newaxis]
        flat_win = win.reshape(ch.shape[0], ch.shape[1], mn)
        
        # Độ lệch so với tâm
        diff = np.abs(flat_win - center)
        # Lấy chỉ số k điểm có độ lệch nhỏ nhất
        k_indices = np.argpartition(diff, k - 1, axis=-1)[:, :, :k]
        
        # Thu thập k giá trị gần nhất
        k_values = np.take_along_axis(flat_win, k_indices, axis=-1)
        res = np.mean(k_values, axis=-1)
        return np.clip(np.round(res), 0, 255).astype(np.uint8)

    return _apply_op(image, _op)


# 11. Gaussian Filter (Ch.2 tr.106)
def create_gaussian_kernel(kernel_size: int = 3, sigma: float = 1.0) -> np.ndarray:
    """Tạo ma trận trọng số Gauss chuẩn hóa 2D (Chương 2, slide 106)."""
    radius = kernel_size // 2
    s, t = np.meshgrid(np.arange(-radius, radius + 1), np.arange(-radius, radius + 1))
    kernel = np.exp(-(s**2 + t**2) / (2.0 * sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel.astype(np.float32)


def gaussian_filter(
    image: np.ndarray, 
    kernel_size: int = 3, 
    sigma: float = 1.0, 
    pad_mode: str = "mirror"
) -> np.ndarray:
    """Bộ lọc thông thấp Gaussian: làm mịn ảnh theo phân phối hình chuông Gauss."""
    kernel = create_gaussian_kernel(kernel_size, sigma)

    def _op(ch):
        conv = convolve2d(ch, kernel, pad_mode)
        return np.clip(np.round(conv), 0, 255).astype(np.uint8)

    return _apply_op(image, _op)
