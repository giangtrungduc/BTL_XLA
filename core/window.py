"""
Module core/window.py
Tiện ích cửa sổ trượt (Sliding Window View) dùng chung cho các bộ lọc không gian (Chương 2).
Tận dụng stride_tricks của NumPy để vector hóa, đạt tốc độ tính toán cao
nhưng tuân thủ 100% công thức toán học trong bài giảng.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from core.padding import pad_image


def get_sliding_windows(channel: np.ndarray, kernel_size: int, pad_mode: str = "mirror") -> np.ndarray:
    """
    Tạo các cửa sổ trượt kích thước k x k cho ảnh 1 kênh 2D.
    
    Tham số:
        channel: Mảng 2D (H, W).
        kernel_size: Kích thước cửa sổ (phải là số lẻ: 3, 5, 7, ...).
        pad_mode: 'mirror', 'replicate', 'zero'.
        
    Trả về:
        Mảng view 4D có shape (H, W, kernel_size, kernel_size).
    """
    assert kernel_size % 2 == 1, "Kích thước cửa sổ phải là số lẻ (3, 5, 7, ...)"
    pad_len = kernel_size // 2
    padded = pad_image(channel, pad_len, pad_len, mode=pad_mode)
    windows = sliding_window_view(padded, (kernel_size, kernel_size))
    return windows


def convolve2d(channel: np.ndarray, kernel: np.ndarray, pad_mode: str = "mirror") -> np.ndarray:
    """
    Tích chập / tương quan không gian 2D cho ảnh 1 kênh (Chương 2, slide 91-100):
        g(x, y) = sum_{s=-a..a} sum_{t=-b..b} w(s, t) * f(x + s, y + t)

    Cài đặt bằng cách "dịch ảnh rồi cộng có trọng số": với mỗi hệ số w(s, t) của mặt nạ,
    lấy ảnh đã đệm biên dịch đi (s, t) rồi nhân với w(s, t) và cộng dồn.
    Kết quả giống hệt nhân từng cửa sổ với mặt nạ, nhưng chỉ tốn bộ nhớ O(H x W)
    nên chạy được với ảnh lớn (vài chục megapixel).

    Trả về mảng float32 (chưa làm tròn, có thể âm).
    """
    kh, kw = kernel.shape
    assert kh % 2 == 1 and kw % 2 == 1, "Kích thước mặt nạ phải là số lẻ"
    ph, pw = kh // 2, kw // 2
    padded = pad_image(channel.astype(np.float32), ph, pw, mode=pad_mode)
    h, w = channel.shape[:2]
    out = np.zeros((h, w), dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            wij = float(kernel[i, j])
            if wij != 0.0:
                out += wij * padded[i:i + h, j:j + w]
    return out
