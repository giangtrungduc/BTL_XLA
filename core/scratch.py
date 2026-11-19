"""
Module core/scratch.py
Phát hiện và điền vết xước / bụi / vết nứt mảnh trên ảnh cũ (Chương 2, 5, 6).
Toàn bộ thuật toán cài đặt bằng NumPy (code from scratch), không dùng OpenCV.

Quy trình:
  1. Bản đồ đặc trưng: hiệu giữa ảnh và ảnh lọc trung vị 7x7.
     Lọc trung vị xóa các cấu trúc mảnh hơn nửa cửa sổ (vết xước rộng 1-3 px, hạt bụi),
     nên hiệu số lớn đúng tại các vết đó (Chương 2: lọc thống kê thứ tự).
     Vết xước trên ảnh in thường là vệt SÁNG (lớp nhũ tương bị bong), nên mặc định chỉ lấy phần dương.
  2. Phân ngưỡng tự động Otsu trên bản đồ đặc trưng (Chương 6, slide 71), điều chỉnh theo độ nhạy.
  3. Lọc theo hình dạng bằng thành phần liên thông 8 lân cận (Chương 5, slide 64-70):
       - bỏ thành phần quá nhỏ (điểm nhiễu muối tiêu đơn lẻ),
       - giữ thành phần dạng đường (dài, mảnh) hoặc đốm bụi nhỏ,
       - bỏ mảng lớn đặc (thường là chi tiết sáng thật của ảnh).
  4. Giãn mặt nạ 3x3 (Chương 5) để phủ trọn mép vết.
  5. Điền vết theo kiểu "bóc vỏ hành": lặp lại, mỗi vòng điền các điểm ở mép mặt nạ bằng
     trung vị của các điểm lành trong cửa sổ lân cận, rồi coi chúng là điểm lành cho vòng sau.
     Nhờ vậy điền được cả vết rộng hơn cửa sổ lọc. Điểm ngoài mặt nạ giữ nguyên 100%.

Giới hạn: không khôi phục được vùng rách/mất lớn (không còn thông tin xung quanh để nội suy).
"""
from typing import Tuple
import numpy as np

from core.denoise import median_filter
from core.sharpen import compute_laplacian
from core.morphology import dilate, morph_close, label_components
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


def _component_is_defect(size: int, bh: int, bw: int, min_length: int, max_speck: int) -> bool:
    """Luật hình dạng: vệt dài-mảnh (vết xước/nứt) hoặc đốm nhỏ (bụi) thì coi là khuyết tật."""
    length = max(bh, bw)
    fill_ratio = size / float(bh * bw)
    elongation = length / float(max(1, min(bh, bw)))
    is_line = length >= min_length and (elongation >= 3.0 or fill_ratio <= 0.35)
    is_speck = size <= max_speck
    return is_line or is_speck


def detect_scratch_mask(
    image: np.ndarray,
    sensitivity: float = 0.5,
    method: str = "median_diff",
    polarity: str = "bright",
    min_size: int = 4,
    min_length: int = 12,
    max_speck: int = None,
) -> np.ndarray:
    """
    Phát hiện mặt nạ vết xước.

    Tham số:
        image: Ảnh xám hoặc RGB uint8.
        sensitivity: Độ nhạy 0..1. Càng cao ngưỡng càng thấp, bắt được vết mờ hơn.
        method: 'median_diff' (ảnh - trung vị 7x7) hoặc 'laplacian' (|Laplacian|).
        polarity: 'bright' (vệt sáng), 'dark' (vệt tối) hoặc 'both'.
        min_size: Thành phần nhỏ hơn số điểm này bị bỏ (nhiễu muối tiêu).
        min_length: Chiều dài tối thiểu (px) để coi là vệt xước.
        max_speck: Đốm nhỏ hơn hoặc bằng số điểm này được coi là bụi
                   (mặc định tăng theo độ nhạy: 0.5 -> 70 px, 0.9 -> 240 px).

    Trả về:
        Mặt nạ uint8 (255: vết cần điền, 0: điểm lành).
    """
    if image.ndim == 3:
        gray = np.clip(np.round(0.299 * image[..., 0] + 0.587 * image[..., 1]
                                + 0.114 * image[..., 2]), 0, 255).astype(np.uint8)
    else:
        gray = image

    if method == "laplacian":
        lap = compute_laplacian(gray, mode="8-neighbor")
        if polarity == "bright":
            feat = -lap          # điểm sáng hơn xung quanh -> Laplacian âm
        elif polarity == "dark":
            feat = lap
        else:
            feat = np.abs(lap)
    else:
        med = median_filter(gray, kernel_size=7)
        diff = gray.astype(np.float32) - med.astype(np.float32)
        if polarity == "bright":
            feat = diff
        elif polarity == "dark":
            feat = -diff
        else:
            feat = np.abs(diff)
    feature_map = np.clip(feat, 0, 255).astype(np.uint8)

    # Ngưỡng Otsu chỉ tính trên các điểm có phản hồi (> 0) để không bị nền phẳng chi phối
    nonzero = feature_map[feature_map > 0]
    otsu_th = otsu_threshold(nonzero) if nonzero.size > 0 else 128
    sens = max(0.05, min(float(sensitivity), 0.95))
    if max_speck is None:
        max_speck = int(round(20 + 200 * sens ** 2 * (1.0 if sens <= 0.5 else 1.25)))
    # Độ nhạy 0.5 -> ngưỡng = 0.5 x Otsu; độ nhạy càng cao ngưỡng càng thấp
    th = int(round(otsu_th * (1.0 - sens)))
    # Sàn ngưỡng theo mức nhiễu: sigma ước lượng bằng MAD của hiệu số (bền với ngoại lai).
    # Ảnh nhiễu hạt mạnh -> ngưỡng tự nâng lên, tránh coi hạt nhiễu là vết xước.
    noise_sigma = 1.4826 * float(np.median(np.abs(diff if method != "laplacian" else lap)))
    th = max(th, int(round(12 + (4.5 - 2.0 * sens) * noise_sigma)))
    th = max(12, min(th, 230))
    raw_mask = feature_map > th
    # Phép đóng 3x3 (Chương 5): nối các đoạn vết nứt bị đứt quãng trước khi xét hình dạng
    raw_mask = morph_close(raw_mask.astype(np.uint8) * 255, kernel_size=3) > 0

    # Lọc theo hình dạng bằng thành phần liên thông
    labels, comps = label_components(raw_mask)
    keep = np.zeros(len(comps) + 1, dtype=bool)
    for idx, (size, y0, x0, y1, x1) in enumerate(comps, start=1):
        if size >= min_size and _component_is_defect(size, y1 - y0 + 1, x1 - x0 + 1, min_length, max_speck):
            keep[idx] = True
    clean = keep[labels]

    mask = clean.astype(np.uint8) * 255
    if np.any(mask):
        mask = dilate(mask, kernel_size=3)
    return mask


def fill_masked(
    image: np.ndarray,
    mask: np.ndarray,
    method: str = "median",
    kernel_size: int = 5,
    k: int = 5,
    max_iter: int = 60,
) -> np.ndarray:
    """
    Điền các điểm thuộc mặt nạ bằng thống kê của điểm lành lân cận (Chương 2, slide 109-116).

    Mỗi vòng lặp:
      - "Mép" = các điểm còn thuộc mặt nạ có ít nhất 1 điểm lành trong cửa sổ kernel_size x kernel_size.
      - Mỗi điểm mép nhận giá trị trung vị (method='median') hoặc trung bình k điểm lành gần trung vị nhất
        (method='k_nearest') của CÁC ĐIỂM LÀNH trong cửa sổ.
      - Các điểm vừa điền trở thành điểm lành cho vòng sau.
    Lặp đến khi hết mặt nạ (hoặc quá max_iter vòng). Điểm ngoài mặt nạ không bị thay đổi.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    if mask is None or not np.any(mask > 0):
        return image.copy()

    is_color = image.ndim == 3
    pad = kernel_size // 2
    work = image.astype(np.float32).copy()
    if not is_color:
        work = work[..., np.newaxis]
    n_ch = work.shape[2]
    remaining = mask > 0

    for _ in range(max_iter):
        if not np.any(remaining):
            break
        valid = ~remaining
        padded_valid = pad_image(valid.astype(np.uint8), pad, pad, mode="zero").astype(bool)
        valid_win = sliding_window_view(padded_valid, (kernel_size, kernel_size))
        has_valid = valid_win.any(axis=(-2, -1))
        frontier = remaining & has_valid
        if not np.any(frontier):
            break
        ys, xs = np.nonzero(frontier)
        vw = valid_win[ys, xs].reshape(len(ys), -1)               # (N, k*k)
        new_vals = np.empty((len(ys), n_ch), dtype=np.float32)
        for c in range(n_ch):
            padded_c = pad_image(work[..., c], pad, pad, mode="mirror")
            win_c = sliding_window_view(padded_c, (kernel_size, kernel_size))
            vals = win_c[ys, xs].reshape(len(ys), -1).astype(np.float32)
            vals = np.where(vw, vals, np.nan)
            med = np.nanmedian(vals, axis=1)
            if method == "k_nearest":
                dist = np.abs(vals - med[:, None])
                dist = np.where(np.isnan(dist), np.inf, dist)
                kk = max(1, min(k, vals.shape[1]))
                idx = np.argsort(dist, axis=1)[:, :kk]
                chosen = np.take_along_axis(vals, idx, axis=1)
                chosen_valid = np.take_along_axis(vw, idx, axis=1)
                chosen = np.where(chosen_valid, chosen, np.nan)
                med = np.nanmean(chosen, axis=1)
            new_vals[:, c] = med
        work[ys, xs, :] = new_vals
        remaining = remaining & ~frontier

    if np.any(remaining):
        # Phòng hờ: vùng còn sót (rất lớn) được điền bằng trung vị toàn cục của điểm lành
        valid_all = ~(mask > 0)
        for c in range(n_ch):
            fallback = np.median(work[..., c][valid_all]) if np.any(valid_all) else 128.0
            work[..., c][remaining] = fallback

    out = np.clip(np.round(work), 0, 255).astype(np.uint8)
    return out if is_color else out[..., 0]
