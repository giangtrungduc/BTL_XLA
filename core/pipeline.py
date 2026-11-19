"""
Module core/pipeline.py
Quy trình khôi phục ảnh cũ có thứ tự cố định và chẩn đoán tự động (Chương 2, 3, 5, 6).
Toàn bộ thuật toán được cài đặt bằng NumPy (code from scratch).

THỨ TỰ CỐ ĐỊNH 4 BƯỚC:
    1. Vết xước  -> 2. Khử nhiễu -> 3. Sáng & tương phản -> 4. Làm nét

Lý do thứ tự:
  - Vết xước được phát hiện TRƯỚC khử nhiễu: lọc trung vị làm mờ/xóa một phần vết mảnh,
    nếu phát hiện sau thì mặt nạ bắt nhầm vào biên thật của ảnh.
  - Khử nhiễu trước tăng tương phản: kéo giãn histogram cũng kéo giãn biên độ nhiễu.
  - Làm nét cuối cùng: đạo hàm khuếch đại nhiễu (Ch.6 tr.29-31), nên phải làm sạch trước.

NGUYÊN TẮC:
  - Mỗi bước là hàm thuần khiết f(ảnh, tham số) -> ảnh mới.
  - run_pipeline() luôn chạy lại từ ẢNH GỐC, không cộng dồn kết quả cũ.
  - Ảnh màu: tương phản xử lý trên kênh I của HSI (Ch.3 tr.53) để giữ sắc độ.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import numpy as np

from core.denoise import (
    median_filter, arithmetic_mean_filter, geometric_mean_filter,
    harmonic_mean_filter, contraharmonic_mean_filter, max_filter, min_filter,
    midpoint_filter, alpha_trimmed_mean_filter, k_nearest_mean_filter,
    gaussian_filter
)
from core.scratch import detect_scratch_mask, fill_masked
from core.intensity import (
    gamma_transform, log_transform, negative_transform, piecewise_linear_transform,
    auto_contrast_stretch, percentile_range
)
from core.histogram import equalize_histogram
from core.sharpen import laplacian_sharpen, combined_sharpen
from core.color import process_color_on_intensity


STAGE_ORDER = ["scratch", "denoise", "contrast", "sharpen"]

STAGE_TITLES = {
    "scratch": "Vết xước",
    "denoise": "Khử nhiễu",
    "contrast": "Sáng & tương phản",
    "sharpen": "Làm nét",
}

# Khóa phương pháp chuẩn (giao diện truyền đúng các khóa này)
DENOISE_METHODS = [
    "median", "mean", "gaussian", "alpha_trimmed", "k_nearest", "geometric",
    "harmonic", "contraharmonic", "midpoint", "max", "min",
]
CONTRAST_METHODS = ["stretch", "equalize", "gamma", "log", "negative"]
SHARPEN_METHODS = ["combined", "laplacian_4", "laplacian_8"]


@dataclass
class Stage:
    name: str          # 'scratch', 'denoise', 'contrast', 'sharpen'
    enabled: bool = True
    method: str = "default"
    params: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Chuẩn hóa tên phương pháp (chấp nhận cả khóa chuẩn lẫn tên mô tả cũ)
# ---------------------------------------------------------------------------
def _norm_denoise(method: str) -> str:
    m = (method or "").lower().strip()
    if m in DENOISE_METHODS:
        return m
    rules = [
        (("contra", "điều hòa ngược"), "contraharmonic"),
        (("harmonic", "điều hòa"), "harmonic"),
        (("geometric", "hình học"), "geometric"),
        (("alpha", "cắt tỉa"), "alpha_trimmed"),
        (("k_nearest", "k-nearest", "gần nhất"), "k_nearest"),
        (("gauss",), "gaussian"),
        (("midpoint", "điểm giữa"), "midpoint"),
        (("median", "trung vị"), "median"),
        (("max", "cực đại"), "max"),
        (("min", "cực tiểu"), "min"),
        (("mean", "số học", "arithmetic"), "mean"),
    ]
    for keys, val in rules:
        if any(k in m for k in keys):
            return val
    return "median"


def _norm_contrast(method: str) -> str:
    m = (method or "").lower().strip()
    if m in CONTRAST_METHODS:
        return m
    if "equal" in m or "cân bằng" in m:
        return "equalize"
    if "stretch" in m or "kéo giãn" in m or "piecewise" in m or "từng đoạn" in m:
        return "stretch"
    if "log" in m:
        return "log"
    if "neg" in m or "âm bản" in m:
        return "negative"
    return "gamma"


def _norm_sharpen(method: str) -> str:
    m = (method or "").lower().strip()
    if m in SHARPEN_METHODS:
        return m
    if "combined" in m or "kết hợp" in m:
        return "combined"
    if "4" in m:
        return "laplacian_4"
    return "laplacian_8"


# ---------------------------------------------------------------------------
# Từng bước
# ---------------------------------------------------------------------------
def apply_scratch(img: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    mask = detect_scratch_mask(
        img,
        sensitivity=params.get("sensitivity", 0.5),
        polarity=params.get("polarity", "bright"),
    )
    if np.any(mask > 0):
        img = fill_masked(img, mask, method=params.get("fill_method", "median"),
                          kernel_size=params.get("kernel_size", 5))
    return img, mask


def apply_denoise(img: np.ndarray, method: str, params: Dict[str, Any]) -> np.ndarray:
    m = _norm_denoise(method)
    k = int(params.get("kernel_size", 3))
    pad = params.get("pad_mode", "mirror")
    if m == "median":
        return median_filter(img, kernel_size=k, pad_mode=pad)
    if m == "mean":
        return arithmetic_mean_filter(img, kernel_size=k, pad_mode=pad)
    if m == "gaussian":
        return gaussian_filter(img, kernel_size=k, sigma=params.get("sigma", k / 3.0), pad_mode=pad)
    if m == "alpha_trimmed":
        d = int(params.get("d", max(2, (k * k) // 3)))
        d = min(d, k * k - 1)
        return alpha_trimmed_mean_filter(img, kernel_size=k, d=d, pad_mode=pad)
    if m == "k_nearest":
        return k_nearest_mean_filter(img, kernel_size=k, k=int(params.get("k", max(3, (k * k) // 2))), pad_mode=pad)
    if m == "geometric":
        return geometric_mean_filter(img, kernel_size=k, pad_mode=pad)
    if m == "harmonic":
        return harmonic_mean_filter(img, kernel_size=k, pad_mode=pad)
    if m == "contraharmonic":
        return contraharmonic_mean_filter(img, kernel_size=k, Q=params.get("Q", 1.5), pad_mode=pad)
    if m == "midpoint":
        return midpoint_filter(img, kernel_size=k, pad_mode=pad)
    if m == "max":
        return max_filter(img, kernel_size=k, pad_mode=pad)
    if m == "min":
        return min_filter(img, kernel_size=k, pad_mode=pad)
    return median_filter(img, kernel_size=k, pad_mode=pad)


def apply_contrast(img: np.ndarray, method: str, params: Dict[str, Any]) -> np.ndarray:
    m = _norm_contrast(method)

    def _on_channel(ch: np.ndarray) -> np.ndarray:
        if m == "equalize":
            return equalize_histogram(ch, method=params.get("eq_mode", "stretch_to_zero"))
        if m == "stretch":
            if "r1" in params and "r2" in params:
                return piecewise_linear_transform(ch, params["r1"], params.get("s1", 0),
                                                  params["r2"], params.get("s2", 255))
            return auto_contrast_stretch(ch, clip_percent=params.get("clip", 1.0))
        if m == "log":
            return log_transform(ch)
        if m == "negative":
            return negative_transform(ch)
        return gamma_transform(ch, gamma=params.get("gamma", 1.0))

    if img.ndim == 3:
        return process_color_on_intensity(img, _on_channel)
    return _on_channel(img)


def apply_sharpen(img: np.ndarray, method: str, params: Dict[str, Any]) -> np.ndarray:
    m = _norm_sharpen(method)
    k = float(params.get("k", 0.4))
    if m == "combined":
        return combined_sharpen(img, k_lap=k, gamma_val=params.get("gamma_val", 1.0))
    mode = "4-neighbor" if m == "laplacian_4" else "8-neighbor"
    return laplacian_sharpen(img, k=k, mode=mode, pad_mode=params.get("pad_mode", "mirror"))


def run_pipeline(orig_image: np.ndarray, stages: List[Stage], return_steps: bool = False):
    """
    Chạy các bước theo thứ tự cố định STAGE_ORDER, luôn bắt đầu từ ảnh gốc.

    Trả về:
        - return_steps=False: ảnh kết quả.
        - return_steps=True: (ảnh kết quả, danh sách [(tên bước, ảnh sau bước)], mặt nạ vết xước hoặc None).
    """
    if orig_image is None or orig_image.size == 0:
        return (orig_image, [], None) if return_steps else orig_image

    img = orig_image.copy()
    stage_map = {s.name: s for s in stages}
    steps = []
    scratch_mask = None

    for name in STAGE_ORDER:
        st = stage_map.get(name)
        if st is None or not st.enabled:
            continue
        if name == "scratch":
            img, scratch_mask = apply_scratch(img, st.params)
        elif name == "denoise":
            img = apply_denoise(img, st.method, st.params)
        elif name == "contrast":
            img = apply_contrast(img, st.method, st.params)
        elif name == "sharpen":
            img = apply_sharpen(img, st.method, st.params)
        if return_steps:
            steps.append((name, img.copy()))

    if return_steps:
        return img, steps, scratch_mask
    return img


# ---------------------------------------------------------------------------
# Chẩn đoán tự động
# ---------------------------------------------------------------------------
def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        g = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
        return np.clip(np.round(g), 0, 255).astype(np.uint8)
    return image


def analyze_image(image: np.ndarray) -> Dict[str, float]:
    """
    Đo các đặc trưng tổn hại của ảnh (dùng cho chẩn đoán và cho phần phân tích dữ liệu trong báo cáo).

    - impulse_ratio: tỉ lệ điểm cực trị (<=5 hoặc >=250) lệch mạnh (>50) so với trung vị 3x3
                     -> dấu hiệu nhiễu xung muối tiêu (Ch.2 tr.131).
    - noise_sigma:   ước lượng độ lệch chuẩn nhiễu hạt = 1.4826 * MAD(ảnh - trung vị 3x3),
                     chỉ tính trên các điểm không phải nhiễu xung.
    - scratch_ratio: tỉ lệ điểm thuộc các vệt dạng đường (vết xước/nứt) trong mặt nạ.
    - p1, p99, dynamic_range, mean: phân vị 1%/99%, độ rộng dải xám và độ sáng trung bình
                     đo trên ảnh đã lọc trung vị (để nhiễu không làm sai lệch).
    """
    gray = _to_gray(image)
    med3 = median_filter(gray, kernel_size=3)
    resid = gray.astype(np.float32) - med3.astype(np.float32)
    extreme = (gray <= 5) | (gray >= 250)
    impulse = extreme & (np.abs(resid) > 50)
    impulse_ratio = float(np.mean(impulse))

    r = resid[~impulse]
    if r.size > 0:
        noise_sigma = float(1.4826 * np.median(np.abs(r - np.median(r))))
    else:
        noise_sigma = 0.0

    # Chỉ đếm thành phần dạng đường (max_speck=0) để đốm nhiễu muối tiêu không bị tính là vết xước
    mask = detect_scratch_mask(image, sensitivity=0.5, max_speck=0)
    scratch_ratio = float(np.mean(mask > 0))

    p1, p99 = percentile_range(med3, 1.0)
    return {
        "impulse_ratio": impulse_ratio,
        "noise_sigma": noise_sigma,
        "scratch_ratio": scratch_ratio,
        "p1": float(p1),
        "p99": float(p99),
        "dynamic_range": float(p99 - p1),
        "mean": float(np.mean(med3)),
    }


# Ngưỡng của luật chẩn đoán (đặt thành hằng số để dễ trình bày và thử nghiệm)
TH_IMPULSE = 0.003        # > 0.3% điểm là nhiễu xung -> lọc trung vị
TH_IMPULSE_HEAVY = 0.12   # > 12% -> trung vị 5x5
TH_NOISE_SIGMA = 6.0      # sigma nhiễu hạt > 6 mức xám -> lọc Gauss / alpha-trimmed
TH_SCRATCH = 0.002        # > 0.2% điểm thuộc vệt xước dạng đường -> bật bước vết xước
TH_RANGE_LOW = 70         # dải xám (p99 - p1) < 70 -> cân bằng histogram
TH_RANGE_OK = 180         # 70..180 -> kéo giãn tương phản
TH_DARK = 80              # dải đủ rộng nhưng quá tối -> gamma < 1
TH_BRIGHT = 175           # dải đủ rộng nhưng quá sáng -> gamma > 1


def diagnose_with_report(image: np.ndarray) -> Tuple[List[Stage], List[str], Dict[str, float]]:
    """
    Chẩn đoán tổn hại và đề xuất cấu hình 4 bước, kèm lời giải thích cho từng quyết định.
    Trả về (stages, report, stats).
    """
    stats = analyze_image(image)
    report: List[str] = []

    # 1. Vết xước
    sc = stats["scratch_ratio"]
    scratch_on = sc > TH_SCRATCH
    scratch = Stage("scratch", scratch_on, "median_fill",
                    {"sensitivity": 0.5, "polarity": "bright", "fill_method": "median", "kernel_size": 5})
    report.append(
        f"Vết xước: {sc * 100:.2f}% điểm ảnh "
        + ("> 0.2% → bật phát hiện và điền vết." if scratch_on else "≤ 0.2% → bỏ qua.")
    )

    # 2. Khử nhiễu
    imp, sig = stats["impulse_ratio"], stats["noise_sigma"]
    if imp > TH_IMPULSE and sig > TH_NOISE_SIGMA:
        denoise = Stage("denoise", True, "alpha_trimmed", {"kernel_size": 3, "d": 4, "pad_mode": "mirror"})
        reason = (f"nhiễu xung {imp * 100:.2f}% và nhiễu hạt σ≈{sig:.1f} → lọc alpha-trimmed 3x3 "
                  f"(bỏ 2 giá trị lớn nhất/nhỏ nhất rồi lấy trung bình).")
    elif imp > TH_IMPULSE:
        k = 5 if imp > TH_IMPULSE_HEAVY else 3
        denoise = Stage("denoise", True, "median", {"kernel_size": k, "pad_mode": "mirror"})
        reason = f"nhiễu xung {imp * 100:.2f}% > 0.3% → lọc trung vị {k}x{k}."
    elif sig > TH_NOISE_SIGMA:
        denoise = Stage("denoise", True, "gaussian", {"kernel_size": 5, "sigma": 1.0, "pad_mode": "mirror"})
        reason = f"nhiễu hạt σ≈{sig:.1f} > 6 → lọc Gauss 5x5 (σ=1.0)."
    else:
        denoise = Stage("denoise", False, "median", {"kernel_size": 3, "pad_mode": "mirror"})
        reason = f"nhiễu xung {imp * 100:.2f}%, nhiễu hạt σ≈{sig:.1f} đều thấp → bỏ qua."
    report.append("Khử nhiễu: " + reason)

    # 3. Sáng & tương phản
    rng, mean = stats["dynamic_range"], stats["mean"]
    if rng < TH_RANGE_LOW:
        contrast = Stage("contrast", True, "equalize", {"eq_mode": "stretch_to_zero"})
        reason = f"dải xám chỉ {rng:.0f} mức (< 70) → cân bằng histogram."
    elif rng < TH_RANGE_OK:
        contrast = Stage("contrast", True, "stretch", {"clip": 1.0})
        reason = (f"dải xám {stats['p1']:.0f}–{stats['p99']:.0f} ({rng:.0f} mức) bị hẹp/bạc màu "
                  f"→ kéo giãn tuyến tính về 0–255.")
    elif mean < TH_DARK:
        contrast = Stage("contrast", True, "gamma", {"gamma": 0.75})
        reason = f"dải xám đủ rộng nhưng ảnh tối (trung bình {mean:.0f}) → gamma 0.75."
    elif mean > TH_BRIGHT:
        contrast = Stage("contrast", True, "gamma", {"gamma": 1.3})
        reason = f"dải xám đủ rộng nhưng ảnh sáng (trung bình {mean:.0f}) → gamma 1.3."
    else:
        contrast = Stage("contrast", False, "stretch", {"clip": 1.0})
        reason = f"dải xám {rng:.0f} mức, trung bình {mean:.0f} đã ổn → bỏ qua."
    report.append("Tương phản: " + reason)

    # 4. Làm nét: nhẹ, chỉ ở vùng biên (Laplacian có trọng số Sobel).
    #    Ảnh nhiều nhiễu hạt thì không làm nét vì đạo hàm khuếch đại phần nhiễu còn sót (Ch.6 tr.29-31).
    if sig > TH_NOISE_SIGMA:
        sharpen = Stage("sharpen", False, "combined", {"k": 0.3})
        report.append(f"Làm nét: bỏ qua vì ảnh còn nhiễu hạt (σ≈{sig:.1f}), đạo hàm sẽ khuếch đại nhiễu.")
    else:
        sharpen = Stage("sharpen", True, "combined", {"k": 0.3})
        report.append("Làm nét: Laplacian có trọng số Sobel, k = 0.3 (chỉ tăng nét ở biên).")

    return [scratch, denoise, contrast, sharpen], report, stats


def diagnose(image: np.ndarray) -> List[Stage]:
    """Giữ tương thích: chỉ trả về danh sách bước (xem diagnose_with_report để có lời giải thích)."""
    if image is None or image.size == 0:
        return [
            Stage("scratch", False, "median_fill", {"sensitivity": 0.5}),
            Stage("denoise", False, "median", {"kernel_size": 3}),
            Stage("contrast", False, "stretch", {"clip": 1.0}),
            Stage("sharpen", False, "combined", {"k": 0.4}),
        ]
    stages, _, _ = diagnose_with_report(image)
    return stages
