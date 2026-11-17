"""
Module core/metrics.py
Các tiêu chí đánh giá định lượng chất lượng ảnh (Chương 4, slide 68-70).
Toàn bộ thuật toán được cài đặt trực tiếp bằng NumPy (Code from scratch).

Bao gồm:
  1. Sai số căn quân phương (Root Mean Square Error - e_rms) - Slide tr.68
  2. Tỷ số tín hiệu trên nhiễu bình phương trung bình (SNR_ms) - Slide tr.69
  3. Tỷ số tín hiệu cực đại trên nhiễu (PSNR)
"""
import numpy as np


def calculate_rms_error(f_hat: np.ndarray, f: np.ndarray) -> float:
    """
    Tính sai số căn quân phương (Root Mean Square Error - e_rms) - Chương 4, slide 68.
    
    Công thức:
        e_rms = sqrt( (1 / (M * N)) * sum_{x=0}^{M-1} sum_{y=0}^{N-1} [f_hat(x, y) - f(x, y)]^2 )
        
    Ý nghĩa:
        Đo độ lệch trung bình giữa hai ảnh. Khi đối chứng với OpenCV,
        e_rms = 0 hoặc xấp xỉ 0 chứng minh thuật toán tự viết bằng NumPy hoàn toàn đúng đắn.
    """
    f_hat_f = f_hat.astype(np.float64)
    f_f = f.astype(np.float64)
    mse = np.mean(np.square(f_hat_f - f_f))
    return float(np.sqrt(mse))


def calculate_snr_ms(f_hat: np.ndarray, f: np.ndarray) -> float:
    """
    Tính tỷ số tín hiệu trên nhiễu bình phương trung bình (SNR_ms) - Chương 4, slide 69.
    
    Công thức:
        SNR_ms = sum(f_hat^2) / sum((f_hat - f)^2)
        
    Trả về:
        Giá trị tỷ số theo thang Decibel (dB): 10 * log10(SNR_ms).
    """
    f_hat_f = f_hat.astype(np.float64)
    f_f = f.astype(np.float64)
    
    signal_power = np.sum(np.square(f_hat_f))
    noise_power = np.sum(np.square(f_hat_f - f_f))
    
    if noise_power == 0:
        return float("inf")  # Hai ảnh trùng khớp hoàn toàn
    if signal_power == 0:
        return 0.0
        
    snr_ratio = signal_power / noise_power
    return float(10.0 * np.log10(snr_ratio))


def calculate_psnr(f_hat: np.ndarray, f: np.ndarray, max_val: float = 255.0) -> float:
    """
    Tính tỷ số tín hiệu cực đại trên nhiễu (PSNR - Peak Signal-to-Noise Ratio).
    
    Công thức:
        PSNR = 20 * log10( max_val / e_rms )
    """
    rms = calculate_rms_error(f_hat, f)
    if rms == 0:
        return float("inf")
    return float(20.0 * np.log10(max_val / rms))
