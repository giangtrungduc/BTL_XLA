"""
Unit test kiểm thử toàn diện các module của core/
"""
import os
import sys

# Đảm bảo nạp được thư mục gốc
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import core


def test_padding():
    img = np.ones((10, 10), dtype=np.uint8)
    p_mirror = core.pad_image(img, 2, 2, mode="mirror")
    assert p_mirror.shape == (14, 14)
    p_rep = core.pad_image(img, 2, 2, mode="replicate")
    assert p_rep.shape == (14, 14)
    p_zero = core.pad_image(img, 2, 2, mode="zero")
    assert p_zero.shape == (14, 14)
    assert p_zero[0, 0] == 0
    print("✓ test_padding: PASSED")


def test_filters():
    np.random.seed(42)
    img = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
    
    # 1. Median
    m = core.median_filter(img, 3)
    assert m.shape == (32, 32)
    
    # 2. Arithmetic mean
    am = core.arithmetic_mean_filter(img, 3)
    assert am.shape == (32, 32)
    
    # 3. Geometric mean
    gm = core.geometric_mean_filter(img, 3)
    assert gm.shape == (32, 32)
    
    # 4. Harmonic mean
    hm = core.harmonic_mean_filter(img, 3)
    assert hm.shape == (32, 32)
    
    # 5. Contraharmonic mean
    chm = core.contraharmonic_mean_filter(img, 3, Q=1.5)
    assert chm.shape == (32, 32)
    
    # 6. Min/Max/Midpoint
    mn = core.min_filter(img, 3)
    mx = core.max_filter(img, 3)
    mp = core.midpoint_filter(img, 3)
    assert np.all(mn <= mx)
    
    # 7. Alpha-trimmed mean
    atm = core.alpha_trimmed_mean_filter(img, 3, d=2)
    assert atm.shape == (32, 32)
    
    # 8. K-nearest mean
    knm = core.k_nearest_mean_filter(img, 3, k=5)
    assert knm.shape == (32, 32)
    
    # 9. Gaussian
    gf = core.gaussian_filter(img, 3, sigma=1.0)
    assert gf.shape == (32, 32)
    
    print("✓ test_filters: PASSED")


def test_intensity_and_histogram():
    img = np.array([[50, 100], [150, 200]], dtype=np.uint8)
    
    # Gamma
    g = core.gamma_transform(img, gamma=0.5)
    assert g[0, 0] > img[0, 0]
    
    # Log
    lg = core.log_transform(img)
    assert lg.shape == (2, 2)
    
    # Negative
    neg = core.negative_transform(img)
    assert np.all(neg == 255 - img)
    
    # Histogram
    h, p = core.compute_histogram(img)
    assert len(h) == 256
    assert np.isclose(np.sum(p), 1.0)
    
    # Equalize (cả 2 phương pháp)
    eq1 = core.equalize_histogram(img, method="stretch_to_zero")
    eq2 = core.equalize_histogram(img, method="textbook")
    assert eq1.shape == (2, 2)
    assert eq2.shape == (2, 2)
    
    print("✓ test_intensity_and_histogram: PASSED")


def test_color_hsi():
    rgb = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
    h, s, i = core.rgb_to_hsi(rgb)
    assert h.shape == (20, 20)
    assert s.shape == (20, 20)
    assert i.shape == (20, 20)
    
    # Chuyển ngược lại RGB
    rgb_reconstructed = core.hsi_to_rgb(h, s, i)
    diff = np.abs(rgb.astype(int) - rgb_reconstructed.astype(int))
    # Sai lệch chuyển đổi màu do làm tròn số học <= 2 mức xám
    assert np.max(diff) <= 2, f"Sai lệch HSI <-> RGB quá lớn: {np.max(diff)}"
    
    # Test xử lý trên kênh I
    def _eq_proc(gray):
        return core.equalize_histogram(gray)
    res_color = core.process_color_on_intensity(rgb, _eq_proc)
    assert res_color.shape == (20, 20, 3)
    
    print("✓ test_color_hsi: PASSED")


def test_sharpening():
    img = np.random.randint(0, 256, (30, 30), dtype=np.uint8)
    
    # Laplacian 4 & 8
    lap4 = core.laplacian_sharpen(img, k=0.4, mode="4-neighbor")
    lap8 = core.laplacian_sharpen(img, k=0.4, mode="8-neighbor")
    assert lap4.shape == (30, 30)
    assert lap8.shape == (30, 30)
    
    # Sobel
    sob = core.sobel_filter(img)
    assert sob.shape == (30, 30)
    
    # Combined mode
    comb = core.combined_sharpen(img, k_lap=0.4, gamma_val=0.85)
    assert comb.shape == (30, 30)

    # Vùng phẳng (gradient = 0) không bị thay đổi bởi chế độ kết hợp
    flat = np.full((30, 30), 120, dtype=np.uint8)
    assert np.array_equal(core.combined_sharpen(flat, k_lap=1.0), flat)
    
    print("✓ test_sharpening: PASSED")


def test_morphology_and_scratch():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 10] = 255  # 1 vạch dọc 10 pixel
    
    dil = core.dilate(mask, 3)
    assert np.sum(dil > 0) > np.sum(mask > 0)
    
    ero = core.erode(dil, 3)
    assert np.sum(ero > 0) <= np.sum(dil > 0)
    
    filtered, num = core.connected_components(mask, min_size=5)
    assert num == 1

    # fill_masked: điểm xước được lấp bằng giá trị nền, điểm ngoài mặt nạ giữ nguyên
    img = np.full((20, 20), 100, dtype=np.uint8)
    img[0, 0] = 7
    img[mask > 0] = 255
    filled = core.fill_masked(img, mask, method="median", kernel_size=5)
    assert np.all(filled[mask > 0] == 100)
    assert filled[0, 0] == 7

    # Vết rộng hơn cửa sổ lọc vẫn được điền hết (điền từ mép vào)
    wide = np.zeros((40, 40), dtype=np.uint8)
    wide[10:30, 10:30] = 255
    img2 = np.full((40, 40), 80, dtype=np.uint8)
    img2[wide > 0] = 250
    filled2 = core.fill_masked(img2, wide, kernel_size=3)
    assert np.all(filled2[wide > 0] == 80)
    
    print("✓ test_morphology_and_scratch: PASSED")


def test_pipeline_and_diagnose():
    img = np.random.randint(50, 150, (60, 60), dtype=np.uint8)
    stages = core.diagnose(img)
    assert [s.name for s in stages] == ["scratch", "denoise", "contrast", "sharpen"]

    out = core.run_pipeline(img, stages)
    assert out.shape == img.shape
    assert out.dtype == np.uint8

    # Ảnh màu đi qua mọi phương pháp của mọi bước mà không lỗi
    from core.pipeline import Stage, DENOISE_METHODS, CONTRAST_METHODS, SHARPEN_METHODS
    rgb = np.random.randint(0, 256, (40, 50, 3), dtype=np.uint8)
    for m in DENOISE_METHODS:
        for k in (3, 5):
            o = core.run_pipeline(rgb, [Stage("denoise", True, m, {"kernel_size": k})])
            assert o.shape == rgb.shape and o.dtype == np.uint8, m
    for m in CONTRAST_METHODS:
        o = core.run_pipeline(rgb, [Stage("contrast", True, m, {})])
        assert o.shape == rgb.shape, m
    for m in SHARPEN_METHODS:
        o = core.run_pipeline(rgb, [Stage("sharpen", True, m, {"k": 0.5})])
        assert o.shape == rgb.shape, m
    o = core.run_pipeline(rgb, [Stage("scratch", True, "median_fill", {"polarity": "both"})])
    assert o.shape == rgb.shape

    # Không bật bước nào -> kết quả trùng ảnh gốc
    o = core.run_pipeline(rgb, [Stage(n, False) for n in ("scratch", "denoise", "contrast", "sharpen")])
    assert np.array_equal(o, rgb)

    # return_steps trả về ảnh sau từng bước đúng thứ tự
    res, steps, mask = core.run_pipeline(img, [Stage("sharpen", True, "laplacian_4", {"k": 0.3}),
                                               Stage("denoise", True, "median", {"kernel_size": 3})],
                                         return_steps=True)
    assert [n for n, _ in steps] == ["denoise", "sharpen"] and mask is None
    assert np.array_equal(steps[-1][1], res)

    print("✓ test_pipeline_and_diagnose: PASSED")


def test_convolution_matches_window_sum():
    """convolve2d (dịch-rồi-cộng) phải cho đúng kết quả nhân cửa sổ với mặt nạ."""
    from core.window import convolve2d, get_sliding_windows
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (37, 41)).astype(np.uint8)
    kernel = rng.normal(size=(5, 5)).astype(np.float32)
    a = convolve2d(img, kernel, "mirror")
    b = np.sum(get_sliding_windows(img.astype(np.float32), 5, "mirror") * kernel, axis=(-2, -1))
    assert np.allclose(a, b, atol=1e-3)
    print("✓ test_convolution_matches_window_sum: PASSED")


def test_scratch_detection_quality():
    """Trên ảnh mô phỏng có đáp án, mặt nạ phải bắt được phần lớn vết xước và ít bắt nhầm."""
    from core.degrade import simulate_old_photo
    yy, xx = np.mgrid[0:200, 0:200]
    clean = (60 + 120 * (((xx - 100) ** 2 + (yy - 100) ** 2) < 60 ** 2)).astype(np.uint8)
    old, gt = simulate_old_photo(clean, "salt_pepper", 0.01, 1.0, num_scratches=5, seed=3)
    mask = core.detect_scratch_mask(old, 0.5) > 0
    gt = gt > 0
    recall = (mask & gt).sum() / gt.sum()
    precision = (mask & core.dilate(gt.astype(np.uint8) * 255, 3).astype(bool)).sum() / max(1, mask.sum())
    assert recall > 0.85, recall
    assert precision > 0.6, precision
    print(f"✓ test_scratch_detection_quality: PASSED (recall={recall:.2f}, precision={precision:.2f})")


def test_auto_restore_improves_psnr():
    """Khôi phục tự động phải làm ảnh GẦN ảnh sạch hơn (PSNR tăng) với mọi kiểu hư hại mô phỏng."""
    from core.degrade import simulate_old_photo
    from core.pipeline import diagnose_with_report
    sys.path.insert(0, os.path.join(ROOT_DIR, "data"))
    from create_samples import make_clean_portrait
    clean = make_clean_portrait()
    cases = {
        "muối tiêu": dict(noise_type="salt_pepper", noise_level=0.03, gamma_val=1.0, num_scratches=0),
        "nhiễu hạt": dict(noise_type="gaussian", noise_level=0.06, gamma_val=1.0, num_scratches=0),
        "bạc màu": dict(noise_type="gaussian", noise_level=0.0, gamma_val=1.0, num_scratches=0, fade=0.5),
        "vết xước": dict(noise_type="gaussian", noise_level=0.0, gamma_val=1.0, num_scratches=8),
        "tổng hợp": dict(noise_type="both", noise_level=0.02, gamma_val=1.0, num_scratches=6, fade=0.45),
    }
    for name, kw in cases.items():
        old, _ = simulate_old_photo(clean, seed=11, **kw)
        stages, report, _ = diagnose_with_report(old)
        out = core.run_pipeline(old, stages)
        before, after = core.calculate_psnr(old, clean), core.calculate_psnr(out, clean)
        assert after > before + 3.0, f"{name}: {before:.2f} -> {after:.2f}"
        assert len(report) == 4
        print(f"   {name:<10} PSNR {before:6.2f} -> {after:6.2f} dB")
    print("✓ test_auto_restore_improves_psnr: PASSED")


def test_downscale():
    from core.resize import downscale_factor, block_downscale
    img = np.arange(16, dtype=np.uint8).reshape(4, 4)
    out = block_downscale(img, 2)
    assert out.shape == (2, 2) and out[0, 0] == 2  # mean(0,1,4,5)=2.5 -> làm tròn chẵn = 2
    assert downscale_factor((3000, 2000), 900) == 4
    assert downscale_factor((500, 400), 900) == 1
    print("✓ test_downscale: PASSED")


if __name__ == "__main__":
    print("==================================================")
    print("   CHẠY UNIT TEST TOÀN BỘ CÁC MODULE TRONG CORE   ")
    print("==================================================")
    test_padding()
    test_filters()
    test_intensity_and_histogram()
    test_color_hsi()
    test_sharpening()
    test_morphology_and_scratch()
    test_pipeline_and_diagnose()
    test_convolution_matches_window_sum()
    test_scratch_detection_quality()
    test_auto_restore_improves_psnr()
    test_downscale()
    print("==================================================")
    print("   ✓✓✓ 100% TẤT CẢ UNIT TEST TRONG CORE ĐÃ PASS!  ")
    print("==================================================")
