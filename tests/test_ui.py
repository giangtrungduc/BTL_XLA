"""
Kiểm thử giao diện ở chế độ không màn hình (offscreen):
nạp ảnh, khôi phục tự động, đổi phương pháp ở mọi thẻ, xem từng bước, chỉ số trước/sau, lưu ảnh.
"""
import os
import sys
import tempfile

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QEventLoop, QTimer

from ui.main_window import MainWindow, SAMPLES, DATA_DIR, load_image_file


def _wait(win, ms=150):
    """Chờ bộ hẹn giờ debounce và luồng nền xử lý xong, rồi xử lý tín hiệu trả về."""
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()
    win.worker.wait()
    QApplication.processEvents()


def test_ui_workflow():
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow(autoload_sample=False)
    win.show()

    # 1. Ảnh thật (màu), không có ảnh đối chiếu
    assert win.load_image_from_path(os.path.join(DATA_DIR, SAMPLES[0][1]))
    _wait(win)
    assert win.result_image is not None and win.result_image.shape == win.preview_image.shape
    print(f"✓ Nạp ảnh thật {win.full_image.shape}")

    win._on_auto_restore()
    _wait(win)
    enabled = [c.get_stage().name for c in win.cards.values() if c.get_stage().enabled]
    assert "scratch" in enabled and "sharpen" in enabled, enabled
    assert "Chẩn đoán" in win.lbl_report.text()
    assert win.m_psnr.val.text() == "—"
    print(f"✓ Khôi phục tự động trên ảnh thật, các bước bật: {enabled}")

    # 2. Ảnh mô phỏng có ảnh sạch: PSNR phải tăng
    _, rel, ref = SAMPLES[1]
    win._load_sample(rel, ref)
    _wait(win)
    win._on_auto_restore()
    _wait(win)
    before, after = [float(x) for x in win.m_psnr.val.text().split("→")]
    assert after > before, (before, after)
    print(f"✓ Ảnh mô phỏng: PSNR {before:.2f} → {after:.2f} dB")

    # 3. Xem từng bước và mặt nạ
    views = [win.card_result.cbo_view.itemData(i) for i in range(win.card_result.cbo_view.count())]
    assert "mask" in views and "step:0" in views, views
    for v in views:
        win.card_result.cbo_view.setCurrentIndex(win.card_result.cbo_view.findData(v))
        QApplication.processEvents()
    print(f"✓ Xem được {len(views)} chế độ hiển thị")

    # 4. Đổi qua mọi phương pháp ở mọi thẻ
    for card in win.cards.values():
        card.chk_enable.setChecked(True)
        combos = [getattr(card, n) for n in ("cbo_method", "cbo_ksize", "cbo_polarity", "cbo_fill") if hasattr(card, n)]
        for cbo in combos:
            for i in range(cbo.count()):
                cbo.setCurrentIndex(i)
                card.sl_main.slider.setValue(card.sl_main.slider.maximum())
                _wait(win, 80)
                assert "Lỗi" not in win.lbl_status.text(), (card.stage_name, cbo.itemData(i), win.lbl_status.text())
    print("✓ Mọi phương pháp ở 4 thẻ chạy không lỗi")

    # 5. Đặt lại -> kết quả trùng ảnh gốc
    win._on_reset()
    _wait(win)
    assert np.array_equal(win.result_image, win.preview_image)
    print("✓ Đặt lại trả về ảnh gốc")

    # 6. Tạo ảnh mô phỏng từ ảnh hiện tại
    win._on_simulate()
    _wait(win)
    assert win.preview_reference is not None
    print("✓ Tạo ảnh cũ mô phỏng")

    # 7. Lưu ở độ phân giải gốc
    win._on_auto_restore()
    _wait(win)
    out_path = os.path.join(tempfile.mkdtemp(), "ket_qua.png")
    from core.pipeline import run_pipeline
    from PIL import Image
    Image.fromarray(run_pipeline(win.full_image, win._collect_stages())).save(out_path)
    saved = load_image_file(out_path)
    assert saved.shape[:2] == win.full_image.shape[:2]
    print("✓ Lưu ảnh kết quả")

    win.close()
    print("\n===> GIAO DIỆN HOẠT ĐỘNG ĐÚNG <===")


if __name__ == "__main__":
    test_ui_workflow()
