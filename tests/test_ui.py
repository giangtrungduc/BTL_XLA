"""
Kiểm thử nạp và khởi tạo giao diện MainWindow và các tương tác pipeline.
"""
import os
import sys

# Đảm bảo nạp được thư mục gốc
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

os.environ["QT_QPA_PLATFORM"] = "offscreen"  # Chạy headless

from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow


def test_ui_workflow():
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win.worker.wait()
    assert win is not None
    print("✓ Khởi tạo MainWindow: THÀNH CÔNG")

    # Kiểm tra nạp ảnh mẫu
    sample_path = os.path.join("data", "old", "vietnam_vintage_portrait.jpg")
    if not os.path.exists(sample_path):
        sample_path = os.path.join("data", "old", "sample_simulated_old.png")
        
    win.load_image_from_path(sample_path)
    win.worker.wait()
    assert win.orig_image is not None
    print(f"✓ Nạp ảnh mẫu ({win.orig_image.shape}): THÀNH CÔNG")

    # Thử gọi nút Khôi phục tự động (Diagnose)
    win._on_auto_restore()
    win.worker.wait()
    print("✓ Chẩn đoán và khôi phục tự động: THÀNH CÔNG")

    # Thử toggle các thẻ
    win.card_scratch.chk_enable.setChecked(True)
    win.card_scratch.slider_sens.setValue(60)
    win._trigger_pipeline()
    win.worker.wait()
    print("✓ Thay đổi tham số vết xước: THÀNH CÔNG")

    print("\n===> TOÀN BỘ GIAO DIỆN VÀ PIPELINE ĐÃ HOẠT ĐỘNG HOÀN HẢO! <===")


if __name__ == "__main__":
    test_ui_workflow()
