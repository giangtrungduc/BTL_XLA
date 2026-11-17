"""
Điểm khởi chạy ứng dụng Desktop: Khôi phục và nâng cao chất lượng ảnh cũ (restore_app).
Học phần: Xử lý ảnh (INT13146) — Học viện Công nghệ Bưu chính Viễn thông (PTIT).
"""
import sys
import os

# Thêm thư mục gốc vào PYTHONPATH
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow


def main():
    # Hỗ trợ tỷ lệ hiển thị cho màn hình độ nét cao (High-DPI)
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"

    app = QApplication(sys.argv)
    app.setApplicationName("OldPhotoRestorer_PTIT")
    app.setOrganizationName("PTIT_FIT1")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
