"""
Điểm khởi chạy ứng dụng: Khôi phục và nâng cao chất lượng ảnh cũ.
Học phần: Xử lý ảnh (INT13146) — Học viện Công nghệ Bưu chính Viễn thông (PTIT).

Chạy:  python main.py
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont

from ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Khôi phục ảnh cũ")
    app.setOrganizationName("PTIT")
    app.setFont(QFont("Segoe UI", 10))

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
