"""
Module ui/image_pair.py
Khung hiển thị trực quan 2 ảnh song song (Before / After) cùng 2 biểu đồ Histogram tương ứng.
Khớp hoàn hảo với bố cục trong ảnh mock-up thiết kế.
"""
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame
)
from PySide6.QtGui import QPixmap, QImage, QColor
from PySide6.QtCore import Qt

from ui.histogram_widget import HistogramWidget


class ImageCard(QFrame):
    """Khung thẻ hiển thị 1 ảnh cùng biểu đồ Histogram tương ứng."""
    
    def __init__(self, title: str, is_result: bool = False, parent=None):
        super().__init__(parent)
        self.is_result = is_result
        self.current_image = None
        
        self.setStyleSheet("""
            ImageCard {
                background-color: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 8px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        # 1. Header thẻ
        header = QHBoxLayout()
        self.lbl_title = QLabel(title)
        self.lbl_title.setStyleSheet("font-weight: bold; font-size: 12px; color: #1e293b;")
        header.addWidget(self.lbl_title)
        header.addStretch()

        if self.is_result:
            self.badge = QLabel("Chưa xử lý")
            self.badge.setStyleSheet("""
                background-color: #e0f2fe;
                color: #0369a1;
                border-radius: 10px;
                padding: 2px 8px;
                font-size: 10px;
                font-weight: 500;
            """)
            header.addWidget(self.badge)
        else:
            self.lbl_info = QLabel("Chưa nạp ảnh")
            self.lbl_info.setStyleSheet("color: #64748b; font-size: 10px; font-family: Consolas;")
            header.addWidget(self.lbl_info)

        layout.addLayout(header)

        # 2. Khung Viewport hiển thị ảnh
        self.img_view = QLabel()
        self.img_view.setAlignment(Qt.AlignCenter)
        self.img_view.setMinimumSize(220, 240)
        self.img_view.setStyleSheet("""
            background-color: #f1f5f9;
            border: 1px dashed #cbd5e1;
            border-radius: 6px;
        """)
        layout.addWidget(self.img_view, 1)

        # 3. Biểu đồ Histogram bên dưới
        hist_title = "Histogram kết quả" if is_result else "Histogram gốc"
        bar_color = QColor(14, 116, 144) if is_result else QColor(30, 64, 175)
        self.hist_widget = HistogramWidget(title=hist_title, bar_color=bar_color)
        layout.addWidget(self.hist_widget)

    def set_image(self, image: np.ndarray, info_text: str = None):
        """Cập nhật ảnh hiển thị và cập nhật histogram."""
        self.current_image = image
        if image is None:
            self.img_view.clear()
            self.hist_widget.set_image(None)
            return

        self._render_pixmap()
        self.hist_widget.set_image(image)

        if not self.is_result and info_text:
            self.lbl_info.setText(info_text)

    def set_badge_text(self, text: str):
        """Cập nhật badge trạng thái các bước đã chạy."""
        if self.is_result:
            self.badge.setText(text)

    def _render_pixmap(self):
        if self.current_image is None:
            return

        img_arr = np.ascontiguousarray(self.current_image)
        h, w = img_arr.shape[:2]
        
        if img_arr.ndim == 2:
            q_img = QImage(img_arr.data, w, h, w, QImage.Format_Grayscale8)
        else:
            q_img = QImage(img_arr.data, w, h, 3 * w, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_img)
        # Scale vừa vặn kích thước hiển thị mà giữ nguyên tỷ lệ khung hình
        target_size = self.img_view.size()
        if target_size.width() > 10 and target_size.height() > 10:
            scaled = pixmap.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.img_view.setPixmap(scaled)
        else:
            self.img_view.setPixmap(pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render_pixmap()
