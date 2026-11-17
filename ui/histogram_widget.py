"""
Module ui/histogram_widget.py
Widget hiển thị biểu đồ phân bố mức xám Histogram tinh gọn bằng QPainter.
Khớp hoàn hảo với phong cách thiết kế trong ảnh mock-up.
"""
import numpy as np
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QColor, QFont, QPen, QBrush
from PySide6.QtCore import Qt

from core.histogram import compute_histogram, compute_stats


class HistogramWidget(QWidget):
    """Widget vẽ biểu đồ cột tần suất Histogram."""
    
    def __init__(self, title: str = "Histogram", bar_color: QColor = QColor(26, 86, 160), parent=None):
        super().__init__(parent)
        self.title = title
        self.bar_color = bar_color
        self.hist_data = np.zeros(256, dtype=int)
        self.mean_val = 0.0
        self.std_val = 0.0
        self.setMinimumHeight(110)
        self.setMaximumHeight(140)
        
    def set_image(self, image: np.ndarray):
        """Cập nhật dữ liệu từ ảnh mới."""
        if image is None or image.size == 0:
            self.hist_data = np.zeros(256, dtype=int)
            self.mean_val = 0.0
            self.std_val = 0.0
        else:
            h, _ = compute_histogram(image)
            self.hist_data = h
            self.mean_val, self.std_val = compute_stats(image)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        # 1. Tiêu đề và thông số
        font_title = QFont("Segoe UI", 9)
        painter.setFont(font_title)
        painter.setPen(QColor(80, 90, 105))
        painter.drawText(6, 16, self.title)
        
        if np.max(self.hist_data) > 0:
            stat_str = f"Mean: {self.mean_val:.1f} | Std: {self.std_val:.1f}"
            font_stat = QFont("Consolas", 8)
            painter.setFont(font_stat)
            painter.setPen(QColor(120, 130, 145))
            txt_w = painter.fontMetrics().horizontalAdvance(stat_str)
            painter.drawText(w - txt_w - 6, 16, stat_str)

        # 2. Khu vực vẽ các cột Histogram
        chart_top = 26
        chart_bottom = h - 18
        chart_h = max(10, chart_bottom - chart_top)
        chart_left = 6
        chart_right = w - 6
        chart_w = max(10, chart_right - chart_left)

        # Nhóm 256 bin thành 32 cột để hiển thị các thanh rộng, đẹp mắt như mock-up
        num_bars = 32
        bins_per_bar = 256 // num_bars
        aggregated = np.zeros(num_bars, dtype=float)
        for i in range(num_bars):
            aggregated[i] = np.sum(self.hist_data[i * bins_per_bar : (i + 1) * bins_per_bar])

        max_val = np.max(aggregated) if np.max(aggregated) > 0 else 1.0

        bar_width = chart_w / float(num_bars)
        gap = 2.0
        actual_bar_w = max(1.0, bar_width - gap)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(self.bar_color))

        for i in range(num_bars):
            val = aggregated[i]
            bar_h = (val / max_val) * chart_h
            bx = chart_left + i * bar_width
            by = chart_bottom - bar_h
            painter.drawRoundedRect(bx, by, actual_bar_w, bar_h, 1.5, 1.5)

        # 3. Đường kẻ đáy và nhãn mức xám
        painter.setPen(QPen(QColor(220, 226, 235), 1))
        painter.drawLine(chart_left, chart_bottom, chart_right, chart_bottom)

        painter.setFont(QFont("Consolas", 7))
        painter.setPen(QColor(140, 150, 165))
        painter.drawText(chart_left, h - 4, "0")
        painter.drawText(chart_left + chart_w // 2 - 8, h - 4, "128")
        painter.drawText(chart_right - 18, h - 4, "255")
