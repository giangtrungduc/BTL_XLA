"""
Module ui/histogram_widget.py
Vẽ histogram mức xám bằng QPainter (64 cột), kèm mean / std và vạch phân vị 1% - 99%.
"""
import numpy as np
from PySide6.QtWidgets import QWidget
from PySide6.QtGui import QPainter, QColor, QFont, QPen, QBrush
from PySide6.QtCore import Qt, QRectF

from core.histogram import compute_histogram, compute_stats


class HistogramWidget(QWidget):
    def __init__(self, title: str = "Histogram", bar_color: QColor = QColor(37, 99, 235), parent=None):
        super().__init__(parent)
        self.title = title
        self.bar_color = bar_color
        self.hist = np.zeros(256, dtype=np.int64)
        self.mean_val = 0.0
        self.std_val = 0.0
        self.p1 = None
        self.p99 = None
        self.setMinimumHeight(96)
        self.setMaximumHeight(120)

    def set_image(self, image):
        if image is None or image.size == 0:
            self.hist = np.zeros(256, dtype=np.int64)
            self.mean_val = self.std_val = 0.0
            self.p1 = self.p99 = None
        else:
            h, _ = compute_histogram(image)
            self.hist = h
            self.mean_val, self.std_val = compute_stats(image)
            cdf = np.cumsum(h) / max(1, h.sum())
            self.p1 = int(np.searchsorted(cdf, 0.01))
            self.p99 = int(np.searchsorted(cdf, 0.99))
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        p.setFont(QFont("Segoe UI", 8))
        p.setPen(QColor(100, 116, 139))
        p.drawText(2, 12, self.title)
        if self.hist.sum() > 0:
            txt = f"mean {self.mean_val:.0f}   std {self.std_val:.0f}   dải 1–99%: {self.p1}–{self.p99}"
            tw = p.fontMetrics().horizontalAdvance(txt)
            p.drawText(w - tw - 2, 12, txt)

        top, bottom, left, right = 20, h - 14, 2, w - 2
        cw, ch = max(10, right - left), max(10, bottom - top)

        # Nền gradient đen -> trắng nhỏ dưới trục để thấy mức xám
        for i in range(0, cw, 2):
            g = int(255 * i / cw)
            p.setPen(QColor(g, g, g))
            p.drawLine(left + i, bottom + 3, left + i, bottom + 6)

        n_bars = 64
        agg = self.hist.reshape(n_bars, -1).sum(axis=1).astype(float)
        mx = agg.max() if agg.max() > 0 else 1.0
        bw = cw / n_bars
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(self.bar_color))
        for i, v in enumerate(agg):
            bh = (v / mx) * ch
            p.drawRect(QRectF(left + i * bw, bottom - bh, max(1.0, bw - 1), bh))

        p.setPen(QPen(QColor(226, 232, 240), 1))
        p.drawLine(left, bottom, right, bottom)

        if self.p1 is not None:
            pen = QPen(QColor(220, 38, 38, 160), 1, Qt.DashLine)
            p.setPen(pen)
            for v in (self.p1, self.p99):
                x = left + v / 255.0 * cw
                p.drawLine(int(x), top, int(x), bottom)
        p.end()
