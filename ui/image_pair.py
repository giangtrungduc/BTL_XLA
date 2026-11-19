"""
Module ui/image_pair.py
Thẻ hiển thị một ảnh kèm histogram. Thẻ kết quả có thêm hộp chọn "Hiển thị" để xem
kết quả sau từng bước hoặc mặt nạ vết xước (rất tiện khi thuyết trình).
"""
import numpy as np
from PySide6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QSizePolicy, QWidget
from PySide6.QtGui import QPixmap, QImage, QColor
from PySide6.QtCore import Qt, Signal

from ui.histogram_widget import HistogramWidget


def numpy_to_qpixmap(img: np.ndarray) -> QPixmap:
    """Chuyển mảng uint8 (xám hoặc RGB) sang QPixmap (QImage.copy() để không phụ thuộc bộ nhớ numpy)."""
    arr = np.ascontiguousarray(img)
    h, w = arr.shape[:2]
    if arr.ndim == 2:
        q = QImage(arr.data, w, h, arr.strides[0], QImage.Format_Grayscale8).copy()
    else:
        q = QImage(arr.data, w, h, arr.strides[0], QImage.Format_RGB888).copy()
    return QPixmap.fromImage(q)


class ImageCard(QFrame):
    """Thẻ ảnh + histogram."""
    viewChanged = Signal(str)

    def __init__(self, title: str, is_result: bool = False, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.is_result = is_result
        self._image = None
        self._pixmap = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(8)

        head_w = QWidget()
        head_w.setFixedHeight(32)
        head = QHBoxLayout(head_w)
        head.setContentsMargins(0, 0, 0, 0)
        self.lbl_title = QLabel(title)
        self.lbl_title.setObjectName("cardTitle")
        head.addWidget(self.lbl_title)
        head.addStretch()
        self.lbl_info = QLabel("")
        self.lbl_info.setObjectName("muted")
        head.addWidget(self.lbl_info)
        if is_result:
            self.cbo_view = QComboBox()
            self.cbo_view.setMinimumWidth(170)
            self.cbo_view.addItem("Kết quả cuối", "final")
            self.cbo_view.currentIndexChanged.connect(
                lambda _: self.viewChanged.emit(self.cbo_view.currentData() or "final"))
            head.addWidget(self.cbo_view)
        lay.addWidget(head_w)

        self.view = QLabel("Chưa có ảnh\nBấm “Mở ảnh” hoặc “Ảnh mẫu”")
        self.view.setObjectName("imageView")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumSize(240, 240)
        self.view.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        lay.addWidget(self.view, 1)

        color = QColor(37, 99, 235) if is_result else QColor(71, 85, 105)
        self.hist = HistogramWidget("Histogram", color)
        lay.addWidget(self.hist)

    # ------------------------------------------------------------------
    def set_view_options(self, options):
        """options: danh sách (khóa, nhãn). Giữ lựa chọn hiện tại nếu vẫn còn."""
        if not self.is_result:
            return
        current = self.cbo_view.currentData()
        self.cbo_view.blockSignals(True)
        self.cbo_view.clear()
        for key, label in options:
            self.cbo_view.addItem(label, key)
        idx = self.cbo_view.findData(current)
        self.cbo_view.setCurrentIndex(idx if idx >= 0 else 0)
        self.cbo_view.blockSignals(False)

    def current_view(self) -> str:
        return (self.cbo_view.currentData() or "final") if self.is_result else "final"

    def set_info(self, text: str):
        self.lbl_info.setText(text)

    def set_image(self, image: np.ndarray, update_hist: bool = True):
        self._image = image
        if image is None:
            self._pixmap = None
            self.view.clear()
            self.view.setText("Chưa có ảnh")
            if update_hist:
                self.hist.set_image(None)
            return
        self._pixmap = numpy_to_qpixmap(image)
        self._render()
        if update_hist:
            self.hist.set_image(image)

    def _render(self):
        if self._pixmap is None:
            return
        size = self.view.size()
        if size.width() < 10 or size.height() < 10:
            self.view.setPixmap(self._pixmap)
            return
        self.view.setPixmap(self._pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render()
