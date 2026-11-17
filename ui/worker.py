"""
Module ui/worker.py
Quản lý luồng xử lý nền QThread để thực thi pipeline mà không làm đơ giao diện người dùng.
"""
import time
from typing import List
import numpy as np
from PySide6.QtCore import QThread, Signal

from core.pipeline import Stage, run_pipeline


class PipelineWorker(QThread):
    """Worker chạy pipeline trong background thread."""
    result_ready = Signal(object, float)  # (result_image, elapsed_ms)
    error_occurred = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image = None
        self._stages = []
        self._is_busy = False

    def process(self, orig_image: np.ndarray, stages: List[Stage]):
        """Kích hoạt chạy pipeline với dữ liệu mới."""
        if orig_image is None or orig_image.size == 0:
            return
            
        self._image = orig_image.copy()
        self._stages = [Stage(s.name, s.enabled, s.method, dict(s.params)) for s in stages]
        
        # Nếu thread đang chạy, nó sẽ tự hoàn tất; ta khởi động lại
        if not self.isRunning():
            self.start()

    def run(self):
        try:
            t0 = time.perf_counter()
            res = run_pipeline(self._image, self._stages)
            elapsed = (time.perf_counter() - t0) * 1000.0
            self.result_ready.emit(res, elapsed)
        except Exception as e:
            self.error_occurred.emit(str(e))
