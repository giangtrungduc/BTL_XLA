"""
Module ui/worker.py
Luồng nền cho giao diện, để kéo thanh trượt không làm đơ cửa sổ.

PipelineWorker: chỉ xử lý YÊU CẦU MỚI NHẤT. Khi người dùng kéo liên tục, các yêu cầu cũ
bị bỏ qua; mỗi kết quả mang mã job để cửa sổ chính bỏ kết quả đã lỗi thời.
TaskWorker: chạy một hàm bất kỳ một lần (dùng khi xuất ảnh ở độ phân giải gốc).
"""
import time
import traceback
from typing import List

import numpy as np
from PySide6.QtCore import QThread, Signal, QMutex, QMutexLocker

from core.pipeline import Stage, run_pipeline


class PipelineWorker(QThread):
    # (job_id, ảnh kết quả, danh sách bước, mặt nạ vết xước, thời gian ms)
    result_ready = Signal(int, object, object, object, float)
    error_occurred = Signal(int, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mutex = QMutex()
        self._job = None          # (job_id, image, stages) đang chờ
        self._active = False      # luồng đang trong vòng lặp xử lý
        self._next_id = 0

    def process(self, image: np.ndarray, stages: List[Stage]) -> int:
        """Gửi yêu cầu xử lý; trả về mã job."""
        if image is None or image.size == 0:
            return -1
        stages_copy = [Stage(s.name, s.enabled, s.method, dict(s.params)) for s in stages]
        with QMutexLocker(self._mutex):
            self._next_id += 1
            job_id = self._next_id
            self._job = (job_id, image, stages_copy)
            need_start = not self._active
            if need_start:
                self._active = True
        if need_start:
            if self.isRunning():
                # Vòng lặp cũ đã thoát nhưng run() chưa trả về: chờ nốt rồi khởi động lại
                self.wait()
            self.start()
        return job_id

    def run(self):
        while True:
            with QMutexLocker(self._mutex):
                job = self._job
                self._job = None
                if job is None:
                    self._active = False
                    return
            job_id, img, stages = job
            try:
                t0 = time.perf_counter()
                res, steps, mask = run_pipeline(img, stages, return_steps=True)
                elapsed = (time.perf_counter() - t0) * 1000.0
                self.result_ready.emit(job_id, res, steps, mask, elapsed)
            except Exception as e:  # báo lỗi về giao diện thay vì làm sập ứng dụng
                traceback.print_exc()
                self.error_occurred.emit(job_id, f"{type(e).__name__}: {e}")


class TaskWorker(QThread):
    """Chạy fn(*args) trong luồng nền một lần."""
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, fn, *args, parent=None):
        super().__init__(parent)
        self._fn = fn
        self._args = args

    def run(self):
        try:
            self.finished_ok.emit(self._fn(*self._args))
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(f"{type(e).__name__}: {e}")
