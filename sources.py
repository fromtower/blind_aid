"""
프레임 소스 추상화.

Pi 5에서는 Picamera2 듀얼 스트림, 개발 PC에서는 동영상 파일/웹캠.
같은 인터페이스라 main.py 는 어디서 돌든 코드가 동일하다.

    for main_frame, lores_frame, meta in source:
        ...

- main_frame  : 고해상도 BGR. ROI 크롭 전용 (신경망 미투입)
- lores_frame : 416x416 BGR. YOLO 입력
- meta        : dict. AnalogueGain / ExposureTime 등 (없으면 빈 dict)
"""

from __future__ import annotations

import time

import cv2
import numpy as np

import config


class PiCameraSource:
    """Pi 5 + picamera2 듀얼 스트림."""

    def __init__(self, main_size=None, lores_size=None, fps=None):
        from picamera2 import Picamera2  # 지연 임포트: PC에서는 안 불림

        self.main_size = main_size or config.MAIN_SIZE
        self.lores_size = lores_size or config.LORES_SIZE
        fps = fps or config.TARGET_FPS

        self.picam = Picamera2()
        cfg = self.picam.create_video_configuration(
            main={"size": self.main_size, "format": "RGB888"},
            lores={"size": self.lores_size, "format": "RGB888"},
            controls={"FrameDurationLimits": (int(1e6 / fps), int(1e6 / fps))},
            buffer_count=4,
        )
        self.picam.configure(cfg)
        self.picam.start()
        time.sleep(1.0)  # AE/AWB 수렴 대기

    def __iter__(self):
        return self

    def __next__(self):
        req = self.picam.capture_request()
        try:
            main = req.make_array("main")
            lores = req.make_array("lores")
            meta = req.get_metadata()
        finally:
            req.release()
        return main, lores, meta

    def close(self):
        try:
            self.picam.stop()
        except Exception:
            pass


class VideoSource:
    """개발용. 동영상 파일 또는 웹캠 인덱스.

    main 은 원본 해상도 그대로, lores 는 416으로 리사이즈해서 흉내낸다.
    Pi 카메라의 ISP 색감과는 다르므로, 임계값 최종 튜닝은 반드시 실기에서.
    """

    def __init__(self, path, lores_size=None, loop=False):
        self.lores_size = lores_size or config.LORES_SIZE
        self.loop = loop
        src = int(path) if str(path).isdigit() else str(path)
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"영상 소스를 열 수 없음: {path}")

    def __iter__(self):
        return self

    def __next__(self):
        ok, frame = self.cap.read()
        if not ok:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
            if not ok:
                raise StopIteration
        lores = cv2.resize(frame, self.lores_size, interpolation=cv2.INTER_AREA)
        return frame, lores, {}

    def close(self):
        self.cap.release()


def open_source(spec: str):
    """spec 이 'pi' 면 PiCamera2, 그 외에는 파일 경로/웹캠 인덱스."""
    if spec == "pi":
        return PiCameraSource()
    return VideoSource(spec, loop=True)


def crop_ratio(img: np.ndarray, roi) -> tuple[np.ndarray, tuple[int, int]]:
    """비율 ROI(x0,y0,x1,y1) 로 크롭. (크롭이미지, 좌상단 오프셋) 반환."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = roi
    px0, py0 = int(x0 * w), int(y0 * h)
    px1, py1 = int(x1 * w), int(y1 * h)
    px0, py0 = max(0, px0), max(0, py0)
    px1, py1 = min(w, px1), min(h, py1)
    if px1 <= px0 or py1 <= py0:
        return img[0:0, 0:0], (0, 0)
    return img[py0:py1, px0:px1], (px0, py0)
