"""
M1: 장애물 검출 + 거리 추정 + 접근 판정.

학습 데이터 0장. COCO 사전학습 YOLO 를 그대로 쓴다.
거리 추정은 bbox 높이 기반 단안 추정이라 캘리브레이션 1회만 필요.

    D = (H_real * FOCAL_PX) / h_px
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

import config


@dataclass
class Obstacle:
    track_id: int
    label: str
    conf: float
    box: tuple[float, float, float, float]   # x1,y1,x2,y2 (lores 좌표)
    dist_m: float                            # 스무딩된 거리
    speed_mps: float                         # 음수 = 접근 중
    side: str                                # "left" | "front" | "right"

    @property
    def is_danger(self) -> bool:
        return self.dist_m <= config.DANGER_DIST_M

    @property
    def is_approaching(self) -> bool:
        return self.speed_mps <= config.APPROACH_MPS


def estimate_distance(label: str, box_h_px: float, focal_px: float | None = None) -> float:
    """bbox 높이(px) → 거리(m). 0 이하면 inf."""
    if box_h_px <= 1:
        return float("inf")
    f = focal_px if focal_px is not None else config.FOCAL_PX
    h_real = config.REAL_HEIGHT_M.get(label, config.DEFAULT_HEIGHT_M)
    return (h_real * f) / box_h_px


def side_of(cx_ratio: float) -> str:
    if cx_ratio < config.LEFT_EDGE:
        return "left"
    if cx_ratio > config.RIGHT_EDGE:
        return "right"
    return "front"


class _Track:
    """거리 시계열 스무딩 + 미분. ByteTrack 의 track_id 를 키로 유지."""

    __slots__ = ("dists", "times", "last_seen")

    def __init__(self):
        self.dists = deque(maxlen=config.DIST_SMOOTH_N)
        self.times = deque(maxlen=config.DIST_SMOOTH_N)
        self.last_seen = 0.0

    def update(self, d: float, t: float) -> tuple[float, float]:
        self.dists.append(d)
        self.times.append(t)
        self.last_seen = t
        smooth = float(np.median(self.dists))
        if len(self.dists) < 3:
            return smooth, 0.0
        # 최소자승 직선 기울기 = 접근 속도(m/s)
        ts = np.asarray(self.times, dtype=np.float64)
        ds = np.asarray(self.dists, dtype=np.float64)
        ts = ts - ts[0]
        if ts[-1] <= 1e-6:
            return smooth, 0.0
        slope = float(np.polyfit(ts, ds, 1)[0])
        return smooth, slope


class ObstacleDetector:
    """ultralytics YOLO 래퍼. ByteTrack 내장 추적 사용(추가 모델 없음)."""

    def __init__(self, weights: str = "yolo11n.pt", imgsz: int | None = None,
                 conf: float = 0.35, device: str = "cpu"):
        from ultralytics import YOLO  # 지연 임포트

        self.model = YOLO(weights)
        self.imgsz = imgsz or config.LORES_SIZE[0]
        self.conf = conf
        self.device = device
        self.tracks: dict[int, _Track] = defaultdict(_Track)
        self._next_fake_id = -1

    def __call__(self, lores_bgr: np.ndarray, t: float) -> list[Obstacle]:
        res = self.model.track(
            lores_bgr, imgsz=self.imgsz, conf=self.conf, device=self.device,
            persist=True, tracker="bytetrack.yaml", verbose=False,
        )
        out: list[Obstacle] = []
        if not res:
            return out
        r = res[0]
        if r.boxes is None or len(r.boxes) == 0:
            return out

        names = r.names
        h, w = lores_bgr.shape[:2]
        for b in r.boxes:
            label = names[int(b.cls[0])]
            if label not in config.OBSTACLE_CLASSES:
                continue
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            box_h = y2 - y1
            raw_d = estimate_distance(label, box_h)
            if raw_d > config.MAX_TRUST_DIST_M:
                continue  # 8m 초과는 신뢰 구간 밖 → 안내하지 않음

            if b.id is not None:
                tid = int(b.id[0])
            else:
                tid = self._next_fake_id
                self._next_fake_id -= 1

            dist, speed = self.tracks[tid].update(raw_d, t)
            out.append(Obstacle(
                track_id=tid, label=label, conf=float(b.conf[0]),
                box=(x1, y1, x2, y2), dist_m=dist, speed_mps=speed,
                side=side_of(((x1 + x2) / 2) / max(w, 1)),
            ))

        out.sort(key=lambda o: o.dist_m)
        return out

    def prune(self, now: float, max_age_s: float = 2.0) -> None:
        """오래 안 보인 트랙만 정리한다.

        ★ 현재 프레임에 안 보인다고 바로 지우면 안 된다. 잠깐 가려졌다가
          다시 나타나는 흔한 상황에서 거리 이력이 통째로 날아가고,
          접근 속도가 0 으로 리셋되어 경고를 놓친다.
        """
        for k in list(self.tracks):
            if now - self.tracks[k].last_seen > max_age_s:
                del self.tracks[k]
