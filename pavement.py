"""
점자블록(황색 유도블록) 추적. 학습 데이터 0장.

HSV 대신 LAB 을 쓰는 이유:
  HSV 의 H 는 조도가 떨어지면 S 가 무너지면서 같이 불안정해진다.
  LAB 의 b* 는 "파랑↔노랑" 축이라 밝기(L)와 분리되어 있어
  역광/그늘/박명에서 훨씬 덜 흔들린다.

출력:
  found        : 블록 검출 여부
  offset       : 화면 중심 대비 좌우 편차 (-1 ~ +1, 음수 = 블록이 왼쪽)
  angle_deg    : 블록 진행 방향 (0 = 정면, 양수 = 오른쪽으로 휨)
  area_ratio   : ROI 대비 면적
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

import config
from sources import crop_ratio


@dataclass
class PaveResult:
    found: bool
    offset: float = 0.0
    angle_deg: float = 0.0
    area_ratio: float = 0.0
    mask: np.ndarray | None = None


def _b_threshold(profile: str) -> int:
    return config.PAVE_B_MIN_NIGHT if profile == "night" else config.PAVE_B_MIN_DAY


def detect_pavement(frame_bgr: np.ndarray, profile: str = "day") -> PaveResult:
    roi, _ = crop_ratio(frame_bgr, (0.0, config.PAVE_ROI_TOP, 1.0, 1.0))
    if roi.size == 0:
        return PaveResult(False)

    # 계산량 절감: ROI 를 폭 320 기준으로 축소 (형상 판정에 충분)
    h, w = roi.shape[:2]
    if w > 320:
        scale = 320.0 / w
        roi = cv2.resize(roi, (320, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        h, w = roi.shape[:2]

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    b_min = _b_threshold(profile)
    a_lo, a_hi = config.PAVE_A_RANGE
    mask = ((B >= b_min) & (A >= a_lo) & (A <= a_hi) &
            (L >= config.PAVE_L_MIN)).astype(np.uint8) * 255

    # 블록 사이 줄눈 때문에 조각나므로 세로로 길게 닫아준다
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return PaveResult(False, mask=mask)

    roi_area = float(h * w)
    big = max(cnts, key=cv2.contourArea)
    area_ratio = cv2.contourArea(big) / roi_area
    if area_ratio < config.PAVE_MIN_AREA_RATIO:
        return PaveResult(False, area_ratio=area_ratio, mask=mask)

    # 진행 방향: 최소외접사각형의 장축 각도
    (cx, cy), (rw, rh), ang = cv2.minAreaRect(big)
    if rw < rh:
        angle_deg = ang
    else:
        angle_deg = ang + 90.0
    angle_deg = ((angle_deg + 90.0) % 180.0) - 90.0   # -90 ~ +90 정규화

    offset = (cx - w / 2.0) / (w / 2.0)

    return PaveResult(True, offset=float(offset), angle_deg=float(angle_deg),
                      area_ratio=float(area_ratio), mask=mask)


class PavementTracker:
    """연속 미검출 카운팅 → '블록을 벗어났습니다' 판정."""

    def __init__(self):
        self.lost = 0
        self.was_on = False

    def update(self, res: PaveResult) -> str | None:
        """안내가 필요하면 문구, 아니면 None."""
        if res.found:
            self.lost = 0
            first = not self.was_on
            self.was_on = True
            if first:
                return "유도블록을 찾았습니다"
            if res.offset < -config.PAVE_OFF_CENTER:
                return "유도블록이 왼쪽에 있습니다"
            if res.offset > config.PAVE_OFF_CENTER:
                return "유도블록이 오른쪽에 있습니다"
            return None

        self.lost += 1
        if self.was_on and self.lost == config.PAVE_LOST_FRAMES:
            self.was_on = False
            return "유도블록을 벗어났습니다"
        return None
