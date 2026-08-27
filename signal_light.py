"""
보행 신호등 판정.

★ 핵심 설계: lores 416 을 쓰지 않는다.
  20m 거리의 보행등(약 30cm)은 원본 2592px 기준 ~38px 인데,
  416 으로 다운스케일하면 ~6px 이 되어 색 판정이 물리적으로 불가능하다.
  따라서 MAIN 스트림 상단 ROI 를 원본 해상도로 크롭해서 처리한다.
  (기존 파이프라인의 "손목 ROI 크롭 → 분류기" 와 동일한 패턴)

판정은 단일 프레임 금지. 5프레임 투표에서 3표 이상일 때만 확정한다.
점멸(곧 바뀜)은 최근 창의 켜짐/꺼짐 전환 횟수로 검출한다.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass

import cv2
import numpy as np

import config
from sources import crop_ratio

RED, GREEN, NONE = "red", "green", "none"


@dataclass
class SignalResult:
    state: str          # "red" | "green" | "none" | "unknown"
    confident: bool     # 투표 통과 여부
    flicker: bool       # 점멸 = 곧 바뀜
    votes: int
    blob_px: int = 0


def _classify_blob(lab_roi, mask) -> str | None:
    a_mean = float(cv2.mean(lab_roi[:, :, 1], mask=mask)[0])
    if a_mean >= config.SIGNAL_RED_A_MIN:
        return RED
    if a_mean <= config.SIGNAL_GREEN_A_MAX:
        return GREEN
    return None


def detect_signal_once(main_bgr: np.ndarray) -> tuple[str, int]:
    """단일 프레임 판정. (상태, blob 픽셀수). 투표 전 단계라 그대로 쓰지 말 것."""
    roi, _ = crop_ratio(main_bgr, config.SIGNAL_ROI)
    if roi.size == 0:
        return NONE, 0

    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

    # 밝기 게이트는 LAB L 이 아니라 채널 최대값 V 로 한다.
    # (L 로 하면 시감 휘도가 낮은 적색등이 통째로 탈락한다 — config.py 주석 참고)
    V = roi.max(axis=2)
    bright = (V >= config.SIGNAL_V_MIN).astype(np.uint8) * 255
    bright = cv2.morphologyEx(
        bright, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    cnts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_state, best_area = NONE, 0
    a_lo, a_hi = config.SIGNAL_ASPECT

    for c in cnts:
        area = cv2.contourArea(c)
        if area < config.SIGNAL_MIN_AREA_PX or area > config.SIGNAL_MAX_AREA_PX:
            continue
        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        circ = 4.0 * np.pi * area / (peri * peri)
        if circ < config.SIGNAL_CIRCULARITY:
            continue

        _, _, bw, bh = cv2.boundingRect(c)
        if bh <= 0:
            continue
        aspect = bw / float(bh)
        if not (a_lo <= aspect <= a_hi):
            continue  # 가로로 긴 간판·전광판·차량 헤드라이트 배제

        blob_mask = np.zeros(V.shape, np.uint8)
        cv2.drawContours(blob_mask, [c], -1, 255, cv2.FILLED)
        st = _classify_blob(lab, blob_mask)
        if st is None:
            continue
        if area > best_area:
            best_state, best_area = st, int(area)

    return best_state, best_area


class SignalVoter:
    """다프레임 투표 + 점멸 검출 + 상태 변화 이벤트."""

    def __init__(self):
        self.window: deque[str] = deque(maxlen=config.SIGNAL_VOTE_N)
        self.confirmed = "unknown"

    def update(self, main_bgr: np.ndarray | None, profile: str) -> SignalResult:
        # 야간에는 아예 판정하지 않는다. 오판보다 "모름"이 안전하다.
        if profile not in config.SIGNAL_REQUIRE_PROFILE or main_bgr is None:
            self.window.clear()
            if self.confirmed != "unknown":
                self.confirmed = "unknown"
            return SignalResult("unknown", False, False, 0)

        st, area = detect_signal_once(main_bgr)
        self.window.append(st)

        cnt = Counter(self.window)
        state, votes = cnt.most_common(1)[0]
        confident = votes >= config.SIGNAL_VOTE_MIN and state != NONE

        # 점멸: 창 안에서 켜짐↔꺼짐 전환이 잦으면 곧 바뀌는 중
        transitions = sum(
            1 for a, b in zip(self.window, list(self.window)[1:])
            if (a == NONE) != (b == NONE)
        )
        flicker = transitions >= config.SIGNAL_FLICKER_MIN

        if confident:
            self.confirmed = state

        return SignalResult(state if confident else "unknown",
                            confident, flicker, votes, area)

    def take_event(self, res: SignalResult) -> str | None:
        """상태가 실제로 바뀐 순간에만 안내 문구를 낸다."""
        if not res.confident:
            return None
        if res.flicker and res.state == GREEN:
            return "녹색불 점멸입니다. 건너지 마세요"
        if res.state == GREEN:
            return "녹색불입니다"
        if res.state == RED:
            return "빨간불입니다. 정지하세요"
        return None
