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


def _classify_patch(bgr_patch, mask) -> str | None:
    """후보 blob 하나만 LAB 로 변환해 적/녹을 가른다.

    ★ ROI 전체를 cvtColor 하지 않는 것이 핵심이다. 등은 ROI 면적의 1% 미만이라
      전체 변환은 99% 가 낭비였다. 이 덕분에 ROI 를 넓혀도 비용이 거의 안 는다.
    """
    lab = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2LAB)
    a_mean = float(cv2.mean(lab[:, :, 1], mask=mask)[0])
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

    rh, rw = roi.shape[:2]

    # 1단계: 밝기 게이트. LAB 변환 없이 채널 최대값만 본다 (가장 싼 연산)
    V = roi.max(axis=2)
    bright = (V >= config.SIGNAL_V_MIN).astype(np.uint8) * 255
    bright = cv2.morphologyEx(
        bright, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    cnts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    a_lo, a_hi = config.SIGNAL_ASPECT
    best_state, best_area, best_score = NONE, 0, 0.0

    for c in cnts:
        area = cv2.contourArea(c)
        if area < config.SIGNAL_MIN_AREA_PX or area > config.SIGNAL_MAX_AREA_PX:
            continue
        peri = cv2.arcLength(c, True)
        if peri <= 0:
            continue
        if 4.0 * np.pi * area / (peri * peri) < config.SIGNAL_CIRCULARITY:
            continue

        bx, by, bw, bh = cv2.boundingRect(c)
        if bh <= 0 or not (a_lo <= bw / float(bh) <= a_hi):
            continue  # 가로로 긴 간판·전광판·차량 헤드라이트 배제

        # 2단계: 이 blob 의 바운딩 박스만 잘라서 색 판정
        patch = roi[by:by + bh, bx:bx + bw]
        local = np.zeros((bh, bw), np.uint8)
        cv2.drawContours(local, [c], -1, 255, cv2.FILLED, offset=(-bx, -by))
        st = _classify_patch(patch, local)
        if st is None:
            continue

        # 3단계: ROI 를 넓힌 만큼 후보가 늘어나므로, 중앙에 가까운 쪽을 선호한다.
        # ★ 면적이 아니라 sqrt(면적) = 선형 크기로 비교한다.
        #   면적은 거리 제곱에 반비례해서, 가까운 가장자리 물체(차량 미등 등)가
        #   면적만으로는 정면 신호등을 너무 쉽게 이긴다.
        cx = (bx + bw / 2.0) / max(rw, 1)
        centrality = 1.0 - config.SIGNAL_CENTER_BIAS * min(1.0, abs(cx - 0.5) * 2.0)
        score = float(np.sqrt(area)) * centrality

        if score > best_score:
            best_state, best_area, best_score = st, int(area), score

    return best_state, best_area


class SignalVoter:
    """다프레임 투표 + 점멸 검출 + 상태 변화 이벤트.

    detector 는 (main_bgr) -> (state, area_px) 인 호출가능 객체면 무엇이든 된다.
    기본값은 이 파일의 색공간 규칙이고, signal_model.ModelSignalDetector 를
    넣으면 YOLO+CNN 경로가 된다. 투표·점멸 로직은 백엔드와 무관하게 공통이다.
    """

    def __init__(self, detector=None):
        self.window: deque[str] = deque(maxlen=config.SIGNAL_VOTE_N)
        self.confirmed = "unknown"
        self.detector = detector if detector is not None else detect_signal_once

    def update(self, main_bgr: np.ndarray | None, profile: str) -> SignalResult:
        # 야간에는 아예 판정하지 않는다. 오판보다 "모름"이 안전하다.
        if profile not in config.SIGNAL_REQUIRE_PROFILE or main_bgr is None:
            self.window.clear()
            if self.confirmed != "unknown":
                self.confirmed = "unknown"
            return SignalResult("unknown", False, False, 0)

        st, area = self.detector(main_bgr)
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
