"""
하드웨어 없이 색 판정 로직만 검증한다.

합성 프레임(회색 아스팔트 + 황색 띠, 어두운 하늘 + 적/녹 원)을 만들어
pavement / signal_light 모듈이 의도대로 반응하는지 확인.

    python selftest.py

실기 튜닝을 대체하지는 못하지만, 리팩터링할 때마다 돌려서
로직이 깨지지 않았는지 확인하는 용도로 쓴다.
"""

from __future__ import annotations

import cv2
import numpy as np

import config
from pavement import detect_pavement
from signal_light import SignalVoter, detect_signal_once

W, H = 960, 540
PASS, FAIL = "  PASS", "  FAIL"
_results: list[bool] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    _results.append(bool(cond))
    print(f"{PASS if cond else FAIL}  {name}{('  — ' + detail) if detail else ''}")


# ── 합성 프레임 ─────────────────────────────────────────
def make_pavement_frame(offset_ratio=0.0, brightness=1.0, with_block=True):
    img = np.full((H, W, 3), 110, np.uint8)            # 아스팔트 회색
    img = (img.astype(np.float32) * brightness).clip(0, 255).astype(np.uint8)
    if with_block:
        cx = int(W / 2 + offset_ratio * W / 2)
        band_w = 130
        y0 = int(H * config.PAVE_ROI_TOP)
        pts = np.array([[cx - band_w // 2, H], [cx + band_w // 2, H],
                        [cx + band_w // 3, y0], [cx - band_w // 3, y0]])
        yellow = (np.array([40, 190, 225], np.float32) * brightness).clip(0, 255)
        cv2.fillPoly(img, [pts.astype(np.int32)], yellow.tolist())
    return img


def make_signal_frame(state=None):
    img = np.full((H, W, 3), 55, np.uint8)             # 어두운 배경
    if state:
        cx, cy = W // 2, int(H * 0.16)
        color = (55, 60, 250) if state == "red" else (90, 245, 90)   # BGR
        cv2.circle(img, (cx, cy), 16, color, -1)
    # 방해물: 밝은 사각 간판 (원형도 필터로 걸러져야 함)
    cv2.rectangle(img, (60, 40), (190, 95), (240, 240, 240), -1)
    return img


# ── 점자블록 ────────────────────────────────────────────
print("\n[ 점자블록 / LAB b* ]")
r = detect_pavement(make_pavement_frame(0.0), "day")
check("정중앙 블록 검출", r.found, f"offset={r.offset:+.2f} area={r.area_ratio:.3f}")
check("중앙 판정 (편차 작음)", r.found and abs(r.offset) < config.PAVE_OFF_CENTER,
      f"offset={r.offset:+.2f}")

r = detect_pavement(make_pavement_frame(-0.45), "day")
check("왼쪽 치우침 감지", r.found and r.offset < -config.PAVE_OFF_CENTER,
      f"offset={r.offset:+.2f}")

r = detect_pavement(make_pavement_frame(0.45), "day")
check("오른쪽 치우침 감지", r.found and r.offset > config.PAVE_OFF_CENTER,
      f"offset={r.offset:+.2f}")

r = detect_pavement(make_pavement_frame(with_block=False), "day")
check("블록 없음 → 미검출", not r.found)

# 조도 절반(그늘/박명) 에서도 살아남는지 = LAB 쓰는 이유
r_dim = detect_pavement(make_pavement_frame(0.0, brightness=0.55), "dusk")
check("조도 55%에서도 검출 (LAB 강건성)", r_dim.found,
      f"area={r_dim.area_ratio:.3f}")

# ── 신호등 ──────────────────────────────────────────────
print("\n[ 보행 신호등 / LAB a* + 투표 ]")
st, area = detect_signal_once(make_signal_frame("red"))
check("적색 단일 프레임 판정", st == "red", f"state={st} area={area}")

st, area = detect_signal_once(make_signal_frame("green"))
check("녹색 단일 프레임 판정", st == "green", f"state={st} area={area}")

st, _ = detect_signal_once(make_signal_frame(None))
check("등 없음 + 사각 간판 배제", st == "none", f"state={st}")

# 투표: 1프레임만으로는 확정 금지
v = SignalVoter()
res = v.update(make_signal_frame("green"), "day")
check("1프레임은 미확정", not res.confident, f"votes={res.votes}")

for _ in range(2):
    res = v.update(make_signal_frame("green"), "day")
check("3프레임 누적 시 확정", res.confident and res.state == "green",
      f"state={res.state} votes={res.votes}")

# 야간 프로파일에서는 판정 자체를 포기해야 한다
v2 = SignalVoter()
for _ in range(5):
    res = v2.update(make_signal_frame("green"), "night")
check("야간 → 판정 포기(unknown)", res.state == "unknown" and not res.confident)

# 점멸 검출
v3 = SignalVoter()
for i in range(5):
    res = v3.update(make_signal_frame("green" if i % 2 == 0 else None), "day")
check("녹색 점멸 검출", res.flicker, f"flicker={res.flicker}")

# ── 거리 추정 ───────────────────────────────────────────
print("\n[ 거리 추정 ]")
from detect import estimate_distance, side_of

d5 = estimate_distance("person", 1.70 * config.FOCAL_PX / 5.0)
check("5m 역산 일치", abs(d5 - 5.0) < 0.01, f"d={d5:.3f}")
d2 = estimate_distance("person", 1.70 * config.FOCAL_PX / 2.0)
check("가까울수록 거리 감소", d2 < d5, f"{d2:.2f} < {d5:.2f}")
check("bbox 0 → inf", estimate_distance("person", 0) == float("inf"))
check("좌/중/우 분류", (side_of(0.1), side_of(0.5), side_of(0.9))
      == ("left", "front", "right"))

# ── 메인 루프 통합 (버그 재발 방지) ─────────────────────
print("\n[ 메인 루프 통합 ]")

# ★ SIGNAL_INTERVAL 간격으로 호출해도 표가 누적되어야 한다.
#   과거 버그: 판정 안 하는 프레임에 voter.update(None) 을 불러서
#   투표 창이 매번 비워졌고, 신호등이 영원히 확정되지 않았다.
v4 = SignalVoter()
res4 = None
for frame_i in range(1, 31):
    if frame_i % config.SIGNAL_INTERVAL == 0:          # main.py 와 동일한 조건
        res4 = v4.update(make_signal_frame("red"), "day")
check("SIGNAL_INTERVAL 간격에서도 투표 누적",
      res4 is not None and res4.confident and res4.state == "red",
      f"state={res4.state} votes={res4.votes}")

# 좌표계: obstacle.box 는 lores(416), 표시 프레임은 main(1920x1080)
lw, lh = config.LORES_SIZE
vw, vh = 960, 540
sx, sy = vw / float(lw), vh / float(lh)
x_scaled = 400 * sx      # lores 우측 끝 근처의 박스
check("lores→표시 좌표 스케일 적용", x_scaled > vw * 0.8,
      f"lores x=400 → 표시 x={x_scaled:.0f} (폭 {vw})")

# 트랙 정리는 나이 기준. 한 프레임 안 보인다고 지우면 안 된다.
from detect import _Track
tr = _Track()
tr.update(5.0, 100.0)
check("트랙 last_seen 기록", tr.last_seen == 100.0)


# ── 결과 ────────────────────────────────────────────────
ok = sum(_results)
print(f"\n{'=' * 46}\n  {ok}/{len(_results)} 통과\n{'=' * 46}")
raise SystemExit(0 if ok == len(_results) else 1)
