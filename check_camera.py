#!/usr/bin/env python3
"""
카메라 1차 진단. 이 파일 하나만 파이에 올려서 돌리면 된다.

    python3 check_camera.py

blindnav 다른 파일에 의존하지 않는다. 여기서 통과해야 본체를 올릴 의미가 있다.

확인 항목
  1. picamera2 임포트 / 센서 지원 모드
  2. 듀얼 스트림(main + lores) 구성 성공 여부
  3. 실측 fps
  4. ★ 채널 순서 (RGB/BGR 함정) — 가장 중요
  5. AnalogueGain / ExposureTime — context.py fallback 이 쓸 값
  6. BH1750 조도센서 유무 (없어도 됨)
"""

import sys
import time

OK = "  [OK] "
NG = "  [NG] "
WARN = "  [??] "

MAIN_SIZE = (1920, 1080)   # 5MP OV5647 안전 모드
LORES_SIZE = (416, 416)
FPS = 15


def hr(title):
    print(f"\n{'=' * 52}\n  {title}\n{'=' * 52}")


# ── 1. 임포트 ───────────────────────────────────────────
hr("1. 라이브러리")
try:
    import numpy as np
except ImportError:
    print(NG + "numpy 없음 →  sudo apt install -y python3-numpy")
    sys.exit(1)
print(OK + f"numpy {np.__version__}")

try:
    import cv2
except ImportError:
    print(NG + "opencv 없음 →  sudo apt install -y python3-opencv")
    sys.exit(1)
print(OK + f"opencv {cv2.__version__}")

try:
    from picamera2 import Picamera2
except ImportError:
    print(NG + "picamera2 없음")
    print("      sudo apt install -y python3-picamera2")
    print("      venv 안이면 --system-site-packages 로 다시 만들 것")
    sys.exit(1)
print(OK + "picamera2 임포트 성공")


# ── 2. 센서 모드 ────────────────────────────────────────
hr("2. 센서 지원 모드")
try:
    picam = Picamera2()
except Exception as e:
    print(NG + f"카메라를 열 수 없음: {e}")
    print("      rpicam-hello -t 3000  으로 하드웨어부터 확인")
    print("      리본 케이블 파란 면이 USB 포트 쪽을 향해야 함")
    sys.exit(1)

modes = picam.sensor_modes
for i, m in enumerate(modes):
    print(f"      [{i}] size={m.get('size')} fps={m.get('fps')} "
          f"bit={m.get('bit_depth')}")

max_w = max((m["size"][0] for m in modes), default=0)
if max_w < MAIN_SIZE[0]:
    print(WARN + f"MAIN_SIZE 폭 {MAIN_SIZE[0]} > 센서 최대 {max_w}")
    print("      config.py 의 MAIN_SIZE 를 낮춰야 함")
else:
    print(OK + f"MAIN_SIZE {MAIN_SIZE} 사용 가능 (센서 최대 폭 {max_w})")
    if max_w >= 2304:
        print(WARN + "센서가 2304 이상 지원 → MAIN_SIZE 올려도 됨 "
                     "(신호등 원거리 판정에 유리)")


# ── 3. 듀얼 스트림 ──────────────────────────────────────
hr("3. 듀얼 스트림 구성")
try:
    cfg = picam.create_video_configuration(
        main={"size": MAIN_SIZE, "format": "RGB888"},
        lores={"size": LORES_SIZE, "format": "RGB888"},
        controls={"FrameDurationLimits": (int(1e6 / FPS), int(1e6 / FPS))},
        buffer_count=4,
    )
    picam.configure(cfg)
    picam.start()
    time.sleep(1.2)  # AE/AWB 수렴
except Exception as e:
    print(NG + f"듀얼 스트림 실패: {e}")
    print("      lores 는 main 보다 작아야 하고, 일부 모드에서 제약이 있음")
    sys.exit(1)
print(OK + "듀얼 스트림 구성 성공")


# ── 4. 캡처 + fps ───────────────────────────────────────
hr("4. 캡처 / 실측 fps")
N = 40
t0 = time.monotonic()
main = lores = None
meta = {}
for _ in range(N):
    req = picam.capture_request()
    try:
        main = req.make_array("main")
        lores = req.make_array("lores")
        meta = req.get_metadata()
    finally:
        req.release()
elapsed = time.monotonic() - t0
fps = N / elapsed

print(OK + f"main  shape={main.shape} dtype={main.dtype}")
print(OK + f"lores shape={lores.shape} dtype={lores.dtype}")
print(OK + f"실측 {fps:.1f} fps  (목표 {FPS})")
if fps < FPS * 0.8:
    print(WARN + "목표 대비 낮음. 이건 캡처만 한 수치이고 "
                 "YOLO 얹으면 더 떨어진다는 뜻")

if main.shape[:2] != (MAIN_SIZE[1], MAIN_SIZE[0]):
    print(WARN + f"요청 {MAIN_SIZE} ≠ 실제 {main.shape[1::-1]} "
                 "→ config.py 를 실제값으로 맞출 것")


# ── 5. ★ 채널 순서 ──────────────────────────────────────
hr("5. 채널 순서 (가장 중요)")
print("""  picamera2 의 "RGB888" 은 이름과 달리 메모리에 BGR 로 들어오는 경우가 있다.
  이게 뒤집혀 있으면 LAB a* 가 반전되어 빨간불과 초록불이 서로 바뀐다.
  시각장애인 보조에서 상상 가능한 최악의 버그다.
""")
cv2.imwrite("chk_asis.jpg", main)
cv2.imwrite("chk_swapped.jpg", cv2.cvtColor(main, cv2.COLOR_RGB2BGR))

h, w = main.shape[:2]
patch = main[h // 2 - 40:h // 2 + 40, w // 2 - 40:w // 2 + 40]
c0, c1, c2 = (float(patch[:, :, i].mean()) for i in range(3))
print(f"  중앙 패치 채널 평균: ch0={c0:.0f} ch1={c1:.0f} ch2={c2:.0f}")
print("""
  ▶ 카메라 앞에 새빨간 물건을 두고 이 스크립트를 다시 돌린 뒤,
    chk_asis.jpg 와 chk_swapped.jpg 를 열어볼 것.

      chk_asis.jpg 가 빨갛다      → 정상. 코드 수정 불필요
      chk_swapped.jpg 가 빨갛다   → 뒤집힘! sources.py 의 __next__ 에서
                                    main  = cv2.cvtColor(main,  cv2.COLOR_RGB2BGR)
                                    lores = cv2.cvtColor(lores, cv2.COLOR_RGB2BGR)
                                    두 줄 추가할 것
""")
cv2.imwrite("chk_lores.jpg", lores)
print(OK + "chk_asis.jpg / chk_swapped.jpg / chk_lores.jpg 저장됨")


# ── 6. 메타데이터 ───────────────────────────────────────
hr("6. 메타데이터 (조도 fallback 용)")
gain = meta.get("AnalogueGain")
expo = meta.get("ExposureTime")
lux_hint = meta.get("Lux")
print(f"      AnalogueGain = {gain}")
print(f"      ExposureTime = {expo} us")
print(f"      Lux(추정)     = {lux_hint}")
if gain is None:
    print(WARN + "AnalogueGain 없음 → context.py 의 gain fallback 불가. "
                 "BH1750 조도센서를 붙이는 게 안전")
else:
    print(OK + "gain 읽힘. 조도센서 없어도 주/야 판정 가능")
    print("      ※ 실내/실외에서 각각 이 값을 적어둔 뒤 "
          "config.GAIN_DAY_MAX 를 그 사이로 잡을 것")
if lux_hint is not None:
    print(OK + f"Lux 메타 제공됨 → config.LUX_* 를 바로 쓸 수 있음")


# ── 7. BH1750 (선택) ────────────────────────────────────
hr("7. BH1750 조도센서 (없어도 됨)")
try:
    from smbus2 import SMBus
    with SMBus(1) as bus:
        bus.write_byte(0x23, 0x10)
        time.sleep(0.2)
        d = bus.read_i2c_block_data(0x23, 0x10, 2)
        print(OK + f"BH1750 감지됨. lux = {((d[0] << 8) | d[1]) / 1.2:.0f}")
except ImportError:
    print(WARN + "smbus2 미설치 (pip install smbus2). 없으면 gain fallback 사용")
except Exception:
    print(WARN + "BH1750 없음 → gain fallback 으로 동작. 문제 없음")


picam.stop()
hr("진단 끝")
print("""  다음 순서
    1) chk_asis.jpg 로 채널 순서 확인   ← 여기 먼저
    2) python3 selftest.py             ← 카메라 없이 로직 검증
    3) python3 main.py --source pi --no-detect --view
                                        ← YOLO 없이 색 로직만, ultralytics 불필요
    4) ultralytics 설치 후 calibrate.py → main.py 전체
""")
