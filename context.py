"""
맥락 축: 조도 프로파일 판정.

기존 Pi5 파이프라인 5축의 "카메라 gain" 항목을 그대로 확장한 것.
BH1750 조도센서가 붙어 있으면 그걸 쓰고, 없으면 카메라 메타데이터
(AnalogueGain × ExposureTime)를 프록시로 쓴다. 센서 없이도 동작한다.

프로파일:
  day   → 전 기능 동작
  dusk  → 점자블록 임계값 완화, 신호등은 계속 판정
  night → 신호등 판정 포기 + 사용자에게 명시적으로 알림
"""

from __future__ import annotations

import config


class LuxSensor:
    """BH1750. 없으면 조용히 비활성화된다."""

    def __init__(self, addr: int | None = None, bus_no: int | None = None):
        self.ok = False
        self.bus = None
        try:
            from smbus2 import SMBus  # 지연 임포트
            self.addr = addr or config.BH1750_ADDR
            self.bus = SMBus(bus_no or config.BH1750_BUS)
            self.bus.write_byte(self.addr, 0x10)  # 연속 고해상도 모드
            self.ok = True
        except Exception:
            self.ok = False

    def read(self) -> float | None:
        if not self.ok:
            return None
        try:
            data = self.bus.read_i2c_block_data(self.addr, 0x10, 2)
            return ((data[0] << 8) | data[1]) / 1.2
        except Exception:
            return None

    def close(self):
        try:
            if self.bus:
                self.bus.close()
        except Exception:
            pass


def profile_from_lux(lux: float) -> str:
    if lux >= config.LUX_DAY:
        return "day"
    if lux >= config.LUX_DUSK:
        return "dusk"
    return "night"


def profile_from_meta(meta: dict) -> str:
    """조도센서 없을 때의 fallback.

    카메라가 어두운 장면에서 gain 을 올리는 성질을 역이용한다.
    센서보다 부정확하지만 추가 부품 0개로 동작한다.
    """
    gain = float(meta.get("AnalogueGain", 1.0) or 1.0)
    if gain <= config.GAIN_DAY_MAX:
        return "day"
    if gain <= config.GAIN_DUSK_MAX:
        return "dusk"
    return "night"


class ContextTracker:
    """프로파일 히스테리시스. 경계에서 깜빡이는 걸 막는다."""

    def __init__(self, sensor: LuxSensor | None = None, hold: int = 10):
        self.sensor = sensor
        self.hold = hold
        self.profile = "day"
        self._pending = "day"
        self._count = 0
        self.last_lux: float | None = None

    def update(self, meta: dict) -> str:
        lux = self.sensor.read() if self.sensor else None
        self.last_lux = lux
        p = profile_from_lux(lux) if lux is not None else profile_from_meta(meta)

        if p == self.profile:
            self._count = 0
        elif p == self._pending:
            self._count += 1
            if self._count >= self.hold:
                self.profile = p
                self._count = 0
        else:
            self._pending = p
            self._count = 1
        return self.profile

    @property
    def source_name(self) -> str:
        return "BH1750" if (self.sensor and self.sensor.ok) else "camera-gain"
