"""
음성 안내.

시각장애인 보조에서 가장 흔한 실패는 "인식 실패"가 아니라 "말이 너무 많음"이다.
계속 떠들면 사용자가 흰지팡이·주변 소리에 집중하지 못해 오히려 위험해진다.

그래서 여기서 하는 일:
  1. 우선순위 (신호등/즉시위험 > 접근경고 > 유도블록 > 상태)
  2. 종류별 쿨다운 — 같은 말 반복 금지
  3. 큐 포화 시 낮은 우선순위 폐기
  4. 별도 스레드에서 재생 → 메인 루프 블로킹 방지

TTS 백엔드:
  espeak  : 오프라인, 즉시 되지만 한국어 발음이 거칠다
  wav     : 고정 문구를 미리 렌더해둔 wav 재생 (★ 대회 데모용 권장)
  print   : 개발용 콘솔 출력
"""

from __future__ import annotations

import queue
import subprocess
import threading
import time
from pathlib import Path

import config


class Announcer:
    def __init__(self, backend: str = "print", wav_dir: str = "voice"):
        self.backend = backend
        self.wav_dir = Path(wav_dir)
        self.q: queue.PriorityQueue = queue.PriorityQueue()
        self.last_spoken: dict[str, float] = {}
        self.last_text: dict[str, str] = {}
        self._seq = 0
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._worker, daemon=True)
        self._th.start()

    # ── 입력 ────────────────────────────────────────────
    def say(self, kind: str, text: str, force: bool = False) -> bool:
        """kind 는 config.PRIORITY 의 키. 쿨다운에 걸리면 False 반환."""
        now = time.monotonic()
        cd = config.COOLDOWN_S.get(kind, 3.0)

        if not force:
            if now - self.last_spoken.get(kind, -1e9) < cd:
                # 같은 종류라도 내용이 바뀌었고 우선순위가 최상위면 통과시킨다
                if not (config.PRIORITY.get(kind, 9) == 0
                        and self.last_text.get(kind) != text):
                    return False

        if self.q.qsize() >= config.QUEUE_MAX:
            self._drop_low_priority(config.PRIORITY.get(kind, 9))

        self.last_spoken[kind] = now
        self.last_text[kind] = text
        self._seq += 1
        self.q.put((config.PRIORITY.get(kind, 9), self._seq, kind, text))
        return True

    def _drop_low_priority(self, incoming_pri: int) -> None:
        kept = []
        while not self.q.empty():
            try:
                item = self.q.get_nowait()
            except queue.Empty:
                break
            if item[0] <= incoming_pri:
                kept.append(item)
        for it in kept:
            self.q.put(it)

    # ── 출력 ────────────────────────────────────────────
    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                _, _, kind, text = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._speak(text)
            except Exception as e:  # 음성이 죽어도 본체는 계속 돌아야 한다
                print(f"[TTS 실패] {text} ({e})")

    def _speak(self, text: str) -> None:
        if self.backend == "print":
            print(f"[안내] {text}", flush=True)
            return
        if self.backend == "espeak":
            subprocess.run(["espeak-ng", "-v", "ko", "-s", "165", text],
                           check=False, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)
            return
        if self.backend == "wav":
            path = self.wav_dir / f"{_slug(text)}.wav"
            if path.exists():
                subprocess.run(["aplay", "-q", str(path)], check=False)
            else:
                print(f"[안내:wav없음] {text}", flush=True)
            return
        print(f"[안내] {text}", flush=True)

    def close(self) -> None:
        self._stop.set()
        self._th.join(timeout=1.0)


def _slug(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text)[:60]


# 미리 렌더해두면 좋은 고정 문구 목록 (make_voice.py 가 이걸 읽는다)
PHRASES = [
    "녹색불입니다",
    "빨간불입니다. 정지하세요",
    "녹색불 점멸입니다. 건너지 마세요",
    "신호를 확인할 수 없습니다",
    "유도블록을 찾았습니다",
    "유도블록을 벗어났습니다",
    "유도블록이 왼쪽에 있습니다",
    "유도블록이 오른쪽에 있습니다",
    "정면 장애물",
    "왼쪽 장애물",
    "오른쪽 장애물",
    "정지하세요",
    "시스템을 시작합니다",
]
