"""라즈베리파이에서 신호등 백엔드 속도를 비교한다.

PC(x86)에서는 NCNN 이 PyTorch 보다 느렸다. NCNN 은 ARM NEON 에 최적화된
라이브러리라 x86 에서는 MKL 을 쓰는 PyTorch 에 밀린다. 파이(Cortex-A76)에서는
뒤집힐 것으로 보지만 **확인 전에는 추측이다.** 이 스크립트로 재고 결정한다.

PC 실측 (참고용, 1920x1080 입력)
    .pt  @1024   81.1 ms
    .pt  @640    36.7 ms
    NCNN @640    85.7 ms
    NCNN @1024  248.8 ms

사용:
    python bench_pi.py
    python bench_pi.py --repeat 10 --source test.mp4
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

CANDIDATES: list[tuple[str, str, int]] = [
    (".pt  @640", "MCN.pt", 640),
    (".pt  @1024", "MCN.pt", 1024),
    ("NCNN @640", "MCN_640_ncnn_model", 640),
    ("NCNN @1024", "MCN_1024_ncnn_model", 1024),
]


def make_frame(source: str | None, w: int, h: int) -> np.ndarray:
    """실제 영상 한 장이 있으면 그걸 쓴다. 빈 프레임은 후처리(NMS) 비용이
    빠져서 낙관적으로 나온다."""
    if source and Path(source).exists():
        import cv2

        cap = cv2.VideoCapture(source)
        ok, frame = cap.read()
        cap.release()
        if ok:
            return frame
        print(f"[경고] {source} 를 읽지 못해 합성 프레임으로 대체합니다")
    return np.zeros((h, w, 3), np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--source", default="test.mp4", help="실제 프레임을 쓸 영상")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    args = ap.parse_args()

    from signal_model import ModelSignalDetector

    frame = make_frame(args.source, args.width, args.height)
    print(f"입력 {frame.shape[1]}x{frame.shape[0]}, {args.repeat}회 평균\n")
    print(f"{'백엔드':>12} {'imgsz':>6} {'분류기':>7} {'ms/프레임':>11} {'단독 FPS':>10}")

    results: list[tuple[str, float]] = []
    for name, weights, imgsz in CANDIDATES:
        if not Path(weights).exists():
            print(f"{name:>12} {'':>6} {'':>7} {'파일 없음 — 건너뜀':>11}")
            continue
        try:
            d = ModelSignalDetector(det_weights=weights, imgsz=imgsz)
            d(frame)                                   # 워밍업
            t = time.perf_counter()
            for _ in range(args.repeat):
                d(frame)
            ms = (time.perf_counter() - t) / args.repeat * 1000
            kind = "ONNX" if type(d.cls).__name__ == "_OnnxClassifier" else "Torch"
            print(f"{name:>12} {d.imgsz:6d} {kind:>7} {ms:10.1f}  {1000 / ms:9.1f}")
            results.append((f"{weights} @{d.imgsz}", ms))
        except Exception as e:                          # noqa: BLE001
            print(f"{name:>12} 실패: {type(e).__name__}: {e}")

    if not results:
        return
    best, best_ms = min(results, key=lambda r: r[1])
    print(f"\n가장 빠름: {best}  ({best_ms:.1f} ms)")
    print("\nconfig.py 에 반영할 것:")
    w, sz = best.split(" @")
    print(f"    SIGNAL_DET_WEIGHTS = {w!r}")
    print(f"    SIGNAL_DET_IMGSZ   = {sz}")
    print("\n★ 속도만으로 정하지 말 것. 640 은 1024 보다 검출률이 낮다.")
    print("  얼마나 낮은지는 아직 측정되지 않았다.")


if __name__ == "__main__":
    main()
