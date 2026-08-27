"""
FOCAL_PX 1회 캘리브레이션.

거리 추정 D = (H_real * f) / h_px 에서 f 를 실측한다.
이 값이 틀리면 모든 거리 안내가 통째로 틀리므로 반드시 먼저 하고 시작할 것.

방법
  1. 키를 아는 사람이 카메라에서 정확히 5.0m 떨어져 정면으로 선다
  2. python calibrate.py --source pi --dist 5.0 --height 1.70
  3. 안정되면 스페이스바 → f 값 출력
  4. 출력된 값을 config.FOCAL_PX 에 적는다

주의: f 는 lores(416) 좌표계 기준이다. 스트림 해상도를 바꾸면 다시 재야 한다.
  f_new = f_old * (new_height / old_height)
"""

from __future__ import annotations

import argparse
import statistics

import cv2

import config
from sources import open_source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="pi")
    ap.add_argument("--dist", type=float, required=True, help="실제 거리(m)")
    ap.add_argument("--height", type=float, default=1.70, help="대상 실제 높이(m)")
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--label", default="person")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)

    src = open_source(args.source)
    samples: list[float] = []
    print("대상이 화면에 잡히면 스페이스바로 샘플 수집, q 로 종료")

    try:
        for _, lores, _ in src:
            res = model(lores, imgsz=config.LORES_SIZE[0], conf=0.4, verbose=False)
            vis = lores.copy()
            box_h = None

            if res and res[0].boxes is not None and len(res[0].boxes):
                names = res[0].names
                cands = [b for b in res[0].boxes
                         if names[int(b.cls[0])] == args.label]
                if cands:
                    b = max(cands, key=lambda b: b.xyxy[0][3] - b.xyxy[0][1])
                    x1, y1, x2, y2 = (int(v) for v in b.xyxy[0])
                    box_h = y2 - y1
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    f_now = box_h * args.dist / args.height
                    cv2.putText(vis, f"h={box_h}px  f={f_now:.1f}", (8, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(vis, f"n={len(samples)}", (8, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.imshow("calibrate", vis)
            k = cv2.waitKey(1) & 0xFF

            if k == ord(" ") and box_h:
                samples.append(box_h * args.dist / args.height)
                print(f"  샘플 {len(samples)}: f={samples[-1]:.1f}")
            elif k == ord("q"):
                break
    finally:
        src.close()
        cv2.destroyAllWindows()

    if len(samples) < 3:
        print("샘플이 3개 미만입니다. 다시 측정하세요.")
        return

    f = statistics.median(samples)
    sd = statistics.pstdev(samples)
    print("\n" + "=" * 46)
    print(f"  FOCAL_PX = {f:.1f}      (표준편차 {sd:.1f}, n={len(samples)})")
    print("  config.py 의 FOCAL_PX 에 이 값을 적으세요")
    if sd > f * 0.08:
        print("  ※ 편차가 큽니다. 대상이 흔들렸을 수 있으니 재측정 권장")
    print("=" * 46)


if __name__ == "__main__":
    main()
