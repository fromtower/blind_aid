"""보행 신호등 검출기 학습 (YOLO11n, 3클래스).

`make_det_dataset.py` 로 만든 데이터셋을 학습한다. 추론 시에는 ped_light 만 쓰고,
car_light / traffic_sign 은 명시적 hard negative 역할이다.

GTX 1060 3GB 실측 (400장 서브셋 1에폭 → 전체 6,230장 환산, 전부 amp=True):
    imgsz  batch   전체/에폭   peak VRAM
      640      8      10.8분      1.19GB
      640      4      13.3분      0.68GB
     1024      4      27.0분      1.48GB
     1024      6      67.8분      2.19GB   ← 스래싱
      640     16          —       2.5GB    ← 스래싱
가용 VRAM 한계는 약 2.0GB (디스플레이가 660MB 상시 점유).
amp=True 가 amp=False 보다 22% 빠르고 메모리도 36% 적다 (640/b4 동일 조건 비교).

★ Windows 는 fork 가 없어 DataLoader 워커를 spawn 으로 띄운다.
  __main__ 가드가 없으면 워커가 못 뜨고 학습이 사실상 멈춘다. 반드시 유지할 것.

사용:
    python train_det.py --epochs 25
    python train_det.py --resume runs/det/ped_light/weights/last.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset_det_1024/data.yaml")
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=15,
                    help="이만큼 개선이 없으면 조기 종료. 밤새 헛도는 것을 막는다")
    ap.add_argument("--project", default="runs/det")
    ap.add_argument("--name", default="ped_light")
    ap.add_argument("--resume", default=None, help="last.pt 경로를 주면 이어서 학습")
    args = ap.parse_args()

    t0 = time.time()
    ep_times: list[float] = []

    def on_epoch_end(trainer) -> None:
        ep_times.append(time.time() - t0 - sum(ep_times))
        n = len(ep_times)
        avg = sum(ep_times) / n
        left = (args.epochs - n) * avg
        print(f"\n[진행] {n}/{args.epochs} 에폭  "
              f"평균 {avg / 60:.1f}분/에폭  "
              f"예상 잔여 {left / 3600:.1f}시간\n", flush=True)

    model = YOLO(args.resume or args.weights)
    model.add_callback("on_fit_epoch_end", on_epoch_end)

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=args.workers,
        amp=True,            # 실측상 더 빠르고 메모리도 적다
        cache=False,         # 1920x594 를 RAM 캐시하면 20GB 넘는다
        rect=False,          # rect=True 는 셔플을 끈다. 우리 데이터는 종횡비가 전부 같아
                             # 정렬 이득이 없으므로 셔플을 지키는 쪽이 낫다
        patience=args.patience,
        project=args.project,
        name=args.name,
        exist_ok=True,
        resume=bool(args.resume),
        seed=0,
        plots=True,
        verbose=True,
    )

    out = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\n총 {(time.time() - t0) / 3600:.2f}시간")
    print(f"가중치: {out}")
    print("\n전체 val(4,947장)로 최종 평가:")
    print(f"  yolo val model={out} data=dataset_det_1024/data_fullval.yaml imgsz={args.imgsz}")


if __name__ == "__main__":
    main()
