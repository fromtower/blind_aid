"""보행 신호등 검출기 학습 실행 스크립트.

실행:
    python run_train_det.py

설정을 여기에 고정해 두고 `train_det.py` 를 호출한다. 화면 출력과 로그 파일에
동시에 기록하므로, 밤새 돌린 뒤 아침에 로그만 봐도 무슨 일이 있었는지 알 수 있다.

설정 근거 (GTX 1060 3GB 실측, 400장 서브셋 → 전체 6,230장 환산, 전부 amp=True)
    imgsz  batch   전체/에폭   peak VRAM
      640      8      10.8분      1.19GB   ← 채택
      640      4      13.3분      0.68GB
     1024      4      27.0분      1.48GB
     1024      6      67.8분      2.19GB   스래싱
      640     16          —       2.5GB    스래싱
가용 VRAM 한계는 약 2.0GB (디스플레이가 660MB 상시 점유).

25에폭 x 약 12분(검증 포함) = 약 5시간 예상.
best.pt 는 매 에폭 갱신되므로 중간에 Ctrl+C 로 멈춰도 결과가 남는다.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent

EPOCHS = 25
IMGSZ = 640
BATCH = 8
WORKERS = 2
PATIENCE = 15

DATA = "dataset_det_1024/data.yaml"
FULLVAL = "dataset_det_1024/data_fullval.yaml"
PROJECT = "runs/det"
NAME = "ped_light"


def main() -> int:
    # 이 스크립트는 설정을 고정해 두고 바로 학습을 시작한다.
    # 인자를 붙여 부르면 실수로 학습이 도는 것을 막고 사용법만 보여준다.
    if len(sys.argv) > 1:
        print(__doc__)
        print("이 스크립트는 인자를 받지 않습니다. 설정을 바꾸려면 파일 상단의")
        print("EPOCHS / IMGSZ / BATCH 상수를 수정하거나, train_det.py 를 직접 부르세요:")
        print(f"  python train_det.py --epochs {EPOCHS} --imgsz {IMGSZ} --batch {BATCH}")
        return 0

    os.chdir(ROOT)

    if not (ROOT / DATA).exists():
        print(f"[오류] 데이터셋이 없습니다: {DATA}")
        print("       먼저 데이터셋을 만드세요:")
        print("         python make_det_dataset.py --out dataset_det")
        print("       그 다음 이미지를 1024폭으로 축소하세요.")
        return 1

    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"train_det_{datetime.now():%Y%m%d_%H%M%S}.log"

    cmd = [
        sys.executable, "-u", "train_det.py",
        "--data", DATA,
        "--epochs", str(EPOCHS),
        "--imgsz", str(IMGSZ),
        "--batch", str(BATCH),
        "--workers", str(WORKERS),
        "--patience", str(PATIENCE),
        "--project", PROJECT,
        "--name", NAME,
    ]

    bar = "=" * 60
    header = (
        f"{bar}\n"
        f" 보행 신호등 검출기 학습\n"
        f"   epochs={EPOCHS}  imgsz={IMGSZ}  batch={BATCH}\n"
        f"   로그: {log_path}\n"
        f"   예상 소요: 약 5시간\n"
        f"{bar}\n\n"
        f" 중단하려면 Ctrl+C. best.pt 는 매 에폭 저장되므로 결과는 남습니다.\n"
        f" 학습 중에는 브라우저 등 GPU 를 쓰는 프로그램을 닫아 두세요.\n"
        f" (디스플레이가 660MB 를 상시 점유합니다. 여기서 더 늘면 스래싱 구간입니다)\n\n"
    )

    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    t0 = time.time()
    rc = 0

    with log_path.open("w", encoding="utf-8") as log:
        log.write(header)
        sys.stdout.write(header)
        sys.stdout.flush()

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                env=env, encoding="utf-8", errors="replace", bufsize=1)
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            rc = proc.wait()
        except KeyboardInterrupt:
            print("\n[중단] Ctrl+C 감지 — 학습을 종료합니다.")
            proc.terminate()
            proc.wait()
            rc = 130

        weights = Path(PROJECT) / NAME / "weights"
        footer = (
            f"\n{bar}\n"
            f" 학습 종료 (총 {(time.time() - t0) / 3600:.2f}시간, 종료코드 {rc})\n"
            f"   가중치: {weights / 'best.pt'}\n"
            f"   로그  : {log_path}\n\n"
            f" 전체 val(4,947장)로 최종 평가:\n"
            f"   yolo val model={(weights / 'best.pt').as_posix()} "
            f"data={FULLVAL} imgsz={IMGSZ}\n"
            f"{bar}\n"
        )
        sys.stdout.write(footer)
        log.write(footer)

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
