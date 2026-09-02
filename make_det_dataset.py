"""보행 신호등 검출용 YOLO 데이터셋 생성.

AI Hub 056 라벨/원천 tar → `TRAIN_ROI` 크롭 + YOLO txt 라벨.

설계 근거
  - 전체 프레임을 640 정사각으로 줄이면 중앙값 18x37px 보행등이 6x12px 이 되어
    검출 한계선에 걸린다. 그래서 가로로 긴 띠를 잘라 직사각 입력으로 학습한다.
  - 학습 크롭은 배포용 `config.SIGNAL_ROI` 와 다르다. 이유는 TRAIN_ROI 주석 참조.
  - 3클래스로 만든다. ped_light 만 1클래스로 학습하면 생김새가 거의 같은
    차량 신호등이 "배경"이 되어 학습 신호가 충돌한다. car_light / traffic_sign 을
    명시적 hard negative 로 준다. 특히 traffic_sign 은 프로토타입이 실패한
    "신호등 옆 빨간 간판" 오탐을 모델이 직접 배우게 한다.
    추론 시에는 ped_light 만 사용한다.
  - AI Hub 기본 train/val split 을 그대로 쓴다. 프레임 번호 교집합 0,
    ±2 이내 인접 프레임도 0.5% 뿐이라 연속 프레임 누수가 사실상 없다.

사용:
    python make_det_dataset.py --dry-run
    python make_det_dataset.py --out dataset_det
"""

from __future__ import annotations

import argparse
import json
import random
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import config

NAMES = ("ped_light", "car_light", "traffic_sign")
PED, CAR, SIGN = 0, 1, 2

# ★ config.SIGNAL_ROI 를 쓰지 않는다. 그것은 보행자 웨어러블 시점 기준의 배포
#   파라미터이고, 학습 데이터는 차량 주행 시점이라 보행등이 훨씬 아래에 있다.
#   실측: ped_light 중심 y 의 중앙값 0.594, 99% 가 0.798 이하.
#   SIGNAL_ROI(상단 52%)로 자르면 양성 샘플의 83% 가 날아간다.
#
#   검출기가 배우는 것은 "보행등이 어떻게 생겼는지"이지 "화면 어디에 있는지"가
#   아니므로, 학습 크롭과 배포 ROI 는 달라도 된다. 맞춰야 하는 것은 픽셀 스케일이다.
#     학습 1920x594 → imgsz 1024 rect → 배율 0.53, 보행등 약 10x20px
#     배포 1612x561 → imgsz 1024 rect → 배율 0.64, 보행등 약 12x24px
#   남는 차이는 YOLO 의 scale 증강이 흡수한다.
TRAIN_ROI = (0.00, 0.30, 1.00, 0.85)

SPLITS: dict[str, tuple[Path, Path]] = {
    "train": (Path("Training/[라벨]m_train_1920_1080_daylight_1.tar"),
              Path("Training/[원천]m_train_1920_1080_daylight_1.tar")),
    "val":   (Path("Validation/[라벨]m_validation_1920_1080_daylight_1.tar"),
              Path("Validation/[원천]m_validation_1920_1080_daylight_1.tar")),
}


@dataclass(frozen=True)
class DetBox:
    cls: int
    x0: float
    y0: float
    x1: float
    y1: float


def class_of(ann: dict) -> int | None:
    kind = ann.get("class")
    if kind == "traffic_sign":
        return SIGN
    if kind != "traffic_light":
        return None
    t = ann.get("type")
    if t == "pedestrian":
        return PED
    if t in ("car", "bus"):
        return CAR
    return None


def roi_rect(w: int, h: int, roi: tuple[float, float, float, float]
             ) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = roi
    return int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)


def clip_to_roi(box: DetBox, rx0: int, ry0: int, rx1: int, ry1: int,
                keep_ratio: float, min_px: int) -> DetBox | None:
    """ROI 로 잘라 좌표계를 옮긴다. 잘려나간 비율이 크면 버린다."""
    area = max((box.x1 - box.x0) * (box.y1 - box.y0), 1e-6)
    cx0, cy0 = max(box.x0, rx0), max(box.y0, ry0)
    cx1, cy1 = min(box.x1, rx1), min(box.y1, ry1)
    if cx1 - cx0 < min_px or cy1 - cy0 < min_px:
        return None
    if (cx1 - cx0) * (cy1 - cy0) / area < keep_ratio:
        return None
    return DetBox(box.cls, cx0 - rx0, cy0 - ry0, cx1 - rx0, cy1 - ry0)


def collect(label_tar: Path, roi: tuple[float, float, float, float],
            keep_ratio: float, min_px: int
            ) -> tuple[dict[str, list[DetBox]], Counter, Counter]:
    """{이미지명: [ROI 좌표계 박스]}, ROI 통과 수, 원본 수."""
    index: dict[str, list[DetBox]] = {}
    kept: Counter = Counter()
    total: Counter = Counter()
    with tarfile.open(label_tar, "r|") as tf:
        for member in tf:
            if not member.name.endswith(".json"):
                continue
            fp = tf.extractfile(member)
            if fp is None:
                continue
            doc = json.loads(fp.read().decode("utf-8"))
            fname = doc.get("image", {}).get("filename")
            size = doc.get("image", {}).get("imsize") or [1920, 1080]
            if not fname:
                continue
            rx0, ry0, rx1, ry1 = roi_rect(int(size[0]), int(size[1]), roi)
            boxes: list[DetBox] = []
            for ann in doc.get("annotation", []):
                c = class_of(ann)
                if c is None:
                    continue
                total[NAMES[c]] += 1
                x0, y0, x1, y1 = (float(v) for v in ann["box"])
                cb = clip_to_roi(DetBox(c, x0, y0, x1, y1),
                                 rx0, ry0, rx1, ry1, keep_ratio, min_px)
                if cb is not None:
                    boxes.append(cb)
                    kept[NAMES[c]] += 1
            index[fname] = boxes
    return index, kept, total


def choose_images(index: dict[str, list[DetBox]], max_neg: int,
                  seed: int) -> dict[str, list[DetBox]]:
    """ped_light 가 있는 이미지는 전부, 없는 이미지는 상한까지만 (배경 학습용)."""
    pos = {k: v for k, v in index.items() if any(b.cls == PED for b in v)}
    neg = [k for k, v in index.items() if k not in pos and v]
    rng = random.Random(seed)
    rng.shuffle(neg)
    out = dict(pos)
    for k in neg[:max_neg]:
        out[k] = index[k]
    return out


def write_labels(boxes: list[DetBox], w: int, h: int, dst: Path) -> None:
    lines = []
    for b in boxes:
        cx, cy = (b.x0 + b.x1) / 2 / w, (b.y0 + b.y1) / 2 / h
        bw, bh = (b.x1 - b.x0) / w, (b.y1 - b.y0) / h
        lines.append(f"{b.cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    dst.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def run_split(split: str, out: Path, roi: tuple[float, float, float, float],
              keep_ratio: float, min_px: int, max_neg: int, seed: int,
              quality: int, dry_run: bool) -> Counter:
    label_tar, image_tar = SPLITS[split]
    for p in (label_tar, image_tar):
        if not p.exists():
            raise SystemExit(f"[{split}] tar 없음: {p}")

    print(f"[{split}] 라벨 스캔 중… {label_tar.name}")
    index, kept, total = collect(label_tar, roi, keep_ratio, min_px)
    print(f"[{split}] ROI 통과 박스 (원본 대비)")
    for n in NAMES:
        pct = 100 * kept[n] / total[n] if total[n] else 0.0
        print(f"    {n:14s} {kept[n]:6d} / {total[n]:6d} = {pct:5.1f}%")

    chosen = choose_images(index, max_neg, seed)
    n_pos = sum(1 for v in chosen.values() if any(b.cls == PED for b in v))
    print(f"[{split}] 이미지 {len(chosen)}장 "
          f"(ped_light 포함 {n_pos}, 배경용 {len(chosen) - n_pos})")
    if dry_run:
        return Counter({NAMES[b.cls]: 1 for v in chosen.values() for b in v})

    img_dir = out / "images" / split
    lab_dir = out / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    written: Counter = Counter()
    seen = 0
    print(f"[{split}] 원천 스트리밍 중… {image_tar.name}")
    with tarfile.open(image_tar, "r|") as tf:
        for member in tf:
            name = Path(member.name).name
            if name not in chosen:
                continue
            fp = tf.extractfile(member)
            if fp is None:
                continue
            img = cv2.imdecode(np.frombuffer(fp.read(), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            rx0, ry0, rx1, ry1 = roi_rect(w, h, roi)
            crop = img[ry0:ry1, rx0:rx1]
            if crop.size == 0:
                continue
            stem = Path(name).stem
            cv2.imwrite(str(img_dir / f"{stem}.jpg"), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, quality])
            boxes = chosen[name]
            write_labels(boxes, crop.shape[1], crop.shape[0], lab_dir / f"{stem}.txt")
            for b in boxes:
                written[NAMES[b.cls]] += 1
            seen += 1
            if seen % 1000 == 0:
                print(f"  … {seen}/{len(chosen)}장")
    print(f"[{split}] 완료: {seen}장, 박스 {dict(written)}")
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset_det")
    ap.add_argument("--roi", nargs=4, type=float, default=list(TRAIN_ROI),
                    metavar=("X0", "Y0", "X1", "Y1"),
                    help="학습 크롭 영역(비율). 배포용 config.SIGNAL_ROI 와 별개다")
    ap.add_argument("--splits", nargs="+", default=["train", "val"], choices=list(SPLITS))
    ap.add_argument("--keep-ratio", type=float, default=0.6,
                    help="ROI 로 자른 뒤 남은 면적 비율이 이보다 작으면 버린다")
    ap.add_argument("--min-px", type=int, default=4, help="ROI 내 최소 변 길이")
    ap.add_argument("--max-neg", type=int, default=2000,
                    help="split 당 ped_light 없는 이미지 상한 (배경 학습용)")
    ap.add_argument("--quality", type=int, default=92, help="JPEG 품질")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    roi = tuple(args.roi)
    x0, y0, x1, y1 = roi
    rw = int((x1 - x0) * config.MAIN_SIZE[0])
    rh = int((y1 - y0) * config.MAIN_SIZE[1])
    print(f"학습 크롭 ROI = {roi} → {rw}x{rh} (종횡비 {rw / rh:.2f})")
    print(f"  (배포용 config.SIGNAL_ROI = {config.SIGNAL_ROI} 는 건드리지 않는다)")

    total: Counter = Counter()
    for split in args.splits:
        total += run_split(split, out, roi, args.keep_ratio, args.min_px,
                           args.max_neg, args.seed, args.quality, args.dry_run)

    if not args.dry_run:
        yaml = (f"path: {out.resolve().as_posix()}\n"
                f"train: images/train\n"
                f"val: images/val\n"
                f"nc: {len(NAMES)}\n"
                f"names: {list(NAMES)}\n")
        (out / "data.yaml").write_text(yaml, encoding="utf-8")
        print(f"\ndata.yaml 작성: {out / 'data.yaml'}")
        print(yaml)
    print("===== 박스 합계 =====", dict(total))


if __name__ == "__main__":
    main()
