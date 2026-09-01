"""보행 신호등(pedestrian traffic_light) 크롭 추출.

AI Hub 056 신호등/도로표지판 데이터의 라벨 tar 을 읽어 보행등 박스만 골라내고,
원천 tar 을 한 번만 순차 스트리밍하면서 해당 이미지를 디코드해 크롭을 저장한다.
18GB 원천 tar 을 디스크에 풀지 않는 것이 핵심 (필요한 이미지가 전체의 30% 미만).

라벨 스키마 (실측 확인):
    {"light_count":"2", "box":[x0,y0,x1,y1], "type":"pedestrian",
     "class":"traffic_light", "direction":"vertical",
     "attribute":[{"red":"on","green":"off","yellow":"off",
                   "left_arrow":"off","x_light":"off","others_arrow":"off"}]}

상태 라벨:
    red  : red 만 on
    green: green 만 on
    off  : 전부 off  (원거리라 판독 불가한 경우가 대부분 → UNKNOWN 역할)
    그 외 조합(x_light/yellow 등)은 표본이 극소수라 버린다.

사용:
    python extract_ped_crops.py --dry-run
    python extract_ped_crops.py --out crops
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

LABELS = ("red", "green", "off")

SPLITS: dict[str, tuple[Path, Path]] = {
    "train": (Path("Training/[라벨]m_train_1920_1080_daylight_1.tar"),
              Path("Training/[원천]m_train_1920_1080_daylight_1.tar")),
    "val":   (Path("Validation/[라벨]m_validation_1920_1080_daylight_1.tar"),
              Path("Validation/[원천]m_validation_1920_1080_daylight_1.tar")),
}


@dataclass(frozen=True)
class Box:
    x0: int
    y0: int
    x1: int
    y1: int
    label: str

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def w(self) -> int:
        return self.x1 - self.x0


def _state_of(attr: dict[str, str]) -> str | None:
    """attribute dict → 상태 문자열. 애매한 조합은 None(버림)."""
    on = sorted(k for k, v in attr.items() if v == "on")
    if not on:
        return "off"
    if on == ["red"]:
        return "red"
    if on == ["green"]:
        return "green"
    return None


def collect_boxes(label_tar: Path, min_h: int) -> dict[str, list[Box]]:
    """라벨 tar → {이미지파일명: [Box, ...]}. 보행등만, 높이 필터 적용."""
    out: dict[str, list[Box]] = {}
    with tarfile.open(label_tar, "r|") as tf:
        for member in tf:
            if not member.name.endswith(".json"):
                continue
            fp = tf.extractfile(member)
            if fp is None:
                continue
            doc = json.loads(fp.read().decode("utf-8"))
            fname = doc.get("image", {}).get("filename")
            if not fname:
                continue
            boxes: list[Box] = []
            for ann in doc.get("annotation", []):
                if ann.get("class") != "traffic_light" or ann.get("type") != "pedestrian":
                    continue
                attrs = ann.get("attribute") or []
                if not attrs:
                    continue
                state = _state_of(attrs[0])
                if state is None:
                    continue
                x0, y0, x1, y1 = (int(v) for v in ann["box"])
                if y1 - y0 < min_h or x1 - x0 <= 0:
                    continue
                boxes.append(Box(x0, y0, x1, y1, state))
            if boxes:
                out[fname] = boxes
    return out


def thin_off(index: dict[str, list[Box]], max_off: int, seed: int) -> dict[str, list[Box]]:
    """off 표본이 red/green 을 압도하므로 무작위로 상한까지 줄인다."""
    offs = [(fn, i) for fn, bs in index.items() for i, b in enumerate(bs) if b.label == "off"]
    if len(offs) <= max_off:
        return index
    rng = random.Random(seed)
    drop = set(rng.sample(offs, len(offs) - max_off))
    thinned: dict[str, list[Box]] = {}
    for fn, bs in index.items():
        kept = [b for i, b in enumerate(bs) if (fn, i) not in drop]
        if kept:
            thinned[fn] = kept
    return thinned


def crop_with_pad(img: np.ndarray, box: Box, pad: float) -> np.ndarray:
    """함체 주변에 여백을 조금 남긴다. 함체 테두리가 보여야 위/아래 점등 위치를 배운다."""
    h, w = img.shape[:2]
    px, py = int(round(box.w * pad)), int(round(box.h * pad))
    x0, y0 = max(0, box.x0 - px), max(0, box.y0 - py)
    x1, y1 = min(w, box.x1 + px), min(h, box.y1 + py)
    return img[y0:y1, x0:x1]


def run_split(split: str, out_root: Path, min_h: int, pad: float,
              max_off: int, seed: int, dry_run: bool) -> Counter:
    label_tar, image_tar = SPLITS[split]
    for p in (label_tar, image_tar):
        if not p.exists():
            raise SystemExit(f"[{split}] tar 없음: {p}")

    print(f"[{split}] 라벨 스캔 중… {label_tar.name}")
    index = collect_boxes(label_tar, min_h)
    raw = Counter(b.label for bs in index.values() for b in bs)
    index = thin_off(index, max_off, seed)
    plan = Counter(b.label for bs in index.values() for b in bs)
    print(f"[{split}] 이미지 {len(index)}장, 박스 {sum(plan.values())}개")
    print(f"[{split}] 원본 {dict(raw)} → off 상한 적용 후 {dict(plan)}")
    if dry_run:
        return plan

    for lab in LABELS:
        (out_root / split / lab).mkdir(parents=True, exist_ok=True)
    manifest = out_root / f"{split}_manifest.csv"

    saved: Counter = Counter()
    seen = 0
    print(f"[{split}] 원천 스트리밍 중… {image_tar.name} (한 번만 순차 읽기)")
    with tarfile.open(image_tar, "r|") as tf, manifest.open("w", newline="", encoding="utf-8") as mf:
        writer = csv.writer(mf)
        writer.writerow(["path", "src_image", "label", "x0", "y0", "x1", "y1", "w", "h"])
        for member in tf:
            name = Path(member.name).name
            boxes = index.get(name)
            if not boxes:
                continue
            fp = tf.extractfile(member)
            if fp is None:
                continue
            img = cv2.imdecode(np.frombuffer(fp.read(), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            for i, box in enumerate(boxes):
                patch = crop_with_pad(img, box, pad)
                if patch.size == 0:
                    continue
                dst = out_root / split / box.label / f"{Path(name).stem}_{i}.png"
                cv2.imwrite(str(dst), patch)
                writer.writerow([dst.as_posix(), name, box.label,
                                 box.x0, box.y0, box.x1, box.y1, box.w, box.h])
                saved[box.label] += 1
            seen += 1
            if seen % 500 == 0:
                print(f"  … {seen}/{len(index)}장 처리, 크롭 {sum(saved.values())}개")
    print(f"[{split}] 완료: {dict(saved)}  → {out_root / split}")
    print(f"[{split}] 매니페스트: {manifest}")
    return saved


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="crops", help="출력 루트")
    ap.add_argument("--splits", nargs="+", default=["train", "val"], choices=list(SPLITS))
    ap.add_argument("--min-h", type=int, default=20,
                    help="이보다 낮은 박스는 버린다(px). 20 미만은 색 판독이 물리적으로 불가")
    ap.add_argument("--pad", type=float, default=0.15, help="함체 주변 여백 비율")
    ap.add_argument("--max-off", type=int, default=3000, help="split 당 off 표본 상한")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="라벨만 세고 이미지는 건드리지 않음")
    args = ap.parse_args()

    out_root = Path(args.out)
    total: Counter = Counter()
    for split in args.splits:
        total += run_split(split, out_root, args.min_h, args.pad,
                           args.max_off, args.seed, args.dry_run)

    print("\n===== 합계 =====")
    for lab in LABELS:
        print(f"  {lab:5s} {total[lab]:6d}")
    if total["green"]:
        print(f"  red/green 불균형 = {total['red'] / total['green']:.1f} : 1")


if __name__ == "__main__":
    main()
