"""보행 신호등 크롭 3클래스(red / green / off) 분류기 학습.

`extract_ped_crops.py` 가 만든 crops/ 와 manifest CSV 를 입력으로 받는다.

설계 근거
  - 3클래스: off 를 버리지 않고 UNKNOWN 역할로 남긴다. 원거리·판독불가 표본이
    다수라 모델이 애매할 때 off 로 기울고, 이 도메인에서는 그 편향 방향이 맞다.
  - 원본 이미지 단위로 재분할: 같은 프레임에서 나온 크롭이 train/val 양쪽에
    걸치면 누수다. AI Hub 기본 split 은 green 분포가 뒤틀려 있어(val 466 > train 205)
    그대로 쓰지 않는다.
  - 상하 반전 금지: 보행등은 위=적 / 아래=녹 이라 vflip 은 라벨을 뒤집는다.
  - 높이 상한 150px: 그보다 큰 크롭은 차량이 신호등 바로 아래를 지날 때 찍힌
    것이라 함체가 비틀려 있다. 보행자 시점과 구도가 다르다.

사용:
    python train_signal_cls.py --epochs 20
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

CLASSES = ("red", "green", "off")
CLS_IDX = {c: i for i, c in enumerate(CLASSES)}


def load_manifests(root: Path, min_h: int, max_h: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for split in ("train", "val"):
        mf = root / f"{split}_manifest.csv"
        if not mf.exists():
            continue
        with mf.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if min_h <= int(r["h"]) <= max_h:
                    rows.append(r)
    if not rows:
        raise SystemExit(f"manifest 없음 또는 조건에 맞는 크롭 없음: {root}")
    return rows


def split_by_image(rows: list[dict[str, str]], val_ratio: float,
                   seed: int) -> tuple[list[dict], list[dict]]:
    """원본 이미지 단위로 나눈다. 크롭 단위로 나누면 같은 프레임이 양쪽에 샌다."""
    by_img: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_img[r["src_image"]].append(r)
    imgs = sorted(by_img)
    random.Random(seed).shuffle(imgs)
    n_val = int(len(imgs) * val_ratio)
    val_imgs = set(imgs[:n_val])
    tr = [r for im in imgs if im not in val_imgs for r in by_img[im]]
    va = [r for im in val_imgs for r in by_img[im]]
    return tr, va


class CropDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], tf: transforms.Compose):
        self.rows = rows
        self.tf = tf

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        img = Image.open(r["path"]).convert("RGB")
        return self.tf(img), CLS_IDX[r["label"]], int(r["h"])


def build_transforms(w: int, h: int) -> tuple[transforms.Compose, transforms.Compose]:
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.RandomAffine(degrees=7, translate=(0.06, 0.06), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.2),
        transforms.RandomApply([transforms.GaussianBlur(3, (0.1, 1.5))], p=0.3),
        transforms.RandomHorizontalFlip(),   # 좌우는 안전. 상하 반전은 절대 금지
        transforms.ToTensor(),
        norm,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        norm,
    ])
    return train_tf, eval_tf


def build_model(arch: str, pretrained: bool) -> nn.Module:
    if arch == "mobilenet_v3_small":
        w = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.mobilenet_v3_small(weights=w)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(CLASSES))
        return m
    if arch == "small_cnn":
        def blk(i: int, o: int) -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(2))
        return nn.Sequential(blk(3, 16), blk(16, 32), blk(32, 64), blk(64, 96),
                             nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                             nn.Dropout(0.2), nn.Linear(96, len(CLASSES)))
    raise SystemExit(f"모르는 arch: {arch}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device
             ) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    model.eval()
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    bins: dict[str, tuple[int, int]] = {
        "20-29": (0, 0), "30-49": (0, 0), "50-99": (0, 0), "100+": (0, 0)}
    for x, y, h in loader:
        pred = model(x.to(device)).argmax(1).cpu()
        for t, p, hh in zip(y.tolist(), pred.tolist(), h.tolist()):
            cm[t, p] += 1
            k = "20-29" if hh < 30 else "30-49" if hh < 50 else "50-99" if hh < 100 else "100+"
            ok, tot = bins[k]
            bins[k] = (ok + int(t == p), tot + 1)
    return cm, bins


def macro_f1(cm: np.ndarray) -> float:
    f1s: list[float] = []
    for i in range(len(CLASSES)):
        prec = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rec = cm[i, i] / cm[i].sum() if cm[i].sum() else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


def report(cm: np.ndarray, bins: dict[str, tuple[int, int]]) -> None:
    print("\n혼동행렬  (행=정답, 열=예측)")
    print(f"{'':>7}" + "".join(f"{c:>8}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"{c:>7}" + "".join(f"{v:8d}" for v in cm[i]))

    print(f"\n{'클래스':>7} {'정밀도':>8} {'재현율':>8} {'F1':>8} {'표본':>7}")
    for i, c in enumerate(CLASSES):
        prec = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rec = cm[i, i] / cm[i].sum() if cm[i].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"{c:>7} {prec:8.3f} {rec:8.3f} {f1:8.3f} {cm[i].sum():7d}")
    print(f"{'macro F1':>7} {macro_f1(cm):26.3f}")

    r, g = CLS_IDX["red"], CLS_IDX["green"]
    print("\n[안전 지표] 적색을 녹색이라고 한 비율 — 사고로 직결")
    print(f"    red -> green : {cm[r, g]:4d} / {cm[r].sum():4d} = "
          f"{100 * cm[r, g] / max(cm[r].sum(), 1):5.2f}%")
    print(f"    green -> red : {cm[g, r]:4d} / {cm[g].sum():4d} = "
          f"{100 * cm[g, r] / max(cm[g].sum(), 1):5.2f}%   (불편하지만 안전)")

    print("\n크롭 높이별 정확도")
    for k, (ok, tot) in bins.items():
        if tot:
            print(f"    {k:>6}px  {ok:5d}/{tot:5d} = {100 * ok / tot:5.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="crops")
    ap.add_argument("--arch", default="mobilenet_v3_small",
                    choices=["mobilenet_v3_small", "small_cnn"])
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--height", type=int, default=128)
    ap.add_argument("--min-h", type=int, default=20)
    ap.add_argument("--max-h", type=int, default=150)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/signal_cls.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    rows = load_manifests(Path(args.crops), args.min_h, args.max_h)
    tr_rows, va_rows = split_by_image(rows, args.val_ratio, args.seed)
    print(f"크롭 {len(rows)}개 (높이 {args.min_h}~{args.max_h}px)")
    print(f"  train {len(tr_rows):5d}  {dict(Counter(r['label'] for r in tr_rows))}")
    print(f"  val   {len(va_rows):5d}  {dict(Counter(r['label'] for r in va_rows))}")

    train_tf, eval_tf = build_transforms(args.width, args.height)
    dl_tr = DataLoader(CropDataset(tr_rows, train_tf), batch_size=args.batch,
                       shuffle=True, num_workers=args.workers, drop_last=True)
    dl_va = DataLoader(CropDataset(va_rows, eval_tf), batch_size=args.batch,
                       shuffle=False, num_workers=args.workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    model = build_model(args.arch, not args.no_pretrained).to(device)

    counts = Counter(r["label"] for r in tr_rows)
    weight = torch.tensor([len(tr_rows) / (len(CLASSES) * max(counts[c], 1))
                           for c in CLASSES], dtype=torch.float32, device=device)
    print(f"클래스 가중치 {dict(zip(CLASSES, [round(float(w), 2) for w in weight]))}")

    crit = nn.CrossEntropyLoss(weight=weight, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = -1.0
    best_cm: np.ndarray | None = None
    best_bins: dict[str, tuple[int, int]] | None = None
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        model.train()
        tot = 0.0
        for x, y, _ in dl_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tot += float(loss) * x.size(0)
        sched.step()

        cm, bins = evaluate(model, dl_va, device)
        macro = macro_f1(cm)
        r, g = CLS_IDX["red"], CLS_IDX["green"]
        r2g = 100 * cm[r, g] / max(cm[r].sum(), 1)
        print(f"ep {ep:3d}  loss {tot / len(tr_rows):.4f}  macroF1 {macro:.3f}  "
              f"red->green {r2g:5.2f}%")

        if macro > best:
            best, best_cm, best_bins = macro, cm, bins
            torch.save({"model": model.state_dict(), "arch": args.arch,
                        "classes": CLASSES, "size": (args.width, args.height),
                        "macro_f1": macro}, out)

    print(f"\n===== 최고 성적 (macro F1 {best:.3f}) =====")
    if best_cm is not None and best_bins is not None:
        report(best_cm, best_bins)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
