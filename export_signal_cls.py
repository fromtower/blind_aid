"""신호등 상태 분류기(MobileNetV3-Small)를 ONNX 로 내보낸다.

검출기는 ultralytics 가 NCNN 으로 내보내지만 분류기는 그 경로를 못 탄다.
ONNX + onnxruntime 으로 가면 **라즈베리파이에 torch 를 설치할 필요가 없다.**
(aarch64 torch 휠은 수백 MB 에 빌드도 느리다)

클래스 순서와 입력 크기는 체크포인트에서 읽어 metadata 로 함께 저장한다.
코드가 순서를 하드코딩하면 나중에 재학습했을 때 조용히 틀린다.

사용:
    python export_signal_cls.py
    python export_signal_cls.py --ckpt models/signal_cls.pt --out models/signal_cls.onnx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import models


def build(num_classes: int) -> nn.Module:
    net = models.mobilenet_v3_small(weights=None)
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, num_classes)
    return net


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/signal_cls.pt")
    ap.add_argument("--out", default="models/signal_cls.onnx")
    ap.add_argument("--opset", type=int, default=12,
                    help="onnxruntime 구버전 호환을 위해 낮게 잡는다")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    classes = list(ck["classes"])
    w, h = ck["size"]
    print(f"체크포인트: 클래스={classes} 입력={w}x{h} macroF1={ck.get('macro_f1'):.3f}")

    net = build(len(classes))
    net.load_state_dict(ck["model"])
    net.eval()

    dummy = torch.randn(1, 3, h, w)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        net, dummy, str(out),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset, do_constant_folding=True,
    )

    # 클래스 순서·입력 크기·정규화 상수를 같이 저장한다.
    # 추론 코드가 이 값을 읽게 해서 하드코딩으로 어긋나는 일을 막는다.
    meta = {
        "classes": classes,
        "size": [w, h],
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "macro_f1": float(ck.get("macro_f1", 0.0)),
    }
    meta_path = out.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # 검증: PyTorch 와 ONNX 출력이 일치하는지 확인한다.
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        x = np.random.randn(1, 3, h, w).astype(np.float32)
        with torch.no_grad():
            ref = net(torch.from_numpy(x)).numpy()
        got = sess.run(None, {"input": x})[0]
        diff = float(np.abs(ref - got).max())
        print(f"PyTorch vs ONNX 최대 오차 {diff:.2e} "
              f"{'— 일치' if diff < 1e-3 else '— ★ 불일치, 확인 필요'}")
    except ImportError:
        print("onnxruntime 미설치 → 출력 대조 생략 (파이에서 설치 후 확인할 것)")

    print(f"\n저장: {out} ({out.stat().st_size / 1e6:.1f}MB)")
    print(f"      {meta_path}")


if __name__ == "__main__":
    main()
