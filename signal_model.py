"""보행 신호등 판정 — 모델 기반 (YOLO 검출 + CNN 분류).

기존 `signal_light.detect_signal_once` 와 동일한 시그니처를 갖는다.
    (main_bgr) -> (state, area_px)
따라서 `SignalVoter` 의 5프레임 투표·점멸 검출 로직은 그대로 재사용된다.

규칙 기반과의 차이
  - ROI 를 쓰지 않는다. 전체 프레임에서 검출한다.
    `config.SIGNAL_ROI`(상단 52%)는 차량 주행 데이터에서 보행등의 15% 밖에
    담지 못했다(중심 y 중앙값 0.594). 위치를 미리 제한하는 대신 검출기가
    형태로 찾는다. SIGNAL_ROI 는 규칙 기반 경로에만 남는다.
  - 색으로 검출하지 않는다. YOLO 가 함체를 찾고, 그 안에서 분류기가 상태를 본다.

검출기 실측 (AI Hub 056 val, IoU 0.5, conf 0.25)
    높이  0~10px  R=0.502     20~30px  R=0.784
         10~15px  R=0.522     30~50px  R=0.900
         15~20px  R=0.819     50px~    R=0.889
15px 미만은 사람 눈으로도 판독 불가라 놓쳐도 되는 대상이다.

분류기 실측 (val 1,993개, 원본 이미지 단위 분리)
    macro F1 0.926 / red→green 오분류 0.29% (2/682, 95% CI 0.08~1.06%)
    가림 실험에서 적색 램프를 가리면 red 만, 녹색 램프를 가리면 green 만
    선택적으로 붕괴 → 색이 아니라 함체 내 점등 위치를 보고 있다.

★ 위 수치는 전부 차량 주행 시점 데이터 기준이다. 보행자 눈높이에서의 성능은
  검증되지 않았다. 실사용 전 직접 촬영 데이터로 재측정할 것.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

import config

RED, GREEN, NONE = "red", "green", "none"

# 분류기 출력 순서. train_signal_cls.py 의 CLASSES 와 일치해야 한다.
_CLS_TO_STATE = {"red": RED, "green": GREEN, "off": NONE}


def _ncnn_imgsz(det_dir: Path) -> int | list[int] | None:
    """NCNN metadata.yaml 에서 내보낼 때 고정된 입력 크기를 읽는다.

    정사각이면 int, 직사각이면 [h, w] 를 준다. 직사각 모델은 카메라 비율(16:9)에
    맞춰 내보낸 것이라 레터박스 패딩이 사라진다. 1024x1024 는 16.6 GFLOPS 인데
    1024x576 은 9.39 GFLOPS — 검출 내용은 같고 연산만 56% 다.
    """
    meta = det_dir / "metadata.yaml"
    if not meta.exists():
        return None

    # ultralytics 는 두 형식을 다 쓴다:
    #   imgsz: [576, 1024]        (인라인)
    #   imgsz:\n- 576\n- 1024     (블록 리스트)
    lines = meta.read_text(encoding="utf-8").splitlines()
    nums: list[int] = []
    for i, line in enumerate(lines):
        if not line.strip().startswith("imgsz:"):
            continue
        tail = line.split(":", 1)[1]
        nums = [int(t) for t in tail.replace("[", " ").replace("]", " ")
                .replace(",", " ").split() if t.isdigit()]
        if not nums:                      # 블록 리스트면 다음 줄들을 읽는다
            for nxt in lines[i + 1:]:
                st = nxt.strip()
                if not st.startswith("-"):
                    break
                tok = st.lstrip("- ").strip()
                if tok.isdigit():
                    nums.append(int(tok))
        break

    if len(nums) >= 2 and nums[0] != nums[1]:
        return [nums[0], nums[1]]         # [h, w]
    return max(nums) if nums else None


@dataclass(frozen=True)
class Detection:
    """검출된 보행등 하나. 좌표는 원본 MAIN 프레임 기준."""

    x0: int
    y0: int
    x1: int
    y1: int
    conf: float

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    @property
    def area(self) -> int:
        return max(self.x1 - self.x0, 0) * max(self.h, 0)


class _OnnxClassifier:
    """onnxruntime 백엔드. 라즈베리파이에 torch 를 설치하지 않아도 된다."""

    def __init__(self, onnx_path: Path) -> None:
        import json

        import onnxruntime as ort

        meta = json.loads(onnx_path.with_suffix(".json").read_text(encoding="utf-8"))
        self.classes: tuple[str, ...] = tuple(meta["classes"])
        self.w, self.h = meta["size"]
        self.mean = np.array(meta["mean"], np.float32).reshape(3, 1, 1)
        self.std = np.array(meta["std"], np.float32).reshape(3, 1, 1)
        self.sess = ort.InferenceSession(str(onnx_path),
                                         providers=["CPUExecutionProvider"])
        self.name = self.sess.get_inputs()[0].name

    def predict(self, rgb: np.ndarray) -> tuple[str, float]:
        x = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
        x = ((x - self.mean) / self.std)[None]
        logits = self.sess.run(None, {self.name: x})[0][0]
        e = np.exp(logits - logits.max())
        p = e / e.sum()
        i = int(p.argmax())
        return self.classes[i], float(p[i])


class _TorchClassifier:
    """PyTorch 백엔드. 개발 PC 용 / ONNX 가 없을 때의 대비책."""

    def __init__(self, ckpt_path: Path, device: str) -> None:
        import torch
        from torchvision import models

        self._torch = torch
        self.device = torch.device(device)
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.classes = tuple(ck["classes"])
        self.w, self.h = ck["size"]
        self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)

        net = models.mobilenet_v3_small(weights=None)
        net.classifier[3] = torch.nn.Linear(net.classifier[3].in_features,
                                            len(self.classes))
        net.load_state_dict(ck["model"])
        self.net = net.to(self.device).eval()

    def predict(self, rgb: np.ndarray) -> tuple[str, float]:
        x = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
        x = ((x - self.mean) / self.std)[None]
        t = self._torch.from_numpy(x).to(self.device)
        with self._torch.no_grad():
            p = self._torch.softmax(self.net(t), dim=1)[0].cpu().numpy()
        i = int(p.argmax())
        return self.classes[i], float(p[i])


def _load_classifier(path: str, device: str):
    """.onnx 면 onnxruntime, .pt 면 torch. onnx 가 있으면 그쪽을 먼저 쓴다."""
    p = Path(path)
    if p.suffix == ".pt":
        alt = p.with_suffix(".onnx")
        if alt.exists() and alt.with_suffix(".json").exists():
            p = alt
    if p.suffix == ".onnx":
        return _OnnxClassifier(p)
    return _TorchClassifier(p, device)


class ModelSignalDetector:
    """YOLO(ped_light) → 크롭 → CNN(red/green/off).

    검출기는 .pt 와 NCNN 디렉터리를 모두 받는다. NCNN 은 내보낼 때 입력 크기가
    고정되므로, 디렉터리의 metadata.yaml 에서 읽어 imgsz 를 맞춘다.
    분류기는 .onnx 가 있으면 우선 사용한다 (파이에 torch 불필요).

    무거운 임포트는 생성 시점까지 미룬다. 의존성이 없는 환경에서도
    main.py 가 규칙 기반으로 뜰 수 있어야 하기 때문이다.
    """

    def __init__(
        self,
        det_weights: str | None = None,
        cls_weights: str | None = None,
        conf: float | None = None,
        imgsz: int | None = None,
        device: str = "cpu",
    ) -> None:
        from ultralytics import YOLO

        self.conf = conf if conf is not None else config.SIGNAL_DET_CONF
        self.device = device

        det_path = Path(det_weights or config.SIGNAL_DET_WEIGHTS)
        self.det = YOLO(str(det_path))
        self.is_ncnn = (det_path / "model.ncnn.param").exists()

        want = imgsz or config.SIGNAL_DET_IMGSZ
        self.imgsz: int | list[int] = want
        if self.is_ncnn:
            # NCNN 은 내보낸 해상도로 고정된다. 다른 값을 넘기면 조용히 틀린다.
            baked = _ncnn_imgsz(det_path)
            if baked and baked != want:
                print(f"[신호등] NCNN 내보내기 해상도 {baked} 사용 (요청 {want} 무시)")
                self.imgsz = baked

        clf = _load_classifier(cls_weights or config.SIGNAL_CLS_WEIGHTS, device)
        self.cls = clf
        self.classes = clf.classes
        self.cls_w, self.cls_h = clf.w, clf.h

    # ── 검출 ────────────────────────────────────────────────
    def detect(self, main_bgr: np.ndarray) -> list[Detection]:
        """전체 프레임에서 ped_light 만 뽑는다. ROI 제한 없음."""
        res = self.det.predict(main_bgr, imgsz=self.imgsz, conf=self.conf,
                               classes=[config.SIGNAL_DET_PED_CLASS],
                               verbose=False, device=self.device)[0]
        out: list[Detection] = []
        if res.boxes is None or len(res.boxes) == 0:
            return out
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        h, w = main_bgr.shape[:2]
        for (x0, y0, x1, y1), c in zip(xyxy, confs):
            out.append(Detection(
                int(max(0, x0)), int(max(0, y0)),
                int(min(w, x1)), int(min(h, y1)), float(c)))
        return out

    # ── 분류 ────────────────────────────────────────────────
    def _crop(self, main_bgr: np.ndarray, d: Detection) -> np.ndarray | None:
        """학습 때와 같은 15% 여백으로 자른다. 함체 테두리가 보여야
        분류기가 위=적 / 아래=녹 이라는 점등 위치를 쓸 수 있다."""
        h, w = main_bgr.shape[:2]
        pad = config.SIGNAL_CROP_PAD
        px, py = int(round((d.x1 - d.x0) * pad)), int(round(d.h * pad))
        x0, y0 = max(0, d.x0 - px), max(0, d.y0 - py)
        x1, y1 = min(w, d.x1 + px), min(h, d.y1 + py)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None
        return main_bgr[y0:y1, x0:x1]

    def classify(self, patch_bgr: np.ndarray) -> tuple[str, float]:
        rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.cls_w, self.cls_h), interpolation=cv2.INTER_AREA)
        return self.cls.predict(rgb)

    # ── 통합 (detect_signal_once 와 동일한 계약) ──────────────
    def __call__(self, main_bgr: np.ndarray) -> tuple[str, int]:
        dets = self.detect(main_bgr)
        if not dets:
            return NONE, 0

        # 여러 개면 화면 중앙에 가까우면서 큰 쪽을 고른다.
        # ★ 면적이 아니라 sqrt(면적)=선형 크기로 비교한다. 면적은 거리 제곱에
        #   반비례해서, 가까운 가장자리 신호등이 정면 신호등을 너무 쉽게 이긴다.
        #   (규칙 기반 detect_signal_once 와 같은 판단 기준)
        w = main_bgr.shape[1]
        best, best_score = None, -1.0
        for d in dets:
            cx = (d.x0 + d.x1) / 2.0 / max(w, 1)
            centrality = 1.0 - config.SIGNAL_CENTER_BIAS * min(1.0, abs(cx - 0.5) * 2.0)
            score = float(np.sqrt(max(d.area, 1))) * centrality * d.conf
            if score > best_score:
                best, best_score = d, score

        assert best is not None
        if best.h < config.SIGNAL_MIN_BOX_H:
            # 너무 작으면 색 판독이 물리적으로 불가능하다. 억지로 분류하지 않는다.
            return NONE, best.area

        patch = self._crop(main_bgr, best)
        if patch is None:
            return NONE, best.area

        label, prob = self.classify(patch)
        if prob < config.SIGNAL_CLS_MIN_PROB:
            return NONE, best.area          # 확신 없으면 "모름". 추측하지 않는다.
        return _CLS_TO_STATE.get(label, NONE), best.area


def build_signal_detector(backend: str, **kwargs):
    """backend='model' 이면 모델 검출기, 'rule' 이면 기존 색공간 규칙."""
    if backend == "rule":
        from signal_light import detect_signal_once
        return detect_signal_once
    if backend == "model":
        return ModelSignalDetector(**kwargs)
    raise ValueError(f"모르는 backend: {backend}")
