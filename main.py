"""
blindnav — 라즈베리파이 5 시각장애인 보행 보조

실행:
  # 개발 PC (동영상으로 테스트)
  python main.py --source sample.mp4 --view --tts print

  # Pi 5 실기
  python main.py --source pi --tts wav

  # 검출 없이 색 로직만 (YOLO 미설치 상태에서도 확인 가능)
  python main.py --source sample.mp4 --no-detect --view

구조
  MAIN  1920x1080 ──▶ 신호등 ROI 크롭(원본 해상도) ──▶ LAB a* + 5프레임 투표
                 └─▶ 점자블록 하단 ROI ──▶ LAB b*
  LORES 416x416  ──▶ YOLO(COCO 사전학습) ──▶ bbox높이→거리→접근속도
  BH1750/gain    ──▶ 조도 프로파일 ──▶ 야간엔 신호등 판정 포기

학습 데이터 0장. 사전학습 가중치 + 색공간 규칙만 사용.
"""

from __future__ import annotations

import argparse
import time

import cv2

import config
from announce import Announcer
from context import ContextTracker, LuxSensor
from pavement import PavementTracker, detect_pavement
from signal_light import SignalResult, SignalVoter
from sources import open_source


def obstacle_message(obs) -> tuple[str, str] | None:
    """가장 위험한 장애물 하나만 안내한다. 여러 개 읽으면 못 알아듣는다."""
    side_kr = {"left": "왼쪽", "front": "정면", "right": "오른쪽"}[obs.side]

    if obs.is_danger:
        return "danger", f"{side_kr} {obs.dist_m:.0f}미터, 정지하세요"
    if obs.dist_m <= config.WARN_DIST_M and obs.is_approaching:
        return "warn", f"{side_kr}에서 접근 중입니다"
    if obs.dist_m <= config.WARN_DIST_M:
        return "warn", f"{side_kr} {obs.dist_m:.0f}미터 앞 장애물"
    return None


def draw_debug(frame, obstacles, pave, sig, profile, fps, view_w=960,
               signal_boxes=None, show_signal_roi=False):
    """디버그 오버레이.

    ★ 주의: obstacle.box 는 lores(416) 좌표계이고 frame 은 main(1920x1080)이다.
      스케일을 안 맞추면 박스가 좌상단 구석에 조그맣게 찍힌다.
    """
    h, w = frame.shape[:2]
    scale = view_w / float(w)
    vis = cv2.resize(frame, (view_w, int(h * scale)), interpolation=cv2.INTER_AREA)
    vh, vw = vis.shape[:2]

    # lores 좌표 → 표시 좌표 변환 계수
    lw, lh = config.LORES_SIZE
    sx, sy = vw / float(lw), vh / float(lh)

    for o in obstacles:
        x1, y1, x2, y2 = o.box
        x1, y1 = int(x1 * sx), int(y1 * sy)
        x2, y2 = int(x2 * sx), int(y2 * sy)
        color = (0, 0, 255) if o.is_danger else (0, 200, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"{o.label} {o.dist_m:.1f}m {o.speed_mps:+.1f}",
                    (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # 신호등: 모델 백엔드는 ROI 가 없으므로 검출 박스를 그대로 그린다.
    # 규칙 백엔드일 때만 ROI 사각형을 표시한다.
    if signal_boxes:
        sx2, sy2 = vw / float(w), vh / float(h)
        for b in signal_boxes:
            cv2.rectangle(vis, (int(b.x0 * sx2), int(b.y0 * sy2)),
                          (int(b.x1 * sx2), int(b.y1 * sy2)), (255, 180, 0), 2)
            cv2.putText(vis, f"{b.conf:.2f} h{b.h}",
                        (int(b.x0 * sx2), max(12, int(b.y0 * sy2) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 180, 0), 1)
    elif show_signal_roi:
        x0, y0, x1_, y1_ = config.SIGNAL_ROI
        cv2.rectangle(vis, (int(x0 * vw), int(y0 * vh)),
                      (int(x1_ * vw), int(y1_ * vh)), (255, 180, 0), 1)

    # 점자블록 ROI 경계선 (색상 기반 유지)
    cv2.line(vis, (0, int(config.PAVE_ROI_TOP * vh)),
             (vw, int(config.PAVE_ROI_TOP * vh)), (0, 255, 255), 1)

    lines = [
        f"FPS {fps:5.1f}   profile={profile}",
        f"signal={sig.state} votes={sig.votes} flicker={sig.flicker} px={sig.blob_px}",
        f"pave found={pave.found} off={pave.offset:+.2f} ang={pave.angle_deg:+.0f}",
    ]
    for i, t in enumerate(lines):
        cv2.putText(vis, t, (8, 20 + i * 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 3)
        cv2.putText(vis, t, (8, 20 + i * 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="pi", help="'pi' | 동영상 경로 | 웹캠 인덱스")
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--tts", default="print", choices=["print", "espeak", "wav"])
    ap.add_argument("--no-detect", action="store_true", help="YOLO 끄고 색 로직만")
    ap.add_argument("--view", action="store_true", help="디버그 창 표시")
    ap.add_argument("--focal", type=float, default=None, help="FOCAL_PX 덮어쓰기")
    ap.add_argument("--signal-backend", default=config.SIGNAL_BACKEND,
                    choices=["model", "rule"],
                    help="model=YOLO+CNN(ROI 미사용) | rule=색공간 규칙(SIGNAL_ROI 사용)")
    ap.add_argument("--signal-conf", type=float, default=None,
                    help="검출 신뢰도 임계값 덮어쓰기 (기본 config.SIGNAL_DET_CONF)")
    ap.add_argument("--save", default=None, metavar="OUT.mp4",
                    help="디버그 오버레이를 영상으로 저장. 헤드리스(SSH) 환경에서 --view 대신")
    ap.add_argument("--loop", action="store_true", help="영상 파일을 무한 반복")
    ap.add_argument("--max-frames", type=int, default=0, help="N 프레임 처리 후 종료 (0=끝까지)")
    args = ap.parse_args()

    if args.signal_conf is not None:
        config.SIGNAL_DET_CONF = args.signal_conf

    if args.focal:
        config.FOCAL_PX = args.focal

    src = open_source(args.source, loop=args.loop)
    ann = Announcer(backend=args.tts)
    ctx = ContextTracker(LuxSensor())
    pave_tracker = PavementTracker()

    # ── 신호등 백엔드 ──
    # model: 전체 프레임에서 YOLO 가 보행등 함체를 찾고, 그 박스만 CNN 이 판정한다.
    #        ROI 제한이 없다. 위치가 아니라 형태로 찾기 때문이다.
    # rule : 기존 색공간 규칙(SIGNAL_ROI 크롭 + LAB a*). 모델이 안 뜰 때의 대비책.
    signal_detector = None
    if args.signal_backend == "model":
        try:
            from signal_model import ModelSignalDetector
            signal_detector = ModelSignalDetector()
            print(f"[신호등] 모델 백엔드 conf={config.SIGNAL_DET_CONF} "
                  f"imgsz={config.SIGNAL_DET_IMGSZ} (ROI 미사용)")
        except Exception as e:
            print(f"[경고] 모델 신호등 로드 실패 → 규칙 기반으로 전환합니다: {e}")
            args.signal_backend = "rule"
    if args.signal_backend == "rule":
        print(f"[신호등] 규칙 백엔드 (레거시). SIGNAL_ROI={config.SIGNAL_ROI}")
    voter = SignalVoter(detector=signal_detector)

    detector = None
    if not args.no_detect:
        try:
            from detect import ObstacleDetector
            detector = ObstacleDetector(weights=args.weights)
        except Exception as e:
            print(f"[경고] 검출기 로드 실패 → 색 로직만 동작합니다: {e}")

    print(f"[起動] source={args.source} focal={config.FOCAL_PX} "
          f"lux={ctx.source_name} tts={args.tts}")
    ann.say("status", "시스템을 시작합니다", force=True)

    writer = None
    overlay = args.view or bool(args.save)

    frame_i = 0
    fps, t_prev = 0.0, time.monotonic()
    last_profile = None
    sig = SignalResult("unknown", False, False, 0)
    signal_boxes: list = []

    try:
        for main_f, lores_f, meta in src:
            now = time.monotonic()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            frame_i += 1

            profile = ctx.update(meta)
            if profile != last_profile:
                if profile == "night":
                    ann.say("status", "신호를 확인할 수 없습니다", force=True)
                last_profile = profile

            # ── 신호등 (MAIN 원본 크롭, N프레임에 1회) ──
            # ★ voter.update 를 매 프레임 호출하면 안 된다.
            #   None 을 넘기면 투표 창이 비워져서 표가 절대 누적되지 않는다.
            #   판정하는 프레임에서만 호출하고, 나머지는 직전 결과를 재사용한다.
            if frame_i % config.SIGNAL_INTERVAL == 0:
                sig = voter.update(main_f, profile)
                msg = voter.take_event(sig)
                if msg:
                    ann.say("signal", msg)
                # 디버그 표시용. 판정과 무관하므로 --view 일 때만 다시 뽑는다.
                if args.view and signal_detector is not None:
                    signal_boxes = signal_detector.detect(main_f)

            # ── 점자블록 (MAIN 하단 ROI) ──
            pave = detect_pavement(main_f, profile)
            pmsg = pave_tracker.update(pave)
            if pmsg:
                ann.say("pave", pmsg)

            # ── 장애물 (LORES → YOLO) ──
            obstacles = []
            if detector and frame_i % config.DETECT_INTERVAL == 0:
                obstacles = detector(lores_f, now)
                detector.prune(now)
                if obstacles:
                    got = obstacle_message(obstacles[0])
                    if got:
                        ann.say(*got)

            if overlay:
                vis = draw_debug(
                    main_f, obstacles, pave, sig, profile, fps,
                    signal_boxes=signal_boxes,
                    show_signal_roi=(args.signal_backend == "rule"))
                if args.save:
                    if writer is None:
                        h_v, w_v = vis.shape[:2]
                        writer = cv2.VideoWriter(
                            args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                            config.TARGET_FPS, (w_v, h_v))
                        print(f"[저장] {args.save} ({w_v}x{h_v})")
                    writer.write(vis)
                if args.view:
                    cv2.imshow("blindnav", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            if args.max_frames and frame_i >= args.max_frames:
                break

    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        ann.close()
        if writer is not None:
            writer.release()
            print(f"[저장 완료] {args.save}")
        if args.view:
            cv2.destroyAllWindows()
        print(f"\n[종료] 총 {frame_i} 프레임, 평균 {fps:.1f} fps")


if __name__ == "__main__":
    main()
