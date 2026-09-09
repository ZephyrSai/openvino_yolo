#!/usr/bin/env python3
"""
ov_raw_bench.py — raw OpenVINO Runtime benchmark (no Ultralytics wrapper,
no NMS) called by test_yolo_intel.sh to characterise each Intel device
(CPU / GPU / NPU) under both OpenVINO performance hints.

Usage:
  python3 ov_raw_bench.py --model <model.xml | dir_openvino_model> \
      --device <CPU|GPU|NPU|GPU.0|AUTO|...> --hint <LATENCY|THROUGHPUT> \
      [--video <path.mp4>] [--runs 15] [--video-max-frames 300] \
      [--cache-dir <dir>]

What it measures (KEY=VALUE lines, one per line):
  DEVICE_FULL_NAME     - what the OpenVINO plugin reports for the device
  READ_MODEL_MS        - core.read_model() (IR parse) time
  COMPILE_MS           - core.compile_model() time. On GPU this is the
                          OpenCL kernel JIT; on NPU it is the full graph
                          compile. Both are one-time costs that OpenVINO's
                          model cache (--cache-dir) can eliminate on the
                          second run, which the caller measures by running
                          this script twice with the same --cache-dir.
  CACHE_USED           - whether a cache dir was configured
  OPTIMAL_NUM_REQUESTS - OPTIMAL_NUMBER_OF_INFER_REQUESTS reported by the
                          plugin for the chosen hint
  FIRST_INFER_MS       - first synchronous infer (lazy allocations etc.)
  STEADY_AVG_MS / MIN / MAX / P95
                        - synchronous single-request latency on a dummy
                          tensor. Pure device inference latency, i.e. the
                          floor below which the Ultralytics end-to-end
                          number (bench_harness.py) can never go.
  VIDEO_FRAMES / VIDEO_AVG_FPS / VIDEO_AVG_MS
                        - decode + resize + raw inference over the test
                          video. Under the THROUGHPUT hint the video pass
                          uses an AsyncInferQueue sized to the plugin's
                          optimal request count, so it reports pipelined
                          throughput rather than serial latency.
"""

import argparse
import os
import statistics
import sys
import time

import numpy as np


def resolve_model_path(p: str) -> str:
    if os.path.isdir(p):
        xmls = [f for f in os.listdir(p) if f.endswith(".xml")]
        if not xmls:
            raise FileNotFoundError(f"no .xml in {p}")
        return os.path.join(p, sorted(xmls)[0])
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--hint", default="LATENCY", choices=["LATENCY", "THROUGHPUT"])
    ap.add_argument("--video", default=None)
    ap.add_argument("--runs", type=int, default=15)
    ap.add_argument("--video-max-frames", type=int, default=300)
    ap.add_argument("--cache-dir", default=None)
    args = ap.parse_args()

    try:
        import openvino as ov
    except Exception as e:
        print(f"ERROR=openvino import failed: {e}")
        sys.exit(1)

    model_xml = resolve_model_path(args.model)
    print(f"MODEL={model_xml}")
    print(f"DEVICE={args.device}")
    print(f"HINT={args.hint}")

    core = ov.Core()
    print(f"OV_VERSION={ov.get_version()}")
    print(f"OV_AVAILABLE_DEVICES={core.available_devices}")
    base_dev = args.device.split(".")[0].split(":")[0]
    try:
        print(f"DEVICE_FULL_NAME={core.get_property(args.device, 'FULL_DEVICE_NAME')}")
    except Exception as e:
        print(f"DEVICE_FULL_NAME_ERROR={e}")

    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
        core.set_property({"CACHE_DIR": args.cache_dir})
        print("CACHE_USED=1")
    else:
        print("CACHE_USED=0")

    t0 = time.perf_counter()
    try:
        model = core.read_model(model_xml)
    except Exception as e:
        print(f"READ_MODEL_ERROR={e}")
        sys.exit(1)
    print(f"READ_MODEL_MS={(time.perf_counter() - t0) * 1000:.1f}")

    # NPU (and GPU for best perf) want static shapes. Ultralytics exports
    # static [1,3,640,640] by default; if someone passed a dynamic IR,
    # pin it here so the NPU plugin doesn't reject it.
    inp = model.inputs[0]
    if inp.get_partial_shape().is_dynamic:
        model.reshape({inp.get_any_name(): [1, 3, 640, 640]})
        print("RESHAPED_TO_STATIC=1")

    config = {"PERFORMANCE_HINT": args.hint}
    t0 = time.perf_counter()
    try:
        compiled = core.compile_model(model, args.device, config)
    except Exception as e:
        print(f"COMPILE_ERROR={e}")
        sys.exit(1)
    print(f"COMPILE_MS={(time.perf_counter() - t0) * 1000:.1f}")

    try:
        nreq = compiled.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
    except Exception:
        nreq = 1
    print(f"OPTIMAL_NUM_REQUESTS={nreq}")
    try:
        print(f"EXECUTION_DEVICES={compiled.get_property('EXECUTION_DEVICES')}")
    except Exception:
        pass

    shape = list(compiled.inputs[0].shape)
    dummy = np.random.rand(*shape).astype(np.float32)
    req = compiled.create_infer_request()

    t0 = time.perf_counter()
    try:
        req.infer({0: dummy})
    except Exception as e:
        print(f"FIRST_INFER_ERROR={e}")
        sys.exit(1)
    print(f"FIRST_INFER_MS={(time.perf_counter() - t0) * 1000:.2f}")

    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        req.infer({0: dummy})
        times.append((time.perf_counter() - t0) * 1000)
    ts = sorted(times)
    p95_idx = max(0, int(len(ts) * 0.95) - 1)
    print(f"STEADY_AVG_MS={statistics.mean(times):.2f}")
    print(f"STEADY_MIN_MS={min(times):.2f}")
    print(f"STEADY_MAX_MS={max(times):.2f}")
    print(f"STEADY_P95_MS={ts[p95_idx]:.2f}")

    if not args.video:
        return
    try:
        import cv2
    except Exception as e:
        print(f"VIDEO_ERROR=opencv not available: {e}")
        return
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"VIDEO_ERROR=could not open {args.video}")
        return

    h, w = shape[2], shape[3]

    def to_tensor(frame):
        resized = cv2.resize(frame, (w, h))
        return resized.transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0

    n = 0
    t_start = time.perf_counter()
    if args.hint == "THROUGHPUT" and int(nreq) > 1:
        queue = ov.AsyncInferQueue(compiled, int(nreq))
        while n < args.video_max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            queue.start_async({0: to_tensor(frame)})
            n += 1
        queue.wait_all()
        t_total = time.perf_counter() - t_start
        cap.release()
        if n:
            print(f"VIDEO_MODE=async_queue_{nreq}")
            print(f"VIDEO_FRAMES={n}")
            print(f"VIDEO_TOTAL_S={t_total:.2f}")
            print(f"VIDEO_AVG_MS={t_total * 1000 / n:.2f}")
            print(f"VIDEO_AVG_FPS={n / t_total:.2f}")
    else:
        frame_times = []
        while n < args.video_max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            tensor = to_tensor(frame)
            t0 = time.perf_counter()
            req.infer({0: tensor})
            frame_times.append((time.perf_counter() - t0) * 1000)
            n += 1
        t_total = time.perf_counter() - t_start
        cap.release()
        if n:
            print("VIDEO_MODE=sync")
            print(f"VIDEO_FRAMES={n}")
            print(f"VIDEO_TOTAL_S={t_total:.2f}")
            print(f"VIDEO_AVG_MS={statistics.mean(frame_times):.2f}")
            print(f"VIDEO_AVG_FPS={n / t_total:.2f}")


if __name__ == "__main__":
    main()
