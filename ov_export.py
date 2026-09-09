#!/usr/bin/env python3
"""
ov_export.py — export a YOLO .pt to OpenVINO IR at a given precision and
move the result to a precision-tagged directory. Called by
test_yolo_intel.sh for every model x precision.

Usage:
  python3 ov_export.py --model yolo11n.pt --precision fp32|fp16|int8 \
      --target /path/yolo11n_fp16_openvino_model [--imgsz 640] [--data coco8.yaml]

Ultralytics writes <stem>_openvino_model/ regardless of precision, so the
output is moved to --target to keep FP32/FP16/INT8 exports apart. Newer
Ultralytics takes quantize=16/8; older takes half=True / int8=True — both
are tried so the script works across versions. A file (not a stdin
heredoc) so that any multiprocessing spawned during export can re-import
its __main__ cleanly.
"""

import argparse
import os
import shutil
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--precision", required=True, choices=["fp32", "fp16", "int8"])
    ap.add_argument("--target", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--data", default=None, help="calibration dataset yaml for int8 (default: ultralytics' task default)")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    kw = dict(format="openvino", imgsz=args.imgsz, batch=1, dynamic=False)
    if args.precision == "int8" and args.data:
        kw["data"] = args.data
    new_api = {"fp32": {}, "fp16": {"quantize": 16}, "int8": {"quantize": 8}}[args.precision]
    old_api = {"fp32": {}, "fp16": {"half": True}, "int8": {"int8": True}}[args.precision]

    out = None
    last_err = None
    for extra in (new_api, old_api):
        try:
            out = model.export(**kw, **extra)
            print(f"EXPORT_ARGS={extra or 'default(fp32)'}")
            break
        except (TypeError, SyntaxError, KeyError) as e:  # unknown kwarg on this ultralytics version
            last_err = e
            print(f"export with {extra} rejected ({e}); trying alternate args")
    if not out:
        print(f"EXPORT_ERROR={last_err}")
        sys.exit(1)

    out = str(out)
    if os.path.isfile(out):  # some versions return the .xml path
        out = os.path.dirname(out)
    target = os.path.abspath(args.target)
    if os.path.abspath(out) != target:
        if os.path.exists(target):
            shutil.rmtree(target)
        shutil.move(out, target)
    xmls = [f for f in os.listdir(target) if f.endswith(".xml")]
    if not xmls:
        print(f"EXPORT_ERROR=no .xml in {target}")
        sys.exit(1)
    print(f"EXPORTED={target}")


if __name__ == "__main__":
    main()
