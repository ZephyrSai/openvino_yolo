# YOLO on Intel Core Ultra CPU/iGPU/NPU — OpenVINO Diagnostic + Benchmark Suite

A full test suite for running [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO (YOLO11 and YOLO26) on Intel laptops and desktops through the Intel AI stack — **OpenVINO** on the CPU, the integrated Arc GPU, and the NPU (Intel AI Boost); **PyTorch XPU** on the iGPU; and **ONNX Runtime** with the OpenVINO Execution Provider. It tells you which devices actually work on your specific chip and how fast each one runs, with real numbers instead of guesswork.

This is the Intel counterpart of [ryzen_yolo](https://github.com/ZephyrSai/ryzen_yolo) — same structure, same metrics, same output format, so results from the two are directly comparable.

## Supported hardware

| Chip family | Examples | iGPU | NPU | Kernel (NPU) |
|---|---|---|---|---|
| Meteor Lake (Core Ultra 100 H/U) | 155H, 165H, 125U | Arc Graphics / Intel Graphics (Xe-LPG) | NPU 3720, ~11 TOPS | ≥ 6.8 |
| Arrow Lake (Core Ultra 200 H/HX/U/S) | 255H, 285H, 265K | Arc 140T/130T, Intel Graphics | NPU 3720, ~13 TOPS | ≥ 6.11 |
| Lunar Lake (Core Ultra 200V) | 258V, 268V, 288V | Arc 140V/130V (Xe2) | NPU 4, ~48 TOPS | ≥ 6.11 |
| Panther Lake (Core Ultra 300) | 3xx | Arc B390/B370 (Xe3) | NPU 5, ~50 TOPS | ≥ 6.17 |
| Core 12th–14th gen (Alder/Raptor Lake) | i7-13700H, i5-1240P | Iris Xe / UHD | none | n/a |

The script auto-detects the family from `lscpu` and adjusts its kernel-version advice and NPU expectations accordingly. The ground truth for what is usable is always OpenVINO's own `available_devices`, which the script probes directly. Discrete Arc cards (A-series, B-series) also work through the same GPU plugin.

## What it does

1. **System checks** — kernel version, `i915`/`xe` GPU module, `/dev/dri/renderD*` and its permissions, `render` group membership, OpenCL ICD, Level Zero loader + GPU driver, `clinfo`, `intel_vpu` NPU module, `/dev/accel/accel0` and its permissions, NPU firmware, Level Zero NPU driver, chip auto-detection.
2. **Environment setup** — creates an isolated venv (`~/yolo-intel-test/venv`), installs PyTorch (XPU build, CPU fallback), Ultralytics, OpenVINO, `onnxruntime-openvino`. Downloads:
   - A public test image (Ultralytics' `bus.jpg`)
   - Six models: **YOLO11** and **YOLO26**, nano/small/medium each
   - A real public test video — `solutions_ci_demo.mp4`, from [Ultralytics' official GitHub Releases CDN](https://github.com/ultralytics/assets/releases/tag/v0.0.0)
3. **Device visibility** — `openvino.Core().available_devices`, the full device name of each, and a tiny compiled matmul on every listed device so "listed" also means "works". Plus `torch.xpu.is_available()` and a matmul on the XPU.
4. **OpenVINO IR export** — every model in FP32 and FP16 (`--int8` adds INT8 post-training quantization on coco8). Works with both the new `quantize=16/8` and the older `half=True`/`int8=True` Ultralytics export APIs.
5. **Full benchmark**, every model × every backend that works:
   - PyTorch CPU (baseline), PyTorch XPU (iGPU)
   - OpenVINO CPU / GPU / NPU via Ultralytics `device="intel:cpu|gpu|npu"`, FP32 and FP16 IRs
   - For each: **cold-load**, **first-inference** (includes the GPU/NPU model compile), **steady-state** avg/min/max/p95/stdev over 15 runs, and **real video FPS** (decode + preprocess + inference + NMS over up to 300 frames)
6. **Raw OpenVINO Runtime characterisation** — nano models on each device under the `LATENCY` and `THROUGHPUT` performance hints (the latter with an async infer queue sized to the plugin's optimal request count), the pure inference-latency floor, and **cold vs cached compile time** using OpenVINO's model cache.
7. **Intel's `benchmark_app`** — reference throughput / median latency per device and hint, directly comparable to Intel's published figures.
8. **ONNX Runtime + OpenVINO Execution Provider** — `device_type` CPU/GPU/NPU, with a check that the session did not silently fall back to the CPU provider.
9. **NPU diagnosis** — one verdict on why the NPU is or isn't usable (kernel too old, firmware missing, permissions on `/dev/accel/accel0`, user-mode driver missing, version mismatch) with the exact fix, plus the relevant `dmesg` lines.
10. **Summary** — pass/fail counts, five comparison tables (video FPS, steady-state latency, first-inference, cold-load, compile time) across every backend and model, and an auto-generated env file naming the fastest device.

## Requirements

- Ubuntu 24.04 (or similar) on an Intel system
- For the GPU: Intel compute runtime (OpenCL + Level Zero) installed at the system level
- For the NPU: kernel with `intel_vpu` and Intel's user-mode NPU driver installed (see [one-time setup](#one-time-driver-setup))
- Internet access to `github.com`, `pypi.org`, and `download.pytorch.org`

CPU-only runs work on any x86 machine with nothing but Python installed. The script never fails because a device is missing — it warns, explains what to install, and benchmarks what is there.

## Usage

```bash
chmod +x test_yolo_intel.sh
./test_yolo_intel.sh                # full run: YOLO11 + YOLO26, n/s/m each, FP32 + FP16
./test_yolo_intel.sh --quick        # nano only from each family — faster, less bandwidth
./test_yolo_intel.sh --int8         # also export + benchmark INT8 IRs
./test_yolo_intel.sh --skip-install # skip venv/package installs (reuse existing setup)
```

Safe to re-run at any time. It never touches your system Python — everything lives in `~/yolo-intel-test/venv`. The harness scripts are called from wherever you cloned this repo, so keep the four files together.

## Output

Everything is written under `~/yolo-intel-test/`:

| Path | Contents |
|---|---|
| `results_summary.txt` | Every PASS/FAIL/WARN line from the run |
| `benchmark_results.csv` | Raw numbers: `backend,model,stage,metric,value` — import into a spreadsheet or plot directly |
| `logs/` | Full stdout/stderr from every individual test, `clinfo`, `benchmark_app`, `dmesg` (NPU), for debugging |
| `yolo_intel_env.sh` | Auto-generated exports with the recommended Ultralytics device |
| `ov_cache/` | OpenVINO compiled-model cache (GPU/NPU) — delete to re-measure cold compiles |
| `*_fp32_openvino_model/`, `*_fp16_openvino_model/` | Exported IRs, ready to reuse |
| `bus.jpg`, `solutions_ci_demo.mp4`, `*.pt`, `*.onnx` | Downloaded test assets and models |

Backend labels in the CSV and tables:

| Label | Meaning |
|---|---|
| `pytorch_cpu`, `pytorch_xpu` | Ultralytics on `.pt` weights, end-to-end (with NMS) |
| `openvino_{cpu,gpu,npu}_{fp32,fp16,int8}` | Ultralytics on OpenVINO IR via `device=intel:<dev>`, end-to-end (with NMS) |
| `ov_raw_{dev}_{latency,throughput}` | Raw OpenVINO runtime, dummy/raw tensors, no NMS |
| `ov_raw_{dev}_latency_cache_{cold,warm}` | Same, measuring compile time without / with the model cache |
| `benchmark_app_{dev}_{latency,throughput}` | Intel's reference tool |
| `onnxrt_openvino_{dev}` | ONNX Runtime, OpenVINO EP, raw tensors, no NMS |

Only the `pytorch_*` and `openvino_*` rows are apples-to-apples with the AMD suite's `cpu` / `gpu_pytorch_rocm` rows; the raw rows are upper bounds on device throughput.

### Using the results

The script asks whether to append the working exports to `~/.bashrc`. Either way, the generated file tells you which device won and how to export for it:

```bash
source ~/yolo-intel-test/yolo_intel_env.sh
yolo export model=yolo11n.pt format=openvino quantize=16     # half=True on older ultralytics
yolo predict model=yolo11n_openvino_model source=your_image.jpg device=$ULTRALYTICS_DEVICE
```

`ULTRALYTICS_DEVICE` will be one of `intel:gpu`, `intel:npu`, `intel:cpu`, `xpu:0` or `cpu`.

## NPU notes

- The NPU is reached only through OpenVINO's NPU plugin. Ultralytics supports it natively as `device="intel:npu"` on an exported OpenVINO model. There is no PyTorch NPU device.
- The NPU needs **static input shapes** (the default Ultralytics export) and prefers **FP16 or INT8**. The suite uses the FP16 IR for all NPU tests; FP32-on-NPU failures are reported as warnings, not failures.
- The **first compile** of a model on the NPU is slow (often 10–60 s). It shows up in `FIRST_INFER_MS`; the `*_cache_warm` rows show what OpenVINO's model cache saves on subsequent runs. Set `CACHE_DIR` (or the env file's `OV_CACHE_DIR`) in your own application.
- **YOLO26 on the NPU** is only supported by Ultralytics on Core Ultra 200V (Lunar Lake) and 300 (Panther Lake) and newer. On Meteor Lake / Arrow Lake expect the `openvino_npu_*/yolo26*` rows to warn — YOLO11 is the NPU model there.
- Firmware and user-mode driver versions must match each other and be compatible with the installed `openvino` package. Version mismatches are the most common cause of "NPU listed but compile fails"; the NPU driver release notes name the matching OpenVINO release.

## One-time driver setup

Run once, before this script. Nothing here is installed by the script itself.

### GPU (OpenCL + Level Zero compute runtime)

Ubuntu 24.04's own archive ships `intel-opencl-icd`, which is enough for the OpenVINO GPU plugin on most Core Ultra chips. For the newest silicon or for PyTorch XPU, use Intel's graphics repository:

```bash
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
  sudo gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble unified" | \
  sudo tee /etc/apt/sources.list.d/intel-gpu-noble.list
sudo apt update
sudo apt install -y libze-intel-gpu1 libze1 intel-opencl-icd clinfo intel-gsc
sudo gpasswd -a $USER render
# log out and back in, then verify:
clinfo | grep "Device Name"
```

Reference: [Intel client GPU driver install guide](https://dgpu-docs.intel.com/driver/client/overview.html).

### NPU (Intel AI Boost)

1. Kernel with `intel_vpu`: Ubuntu 24.04 HWE (`sudo apt install linux-generic-hwe-24.04`) or newer. Verify with `modinfo intel_vpu`.
2. User-mode driver + firmware from [intel/linux-npu-driver releases](https://github.com/intel/linux-npu-driver/releases) — download the `linux-npu-driver-v*-ubuntu2404.tar.gz` for your OpenVINO version, then:

```bash
mkdir npu-driver && tar -xf linux-npu-driver-v*-ubuntu2404.tar.gz -C npu-driver
cd npu-driver
sudo dpkg --purge --force-remove-reinstreq intel-driver-compiler-npu intel-fw-npu intel-level-zero-npu 2>/dev/null
sudo apt install -y libtbb12
sudo dpkg -i *.deb
sudo gpasswd -a $USER render
sudo rmmod intel_vpu; sudo modprobe intel_vpu
```

3. Make `/dev/accel/accel0` accessible to the `render` group persistently:

```bash
echo 'SUBSYSTEM=="accel", KERNEL=="accel*", GROUP="render", MODE="0660"' | \
  sudo tee /etc/udev/rules.d/10-intel-vpu.rules
sudo udevadm control --reload-rules && sudo udevadm trigger --subsystem-match=accel
```

4. Log out and back in, then verify:

```bash
ls -l /dev/accel/accel0
python3 -c "import openvino as ov; print(ov.Core().available_devices)"
```

Reference: [linux-npu-driver overview](https://github.com/intel/linux-npu-driver/blob/main/docs/overview.md).

### PyTorch XPU (optional)

The script installs the XPU wheels from `https://download.pytorch.org/whl/xpu` automatically. They need the Level Zero GPU driver (`libze-intel-gpu1` + `libze1`) from the GPU step above. If that wheel install fails, the script falls back to CPU-only PyTorch and the OpenVINO paths are unaffected.

## Files in this repo

- `test_yolo_intel.sh` — the main diagnostic + benchmark script
- `bench_harness.py` — Ultralytics end-to-end benchmark harness (cold-load, first-infer, steady-state, video FPS) for `cpu`, `xpu:0`, `intel:cpu`, `intel:gpu`, `intel:npu`
- `ov_raw_bench.py` — raw OpenVINO Runtime harness (performance hints, async queue throughput, compile/cache timing)
- `ov_export.py` — OpenVINO IR export helper (FP32/FP16/INT8, handles both the new `quantize=` and legacy `half`/`int8` Ultralytics export APIs)
- `README.md` — this file

## Caveats

- On nano-size models the GPU is often **not** the fastest end-to-end device once pre/post-processing and NMS are counted — CPU or NPU can win. The medium-model rows are where the iGPU pulls ahead. Read your own numbers rather than assuming.
- `ov_raw_*`, `benchmark_app_*` and `onnxrt_*` rows exclude NMS and letterboxing. They are device ceilings, not application throughput.
- The script probes and benchmarks — it does not install kernel packages, the GPU compute runtime, or the NPU driver. Those remain manual one-time steps above.
