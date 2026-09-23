#!/usr/bin/env bash
#
# test_yolo_intel.sh
#
# Full diagnostic + benchmark suite for running Ultralytics YOLO on Intel
# Core Ultra (and older Core) laptops/desktops through the Intel AI stack:
# OpenVINO on CPU / iGPU / NPU, PyTorch XPU on the iGPU, and ONNX Runtime
# with the OpenVINO Execution Provider. It reports what actually works on
# your chip and how fast it runs, with real numbers.
#
#   Chip family                 Examples            iGPU                     NPU
#   -------------------------   -----------------   ----------------------   -------------------
#   Meteor Lake (Ultra 100)     155H/165H/125U      Arc Graphics (Xe-LPG)    NPU 3720 (~11 TOPS)
#   Arrow Lake (Ultra 200 H/U/S) 255H/285H/265K     Arc 140T/130T, Intel Gfx NPU 3720 (~13 TOPS)
#   Lunar Lake (Ultra 200V)     258V/268V/288V      Arc 140V/130V (Xe2)      NPU 4 (~48 TOPS)
#   Panther Lake (Ultra 300)    3xx                 Arc B390/B370 (Xe3)      NPU 5 (~50 TOPS)
#   Core 12th–14th gen          i7-13700H, ...      Iris Xe / UHD            none (CPU+GPU only)
#
# Model coverage: both YOLO11 and YOLO26 (nano/small/medium of each, 6
# models total by default), exported to OpenVINO IR in FP32 and FP16
# (INT8 optional with --int8), so you get generation-vs-generation AND
# precision-vs-precision comparisons on every device your chip exposes.
#
# Tests, in order:
#   0. System / driver sanity checks (kernel, i915/xe + intel_vpu modules,
#      /dev/dri/renderD*, /dev/accel/accel0, render group, Level Zero and
#      OpenCL runtimes, NPU firmware, chip auto-detection from lscpu)
#   1. Python env setup + download test image, models, and a real public
#      test video (Ultralytics' official CI demo asset)
#   2. Device visibility: OpenVINO available_devices (CPU/GPU/NPU) each
#      exercised with a tiny compiled matmul, plus PyTorch XPU visibility
#   3. OpenVINO IR export of every model in FP32 + FP16 (+ INT8 optional)
#   4. FULL BENCHMARK, every model x every backend that works:
#        - PyTorch CPU (baseline), PyTorch XPU (iGPU) if available
#        - OpenVINO CPU / GPU / NPU via Ultralytics device=intel:<dev>,
#          FP32 and FP16 IRs
#      For each: cold-load, first-inference (includes device compile),
#      steady-state avg/min/max/p95/stdev over 15 runs, real video FPS
#      (decode + preprocess + inference + NMS, up to 300 frames).
#   5. Raw OpenVINO Runtime characterisation per device: LATENCY vs
#      THROUGHPUT hints, pure inference latency floor, async-queue video
#      throughput, and cold vs cached compile time (model cache)
#   6. Intel's own benchmark_app (reference numbers per device)
#   7. ONNX Runtime + OpenVINO Execution Provider (CPU/GPU/NPU device_type)
#   8. NPU diagnosis: why the NPU is or isn't usable, with the exact fix
#   9. Summary report: pass/fail table, benchmark comparison tables (FPS /
#      latency / cold-load across all backends+models), and an env file
#      with the recommended device for Ultralytics
#
# Usage:
#   chmod +x test_yolo_intel.sh
#   ./test_yolo_intel.sh                # everything: YOLO11+YOLO26, n/s/m, FP32+FP16
#   ./test_yolo_intel.sh --quick        # nano only from each family
#   ./test_yolo_intel.sh --int8         # also export + benchmark INT8 IRs (PTQ on coco8)
#   ./test_yolo_intel.sh --skip-install # assume venv/deps already installed
#
# Safe to re-run. Does not touch system Python; creates an isolated venv
# at ~/yolo-intel-test/venv. Does NOT install the GPU compute runtime or
# the NPU driver — those are one-time system-level steps (see README);
# this script probes what's there and tells you what's missing.
#
# NPU NOTES (read before expecting the NPU stages to pass):
#   - The Intel NPU is reached through OpenVINO's NPU plugin. Ultralytics
#     supports it natively via device="intel:npu" on an exported OpenVINO
#     model — there is no PyTorch NPU device.
#   - Requirements: intel_vpu kernel module (in-tree since 6.8; 6.11+ for
#     Lunar/Arrow Lake, 6.17+ for Panther Lake recommended), /dev/accel/accel0
#     readable by your user (render group), and Intel's user-mode NPU driver
#     (intel-level-zero-npu + intel-driver-compiler-npu + intel-fw-npu +
#     level-zero) from https://github.com/intel/linux-npu-driver/releases
#   - NPU wants static shapes and prefers FP16/INT8. The FP16 IR is what
#     the NPU benchmarks below use. YOLO26 on NPU is only supported by
#     Ultralytics on Core Ultra 200V / 300 and newer — on Meteor/Arrow
#     Lake expect YOLO26-on-NPU to WARN, not the NPU to be broken.
#   - The first NPU compile of a model is slow (tens of seconds). That
#     shows up in FIRST_INFER_MS. Step 5 shows how much the OpenVINO
#     model cache saves on the second run.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$HOME/yolo-intel-test"
VENV_DIR="$WORKDIR/venv"
LOGDIR="$WORKDIR/logs"
RESULTS_FILE="$WORKDIR/results_summary.txt"
BENCH_CSV="$WORKDIR/benchmark_results.csv"
OV_CACHE="$WORKDIR/ov_cache"
SKIP_INSTALL=0
QUICK_MODE=0
INT8_MODE=0

for arg in "$@"; do
  case "$arg" in
    --skip-install) SKIP_INSTALL=1 ;;
    --quick) QUICK_MODE=1 ;;
    --int8) INT8_MODE=1 ;;
    -h|--help) sed -n '2,70p' "$0"; exit 0 ;;
  esac
done

# Ultralytics "AutoUpdate" pip-installs anything its requirement check cannot
# find. Accelerated ONNX Runtime builds ship under other distribution names
# (onnxruntime-openvino, onnxruntime-qnn, onnxruntime-rocm), so that check fails
# and AutoUpdate installs the plain CPU wheel straight over them - they all
# unpack into the same onnxruntime/ directory and the last one installed wins.
# Every later "accelerator" result is then a CPU result wearing the wrong label.
export YOLO_AUTOINSTALL=False

mkdir -p "$WORKDIR" "$LOGDIR" "$OV_CACHE"
: > "$RESULTS_FILE"
echo "backend,model,stage,metric,value_ms_or_fps" > "$BENCH_CSV"

# ---------- colors / helpers ----------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

section() { echo -e "\n${BLUE}==================================================================${NC}"; echo -e "${BLUE}  $1${NC}"; echo -e "${BLUE}==================================================================${NC}"; }
pass() { echo -e "${GREEN}[PASS]${NC} $1"; echo "[PASS] $1" >> "$RESULTS_FILE"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; echo "[FAIL] $1" >> "$RESULTS_FILE"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; echo "[WARN] $1" >> "$RESULTS_FILE"; }
info() { echo -e "${BLUE}[INFO]${NC} $1"; }

# --- ONNX Runtime integrity -------------------------------------------------
# Exactly one onnxruntime distribution may be installed: the CPU, and every
# accelerated build, unpack into the same onnxruntime/ directory, so a second
# one silently shadows the first. Call this after installs and again after any
# Ultralytics export, which is where an unwanted one tends to appear.
assert_ort_runtime() {
  local stage="${1:-check}"
  local dists
  dists=$(pip list --format=freeze 2>/dev/null | grep -ciE '^onnxruntime(-[a-z]+)?==' || true)
  if [ "${dists:-0}" -gt 1 ]; then
    warn "ONNX Runtime integrity ($stage): $dists onnxruntime distributions installed at once - $(pip list --format=freeze 2>/dev/null | grep -iE '^onnxruntime(-[a-z]+)?==' | tr '\n' ' '). They share one directory, so one is shadowing the other. Keeping onnxruntime-openvino."
    pip uninstall -y -q $(pip list --format=freeze 2>/dev/null | grep -ioE '^onnxruntime(-[a-z]+)?' | grep -iv "^onnxruntime-openvino$") >/dev/null 2>&1 || true
    pip install -q --force-reinstall --no-deps onnxruntime-openvino >/dev/null 2>&1 || true
  fi
  if [ -n "OpenVINOExecutionProvider" ]; then
    local eps
    eps=$(python3 -c "import onnxruntime as ort; print(','.join(ort.get_available_providers()))" 2>/dev/null || echo "")
    case "$eps" in
      *OpenVINOExecutionProvider*) pass "ONNX Runtime integrity ($stage): OpenVINOExecutionProvider present" ;;
      "") warn "ONNX Runtime integrity ($stage): onnxruntime not importable" ;;
      *)  warn "ONNX Runtime integrity ($stage): OpenVINOExecutionProvider is gone (have: $eps). Something replaced the accelerated build - any 'accelerator' ONNX result from here on would really be CPU." ;;
    esac
  fi
}


# ==================================================================
section "STEP 0: System / driver sanity checks"
# ==================================================================

echo "Kernel version:"
uname -r | tee "$LOGDIR/kernel_version.log"
KVER=$(uname -r | grep -oE '^[0-9]+\.[0-9]+')
KVER_MAJOR=$(echo "$KVER" | cut -d. -f1)
KVER_MINOR=$(echo "$KVER" | cut -d. -f2)
kver_ge() { [ "$KVER_MAJOR" -gt "$1" ] || { [ "$KVER_MAJOR" -eq "$1" ] && [ "$KVER_MINOR" -ge "$2" ]; }; }

echo -e "\nCPU model:"
CPU_MODEL=$(lscpu | grep "Model name" | sed 's/Model name:[[:space:]]*//')
echo "$CPU_MODEL" | tee "$LOGDIR/cpu_model.log"

# Best-effort chip family detection from the CPU marketing name. This is
# informational (drives kernel-version advice and NPU expectations); the
# ground truth for what's usable is OpenVINO's available_devices in Step 2.
DETECTED_CHIP="unknown"
EXPECTED_NPU="unknown"
MIN_KERNEL_NPU="6.8"
if echo "$CPU_MODEL" | grep -qE "Ultra [579] 1[0-9]{2}"; then
  DETECTED_CHIP="Meteor Lake (Core Ultra 100)"; EXPECTED_NPU="NPU 3720 (~11 TOPS)"; MIN_KERNEL_NPU="6.8"
elif echo "$CPU_MODEL" | grep -qE "Ultra [579] 2[0-9]{2}V"; then
  DETECTED_CHIP="Lunar Lake (Core Ultra 200V)"; EXPECTED_NPU="NPU 4 (~48 TOPS)"; MIN_KERNEL_NPU="6.11"
elif echo "$CPU_MODEL" | grep -qE "Ultra [579] 2[0-9]{2}"; then
  DETECTED_CHIP="Arrow Lake (Core Ultra 200 H/HX/U/S)"; EXPECTED_NPU="NPU 3720 (~13 TOPS)"; MIN_KERNEL_NPU="6.11"
elif echo "$CPU_MODEL" | grep -qE "Ultra [579] 3[0-9]{2}"; then
  DETECTED_CHIP="Panther Lake (Core Ultra 300)"; EXPECTED_NPU="NPU 5 (~50 TOPS)"; MIN_KERNEL_NPU="6.17"
elif echo "$CPU_MODEL" | grep -qE "i[3579]-1[2-4][0-9]{3}|Core\(TM\) [3579] [0-9]{3}"; then
  DETECTED_CHIP="Core 12th-14th gen (Alder/Raptor Lake)"; EXPECTED_NPU="none (CPU + Iris Xe/UHD GPU only)"
elif echo "$CPU_MODEL" | grep -qi "Intel"; then
  DETECTED_CHIP="Intel (unrecognised generation)"
fi
if [ "$DETECTED_CHIP" != "unknown" ]; then
  pass "Detected CPU family: $DETECTED_CHIP -> expected NPU: $EXPECTED_NPU"
else
  warn "Could not identify an Intel CPU family from '$CPU_MODEL' — will still probe every device below"
fi

echo -e "\nChecking Intel GPU kernel driver (i915 or xe)..."
if lsmod | grep -qE "^xe "; then
  pass "xe kernel module is loaded (new Intel GPU driver)"
  GPU_KMOD="xe"
elif lsmod | grep -qE "^i915 "; then
  pass "i915 kernel module is loaded"
  GPU_KMOD="i915"
else
  fail "Neither i915 nor xe kernel module is loaded — no Intel GPU driver active"
  GPU_KMOD="none"
fi

echo -e "\nChecking for /dev/dri/renderD* (GPU compute node)..."
RENDER_NODES=$(ls /dev/dri/renderD* 2>/dev/null | tr '\n' ' ')
if [ -n "$RENDER_NODES" ]; then
  pass "Render node(s) present: $RENDER_NODES"
  for rn in $RENDER_NODES; do
    if [ -r "$rn" ] && [ -w "$rn" ]; then
      pass "$rn is readable+writable by $USER"
    else
      warn "$rn is NOT accessible by $USER — add yourself to the render group: sudo gpasswd -a \$USER render (then log out/in)"
    fi
  done
else
  fail "No /dev/dri/renderD* node — GPU driver not bound, or running in a container without --device /dev/dri"
fi

echo -e "\nChecking render group membership for current user..."
if groups | grep -qE '\brender\b'; then
  pass "User is in 'render' group"
else
  warn "User NOT in 'render' group — run: sudo gpasswd -a \$USER render, then log out/in (needed for GPU and NPU)"
fi

echo -e "\nIdentifying iGPU via lspci..."
GPU_PCI_STRING=$(lspci -nn 2>/dev/null | grep -iE "VGA|Display|3D" | grep -i intel)
echo "${GPU_PCI_STRING:-<no Intel VGA/Display device in lspci>}" | tee "$LOGDIR/lspci_gpu.log"
[ -n "$GPU_PCI_STRING" ] && pass "Intel GPU found in lspci" || warn "No Intel GPU in lspci output"

echo -e "\nChecking OpenCL ICD + Level Zero GPU driver (what OpenVINO GPU plugin and PyTorch XPU need)..."
if [ -f /etc/OpenCL/vendors/intel.icd ]; then
  pass "Intel OpenCL ICD registered (/etc/OpenCL/vendors/intel.icd)"
else
  warn "Intel OpenCL ICD not registered — install intel-opencl-icd (OpenVINO GPU plugin needs OpenCL)"
fi
if ldconfig -p 2>/dev/null | grep -q "libze_loader.so"; then
  pass "Level Zero loader (libze_loader) present"
else
  warn "Level Zero loader missing — install libze1 (or level-zero) — needed for NPU and PyTorch XPU"
fi
if ldconfig -p 2>/dev/null | grep -qE "libze_intel_gpu"; then
  pass "Level Zero GPU driver (libze_intel_gpu) present"
else
  warn "Level Zero GPU driver missing — install libze-intel-gpu1 (needed by PyTorch XPU; OpenVINO GPU only needs OpenCL)"
fi

echo -e "\nRunning clinfo (OpenCL device listing)..."
if command -v clinfo >/dev/null 2>&1; then
  clinfo > "$LOGDIR/clinfo.log" 2>&1
  if grep -qiE "Device Name.*(Intel|Arc|Graphics)" "$LOGDIR/clinfo.log"; then
    pass "clinfo sees an Intel OpenCL device: $(grep -m1 -iE 'Device Name' "$LOGDIR/clinfo.log" | sed 's/.*Device Name[[:space:]]*//')"
  else
    warn "clinfo ran but lists no Intel GPU device — compute runtime not installed or render node not accessible (see $LOGDIR/clinfo.log)"
  fi
else
  warn "clinfo not installed (sudo apt install clinfo) — non-fatal, OpenVINO probes the GPU directly in Step 2"
fi

echo -e "\nChecking NPU kernel driver (intel_vpu) and /dev/accel/accel0..."
if lsmod | grep -q "^intel_vpu"; then
  pass "intel_vpu kernel module is loaded"
elif modinfo intel_vpu >/dev/null 2>&1; then
  warn "intel_vpu module exists but is not loaded — try: sudo modprobe intel_vpu (no NPU on this chip, or firmware missing)"
else
  warn "intel_vpu module not available in this kernel ($KVER) — NPU needs kernel >= $MIN_KERNEL_NPU for $DETECTED_CHIP"
fi
if [ -e /dev/accel/accel0 ]; then
  if [ -r /dev/accel/accel0 ] && [ -w /dev/accel/accel0 ]; then
    pass "/dev/accel/accel0 exists and is accessible by $USER"
  else
    warn "/dev/accel/accel0 exists but is NOT accessible by $USER — fix: sudo chown root:render /dev/accel/accel0 && sudo chmod g+rw /dev/accel/accel0 (and be in the render group)"
  fi
else
  warn "/dev/accel/accel0 missing — NPU kernel driver not bound (kernel too old, firmware missing, or no NPU on this chip)"
fi
if ls /lib/firmware/updates/intel/vpu/*.bin /lib/firmware/intel/vpu/*.bin >/dev/null 2>&1; then
  pass "NPU firmware present: $(ls /lib/firmware/updates/intel/vpu/*.bin /lib/firmware/intel/vpu/*.bin 2>/dev/null | head -3 | xargs -n1 basename | tr '\n' ' ')"
else
  warn "No NPU firmware under /lib/firmware/{updates/,}intel/vpu — install intel-fw-npu (from linux-npu-driver releases) or linux-firmware"
fi
if ldconfig -p 2>/dev/null | grep -qE "libze_intel_npu|libze_intel_vpu"; then
  pass "Level Zero NPU driver (libze_intel_npu) present"
else
  warn "Level Zero NPU user-mode driver missing — install intel-level-zero-npu + intel-driver-compiler-npu from https://github.com/intel/linux-npu-driver/releases"
fi
if kver_ge "${MIN_KERNEL_NPU%%.*}" "${MIN_KERNEL_NPU##*.}"; then
  pass "Kernel $KVER meets the >= $MIN_KERNEL_NPU recommendation for NPU on $DETECTED_CHIP"
else
  warn "Kernel $KVER is below the >= $MIN_KERNEL_NPU recommendation for NPU on $DETECTED_CHIP — consider: sudo apt install linux-generic-hwe-24.04 (or a newer kernel)"
fi

# ==================================================================
section "STEP 1: Python environment setup"
# ==================================================================

if [ "$SKIP_INSTALL" -eq 0 ]; then
  if [ ! -d "$VENV_DIR" ]; then
    info "Creating venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR" || { fail "venv creation failed — install python3-venv"; exit 1; }
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  pip install --upgrade pip -q

  info "Installing PyTorch (XPU build, for the Intel iGPU via torch.xpu)..."
  if pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu -q 2>>"$LOGDIR/torch_xpu_install.log"; then
    pass "PyTorch XPU build installed"
  else
    warn "PyTorch XPU wheel install failed (see $LOGDIR/torch_xpu_install.log) — falling back to CPU-only PyTorch"
    pip install torch torchvision -q && pass "PyTorch CPU build installed" || fail "PyTorch install failed"
  fi

  info "Installing ultralytics..."
  pip install ultralytics -q && pass "ultralytics installed" || fail "ultralytics install failed"

  info "Installing openvino (runtime + benchmark_app) and export deps..."
  pip install openvino onnx onnxslim -q && pass "openvino installed" || fail "openvino install failed"

  info "Installing onnxruntime-openvino (ONNX Runtime with the OpenVINO Execution Provider)..."
  # onnxruntime-openvino and plain onnxruntime install the same 'onnxruntime'
  # package namespace; remove the plain one first so the EP-enabled build wins.
  pip uninstall -y onnxruntime -q >/dev/null 2>&1 || true
  if pip install onnxruntime-openvino -q 2>>"$LOGDIR/ort_openvino_install.log"; then
    pass "onnxruntime-openvino installed"
  else
    warn "onnxruntime-openvino install failed (see $LOGDIR/ort_openvino_install.log) — installing plain onnxruntime; Step 7 will only have CPUExecutionProvider"
    pip install onnxruntime -q || true
  fi
  assert_ort_runtime "after install"
else
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  info "Skipping installs (--skip-install passed), using existing venv"
fi

echo -e "\nInstalled versions:"
python3 - <<'PYEOF' 2>&1 | tee "$LOGDIR/versions.log"
import importlib
for m in ("torch", "ultralytics", "openvino", "onnxruntime", "cv2", "numpy"):
    try:
        mod = importlib.import_module(m)
        v = getattr(mod, "__version__", None) or getattr(mod, "get_version", lambda: "?")()
        print(f"  {m:<12} {v}")
    except Exception as e:
        print(f"  {m:<12} NOT AVAILABLE ({e})")
PYEOF

# Test image
TEST_IMG="$WORKDIR/bus.jpg"
if [ ! -f "$TEST_IMG" ]; then
  info "Downloading sample test image..."
  python3 - "$TEST_IMG" <<'PYEOF'
import sys, urllib.request
url = "https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg"
try:
    urllib.request.urlretrieve(url, sys.argv[1])
    print("downloaded ok")
except Exception as e:
    print(f"download failed: {e}")
PYEOF
fi
[ -f "$TEST_IMG" ] && pass "Test image ready: $TEST_IMG" || fail "Test image missing — check network access to raw.githubusercontent.com"

# Model set: YOLO11 + YOLO26, nano/small/medium. --quick trims to nano only.
declare -a BENCH_MODEL_NAMES=("yolo11n" "yolo11s" "yolo11m" "yolo26n" "yolo26s" "yolo26m")
if [ "$QUICK_MODE" -eq 1 ]; then
  BENCH_MODEL_NAMES=("yolo11n" "yolo26n")
fi
for m in "${BENCH_MODEL_NAMES[@]}"; do
  MP="$WORKDIR/${m}.pt"
  if [ ! -f "$MP" ]; then
    info "Downloading ${m}.pt..."
    ( cd "$WORKDIR" && python3 - "$m" <<'PYEOF'
import sys
from ultralytics import YOLO
YOLO(f"{sys.argv[1]}.pt")
PYEOF
    )
    [ -f "${m}.pt" ] && mv -f "${m}.pt" "$MP" 2>/dev/null
  fi
  [ -f "$MP" ] && pass "Model ready: $MP" || fail "Could not download ${m}.pt"
done

# Real public test video: Ultralytics' official CI demo asset.
TEST_VIDEO="$WORKDIR/solutions_ci_demo.mp4"
TEST_VIDEO_URL="https://github.com/ultralytics/assets/releases/download/v0.0.0/solutions_ci_demo.mp4"
if [ ! -f "$TEST_VIDEO" ]; then
  info "Downloading public test video (Ultralytics official demo asset, ~2-3MB)..."
  if command -v wget >/dev/null 2>&1; then
    wget -q -O "$TEST_VIDEO" "$TEST_VIDEO_URL"
  else
    curl -sL -o "$TEST_VIDEO" "$TEST_VIDEO_URL"
  fi
  [ -s "$TEST_VIDEO" ] && pass "Test video downloaded: $TEST_VIDEO" || { rm -f "$TEST_VIDEO"; fail "Test video download failed — URL: $TEST_VIDEO_URL"; }
fi
if [ -f "$TEST_VIDEO" ]; then
  VIDEO_INFO=$(python3 - "$TEST_VIDEO" <<'PYEOF'
import sys, cv2
cap = cv2.VideoCapture(sys.argv[1])
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"frames={n} fps={fps:.1f} res={w}x{h}")
PYEOF
)
  info "Test video info: $VIDEO_INFO"
  VIDEO_ARG="$TEST_VIDEO"
else
  VIDEO_ARG=""
fi

# ==================================================================
section "STEP 2: Device visibility — OpenVINO (CPU/GPU/NPU) and PyTorch XPU"
# ==================================================================
# OpenVINO's Core().available_devices is the ground truth for what the
# Intel stack can actually use. For each listed device we compile and run
# a tiny matmul so "listed" also means "works" (a listed NPU with a
# firmware/driver mismatch fails right here rather than mid-benchmark).

OV_DEVICES_OUT=$(python3 - <<'PYEOF' 2>&1
import time
import numpy as np
try:
    import openvino as ov
    try:
        from openvino import opset13 as ops          # 2025.x / 2026.x
    except ImportError:
        from openvino.runtime import opset13 as ops  # <= 2024.x
except Exception as e:
    print(f"OV_IMPORT_ERROR={e}")
    raise SystemExit(0)
core = ov.Core()
print(f"OV_VERSION={ov.get_version()}")
devs = core.available_devices
print(f"OV_AVAILABLE={','.join(devs) if devs else 'none'}")
for d in devs:
    try:
        print(f"OV_NAME_{d}={core.get_property(d, 'FULL_DEVICE_NAME')}")
    except Exception as e:
        print(f"OV_NAME_{d}=<error: {e}>")
# tiny static matmul graph
a = ops.parameter([1, 256, 256], np.float32, name="a")
w = ops.constant(np.random.rand(256, 256).astype(np.float32))
m = ops.matmul(a, w, False, False)
model = ov.Model([m], [a], "probe")
for d in devs:
    base = d.split(".")[0]
    try:
        t0 = time.perf_counter()
        cm = core.compile_model(model, d)
        t1 = time.perf_counter()
        out = cm(np.random.rand(1, 256, 256).astype(np.float32))[0]
        t2 = time.perf_counter()
        print(f"OV_PROBE_{d}=OK compile={1000*(t1-t0):.1f}ms infer={1000*(t2-t1):.2f}ms sum={float(out.sum()):.1f}")
    except Exception as e:
        print(f"OV_PROBE_{d}=ERROR {str(e).splitlines()[0][:200]}")
PYEOF
)
echo "$OV_DEVICES_OUT" | tee "$LOGDIR/openvino_devices.log"

OV_HAS_CPU=0; OV_HAS_GPU=0; OV_HAS_NPU=0
if echo "$OV_DEVICES_OUT" | grep -q "OV_IMPORT_ERROR"; then
  fail "openvino import failed — see $LOGDIR/openvino_devices.log"
else
  OV_AVAIL=$(echo "$OV_DEVICES_OUT" | grep "^OV_AVAILABLE=" | cut -d= -f2)
  pass "OpenVINO $(echo "$OV_DEVICES_OUT" | grep '^OV_VERSION=' | cut -d= -f2-) available_devices: $OV_AVAIL"
  for dev in CPU GPU NPU; do
    line=$(echo "$OV_DEVICES_OUT" | grep -E "^OV_PROBE_${dev}(\.[0-9]+)?=" | head -1)
    name=$(echo "$OV_DEVICES_OUT" | grep -E "^OV_NAME_${dev}(\.[0-9]+)?=" | head -1 | cut -d= -f2-)
    if [ -z "$line" ]; then
      if [ "$dev" = "CPU" ]; then fail "OpenVINO CPU plugin not available (should always be present)";
      else warn "OpenVINO ${dev} plugin lists no device — see Step 0 warnings for the missing driver"; fi
    elif echo "$line" | grep -q "=OK"; then
      pass "OpenVINO ${dev} works: ${name} [$(echo "$line" | cut -d' ' -f2-)]"
      case "$dev" in CPU) OV_HAS_CPU=1;; GPU) OV_HAS_GPU=1;; NPU) OV_HAS_NPU=1;; esac
    else
      fail "OpenVINO ${dev} is listed (${name}) but a trivial compile/infer failed: $(echo "$line" | cut -d' ' -f2-)"
    fi
  done
fi

echo -e "\nChecking PyTorch XPU (torch.xpu) visibility..."
XPU_OUT=$(python3 - <<'PYEOF' 2>&1
import torch
print(f"TORCH_VERSION={torch.__version__}")
try:
    has = hasattr(torch, "xpu") and torch.xpu.is_available()
    print(f"XPU_AVAILABLE={has}")
    if has:
        print(f"XPU_DEVICE_NAME={torch.xpu.get_device_name(0)}")
        x = torch.rand(1024, 1024, device="xpu"); y = torch.rand(1024, 1024, device="xpu")
        z = (x @ y).sum().item()
        print(f"MATMUL_OK={z}")
except Exception as e:
    print(f"ERROR={e}")
PYEOF
)
echo "$XPU_OUT" > "$LOGDIR/pytorch_xpu.log"
XPU_OK=0
if echo "$XPU_OUT" | grep -q "MATMUL_OK="; then
  pass "PyTorch sees the Intel GPU via XPU and runs a matmul: $(echo "$XPU_OUT" | grep XPU_DEVICE_NAME | cut -d= -f2-)"
  XPU_OK=1
elif echo "$XPU_OUT" | grep -q "XPU_AVAILABLE=False"; then
  warn "PyTorch XPU not available ($(echo "$XPU_OUT" | grep TORCH_VERSION | cut -d= -f2)) — either a CPU-only torch wheel was installed, or the Level Zero GPU driver (libze-intel-gpu1) is missing. OpenVINO GPU path is independent of this."
else
  warn "PyTorch XPU errored — see $LOGDIR/pytorch_xpu.log"
fi

# ==================================================================
section "STEP 3: Export every model to OpenVINO IR (FP32 + FP16${INT8_MODE:+ + INT8})"
# ==================================================================
# Ultralytics writes <stem>_openvino_model/ regardless of precision, so
# each export is moved to <stem>_<prec>_openvino_model/ to keep them apart.
# ov_export.py tries quantize=16/8 (new Ultralytics) then half/int8
# (older) so the script works across versions.

export_ir() {
  local model_name="$1" prec="$2"
  local target="$WORKDIR/${model_name}_${prec}_openvino_model"
  if [ -f "$target/${model_name}.xml" ] || ls "$target"/*.xml >/dev/null 2>&1; then
    info "IR already exists: $target"; return 0
  fi
  info "Exporting ${model_name} -> OpenVINO ${prec^^} ..."
  ( cd "$WORKDIR" && python3 "$SCRIPT_DIR/ov_export.py" --model "${model_name}.pt" --precision "$prec" --target "$target" --imgsz 640 ) \
    > "$LOGDIR/ov_export_${model_name}_${prec}.log" 2>&1
  if ls "$target"/*.xml >/dev/null 2>&1; then
    pass "OpenVINO ${prec^^} IR: $target"
  else
    fail "OpenVINO ${prec^^} export failed for ${model_name} — see $LOGDIR/ov_export_${model_name}_${prec}.log"
    return 1
  fi
}

declare -a PRECISIONS=("fp32" "fp16")
[ "$INT8_MODE" -eq 1 ] && PRECISIONS+=("int8")
for m in "${BENCH_MODEL_NAMES[@]}"; do
  [ -f "$WORKDIR/${m}.pt" ] || continue
  for p in "${PRECISIONS[@]}"; do
    export_ir "$m" "$p"
  done
done

# ==================================================================
section "STEP 4: Full benchmark — every model x every working backend, image + video"
# ==================================================================
# bench_harness.py reports per run:
#   COLD_LOAD_MS, FIRST_INFER_MS (device compile lives here for OpenVINO
#   GPU/NPU), STEADY_* (avg/min/max/p95/stdev), VIDEO_* (real end-to-end FPS).
# Results go to $BENCH_CSV (backend,model,stage,metric,value).

run_benchmark() {
  local backend_label="$1" device_arg="$2" model_name="$3" model_path="$4" severity="${5:-fail}"

  [ -e "$model_path" ] || { warn "Model $model_path missing, skipping ${backend_label}/${model_name}"; return; }

  info "Benchmarking [$backend_label] model=$model_name device=$device_arg ..."
  local logfile="$LOGDIR/bench_${backend_label}_${model_name}.log"
  local OUT
  OUT=$(python3 "$SCRIPT_DIR/bench_harness.py" \
        --model "$model_path" \
        --device "$device_arg" \
        --image "$TEST_IMG" \
        ${VIDEO_ARG:+--video "$VIDEO_ARG"} \
        --backend-label "$backend_label" \
        --img-runs 15 \
        --video-max-frames 300 2>&1)
  echo "$OUT" > "$logfile"

  if echo "$OUT" | grep -q "^STEADY_AVG_MS="; then
    local cold first steady_avg steady_p95 vfps vms
    cold=$(echo "$OUT" | grep "^COLD_LOAD_MS=" | cut -d= -f2)
    first=$(echo "$OUT" | grep "^FIRST_INFER_MS=" | cut -d= -f2)
    steady_avg=$(echo "$OUT" | grep "^STEADY_AVG_MS=" | cut -d= -f2)
    steady_p95=$(echo "$OUT" | grep "^STEADY_P95_MS=" | cut -d= -f2)
    vfps=$(echo "$OUT" | grep "^VIDEO_AVG_FPS=" | cut -d= -f2)
    vms=$(echo "$OUT" | grep "^VIDEO_AVG_MS=" | cut -d= -f2)

    pass "[$backend_label/$model_name] cold-load ${cold}ms | first-infer ${first}ms | steady-state ${steady_avg}ms avg (p95 ${steady_p95}ms) | video ${vfps:-N/A} FPS (${vms:-N/A}ms/frame)"
    {
      echo "$backend_label,$model_name,cold_load,ms,$cold"
      echo "$backend_label,$model_name,first_infer,ms,$first"
      echo "$backend_label,$model_name,steady_avg,ms,$steady_avg"
      echo "$backend_label,$model_name,steady_p95,ms,$steady_p95"
      [ -n "$vfps" ] && echo "$backend_label,$model_name,video,fps,$vfps"
      [ -n "$vms" ] && echo "$backend_label,$model_name,video,ms_per_frame,$vms"
    } >> "$BENCH_CSV"
  else
    local err
    err=$(echo "$OUT" | grep -E "_ERROR=" | head -1 | cut -c1-160)
    if [ "$severity" = "warn" ]; then
      warn "[$backend_label/$model_name] did not run (${err:-see $logfile})"
    else
      fail "[$backend_label/$model_name] benchmark failed — ${err:-see $logfile}"
    fi
  fi
}

# --- PyTorch CPU baseline (always) ---
for m in "${BENCH_MODEL_NAMES[@]}"; do
  run_benchmark "pytorch_cpu" "cpu" "$m" "$WORKDIR/${m}.pt"
done

# --- PyTorch XPU on the iGPU (only if Step 2 saw it) ---
if [ "$XPU_OK" -eq 1 ]; then
  for m in "${BENCH_MODEL_NAMES[@]}"; do
    run_benchmark "pytorch_xpu" "xpu:0" "$m" "$WORKDIR/${m}.pt" warn
  done
else
  warn "Skipping PyTorch XPU benchmarks — torch.xpu not usable (Step 2)"
fi

# --- OpenVINO via Ultralytics device=intel:<dev>, each precision ---
for dev in cpu gpu npu; do
  case "$dev" in
    cpu) ok=$OV_HAS_CPU ;; gpu) ok=$OV_HAS_GPU ;; npu) ok=$OV_HAS_NPU ;;
  esac
  if [ "$ok" -ne 1 ]; then
    warn "Skipping OpenVINO ${dev^^} benchmarks — device not usable (Step 2)"
    continue
  fi
  for p in "${PRECISIONS[@]}"; do
    # NPU: FP32 IR is not the intended path (plugin runs FP16 internally and
    # compiles slower); benchmark it anyway but downgrade failures to WARN.
    sev="fail"; [ "$dev" = "npu" ] && sev="warn"
    for m in "${BENCH_MODEL_NAMES[@]}"; do
      run_benchmark "openvino_${dev}_${p}" "intel:${dev}" "$m" "$WORKDIR/${m}_${p}_openvino_model" "$sev"
    done
  done
done

# ==================================================================
section "STEP 5: Raw OpenVINO Runtime — LATENCY vs THROUGHPUT hints, compile cache"
# ==================================================================
# Uses the FP16 IR of the nano models on every working device. Two runs
# per device with the same CACHE_DIR: the second run's COMPILE_MS shows
# how much OpenVINO's model cache saves (matters most for NPU, then GPU).

run_raw() {
  local dev="$1" hint="$2" model_name="$3" cache="$4" label_suffix="$5"
  local ir="$WORKDIR/${model_name}_fp16_openvino_model"
  ls "$ir"/*.xml >/dev/null 2>&1 || { warn "No FP16 IR for $model_name, skipping raw bench"; return; }
  local label="ov_raw_${dev,,}_${hint,,}${label_suffix}"
  local logfile="$LOGDIR/${label}_${model_name}.log"
  local OUT
  OUT=$(python3 "$SCRIPT_DIR/ov_raw_bench.py" --model "$ir" --device "$dev" --hint "$hint" \
        ${VIDEO_ARG:+--video "$VIDEO_ARG"} --runs 15 --video-max-frames 300 \
        ${cache:+--cache-dir "$cache"} 2>&1)
  echo "$OUT" > "$logfile"
  if echo "$OUT" | grep -q "^STEADY_AVG_MS="; then
    local comp avg p95 vfps nreq mode
    comp=$(echo "$OUT" | grep "^COMPILE_MS=" | cut -d= -f2)
    avg=$(echo "$OUT" | grep "^STEADY_AVG_MS=" | cut -d= -f2)
    p95=$(echo "$OUT" | grep "^STEADY_P95_MS=" | cut -d= -f2)
    vfps=$(echo "$OUT" | grep "^VIDEO_AVG_FPS=" | cut -d= -f2)
    nreq=$(echo "$OUT" | grep "^OPTIMAL_NUM_REQUESTS=" | cut -d= -f2)
    mode=$(echo "$OUT" | grep "^VIDEO_MODE=" | cut -d= -f2)
    pass "[$label/$model_name] compile ${comp}ms | raw infer ${avg}ms avg (p95 ${p95}ms) | video ${vfps:-N/A} FPS (${mode:-n/a}, ${nreq} req) — no NMS"
    {
      echo "$label,$model_name,compile,ms,$comp"
      echo "$label,$model_name,steady_avg,ms,$avg"
      echo "$label,$model_name,steady_p95,ms,$p95"
      [ -n "$vfps" ] && echo "$label,$model_name,video,fps,$vfps"
    } >> "$BENCH_CSV"
  else
    fail "[$label/$model_name] raw OpenVINO bench failed — $(echo "$OUT" | grep -E '_ERROR=' | head -1 | cut -c1-160)"
  fi
}

for dev in CPU GPU NPU; do
  case "$dev" in CPU) ok=$OV_HAS_CPU ;; GPU) ok=$OV_HAS_GPU ;; NPU) ok=$OV_HAS_NPU ;; esac
  [ "$ok" -eq 1 ] || continue
  for m in yolo11n yolo26n; do
    [ -f "$WORKDIR/${m}.pt" ] || continue
    run_raw "$dev" LATENCY "$m" "" ""
    run_raw "$dev" THROUGHPUT "$m" "" ""
    if [ "$dev" != "CPU" ]; then
      # cold (populates cache) then warm (reads cache)
      rm -rf "$OV_CACHE/${dev}_${m}"
      run_raw "$dev" LATENCY "$m" "$OV_CACHE/${dev}_${m}" "_cache_cold"
      run_raw "$dev" LATENCY "$m" "$OV_CACHE/${dev}_${m}" "_cache_warm"
    fi
  done
done

# ==================================================================
section "STEP 6: Intel benchmark_app (OpenVINO's reference benchmark tool)"
# ==================================================================
# Runs Intel's own tool on the FP16 nano IRs for 10s per device so you
# have a number directly comparable to Intel's published figures.

if command -v benchmark_app >/dev/null 2>&1; then
  for dev in CPU GPU NPU; do
    case "$dev" in CPU) ok=$OV_HAS_CPU ;; GPU) ok=$OV_HAS_GPU ;; NPU) ok=$OV_HAS_NPU ;; esac
    [ "$ok" -eq 1 ] || continue
    for m in yolo11n yolo26n; do
      ir=$(ls "$WORKDIR/${m}_fp16_openvino_model"/*.xml 2>/dev/null | head -1)
      [ -n "$ir" ] || continue
      for hint in latency throughput; do
        logfile="$LOGDIR/benchmark_app_${dev}_${hint}_${m}.log"
        info "benchmark_app -d $dev -hint $hint -m $(basename "$ir") (10s)..."
        benchmark_app -m "$ir" -d "$dev" -hint "$hint" -t 10 > "$logfile" 2>&1
        thr=$(grep -oE "Throughput:\s*[0-9.]+" "$logfile" | grep -oE "[0-9.]+" | head -1)
        med=$(grep -oE "Median:\s*[0-9.]+" "$logfile" | grep -oE "[0-9.]+" | head -1)
        comp=$(grep -oE "Compile model took [0-9.]+" "$logfile" | grep -oE "[0-9.]+" | head -1)
        if [ -n "$thr" ]; then
          pass "[benchmark_app/${dev}/${hint}/${m}] throughput ${thr} FPS | median latency ${med:-?}ms | compile ${comp:-?}ms"
          {
            echo "benchmark_app_${dev,,}_${hint},${m},video,fps,$thr"
            [ -n "$med" ] && echo "benchmark_app_${dev,,}_${hint},${m},steady_avg,ms,$med"
            [ -n "$comp" ] && echo "benchmark_app_${dev,,}_${hint},${m},compile,ms,$comp"
          } >> "$BENCH_CSV"
        else
          fail "benchmark_app failed on ${dev} for ${m} (${hint}) — see $logfile"
        fi
      done
    done
  done
else
  warn "benchmark_app not on PATH (should ship with the openvino pip package) — skipping"
fi

# ==================================================================
section "STEP 7: ONNX Runtime + OpenVINO Execution Provider (CPU/GPU/NPU)"
# ==================================================================

if ! python3 -c "import onnxruntime" >/dev/null 2>&1; then
  warn "onnxruntime not importable — skipping ONNX Runtime tests"
else
  python3 -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())" > "$LOGDIR/ort_providers.log" 2>&1
  cat "$LOGDIR/ort_providers.log"
  assert_ort_runtime "before ORT stage"
  if grep -q "OpenVINOExecutionProvider" "$LOGDIR/ort_providers.log"; then
    pass "OpenVINOExecutionProvider is available to ONNX Runtime"
    ORT_OV=1
  else
    warn "OpenVINOExecutionProvider NOT available — only CPUExecutionProvider. Install onnxruntime-openvino (pip uninstall onnxruntime first)."
    ORT_OV=0
  fi

  for onnx_model_name in yolo11n yolo26n; do
    src_pt="$WORKDIR/${onnx_model_name}.pt"
    MODEL_ONNX="$WORKDIR/${onnx_model_name}.onnx"
    [ -f "$src_pt" ] || { warn "$src_pt missing, skipping ONNX test for $onnx_model_name"; continue; }
    if [ ! -f "$MODEL_ONNX" ]; then
      info "Exporting ${onnx_model_name} to ONNX..."
      ( cd "$WORKDIR" && python3 - "$onnx_model_name" <<'PYEOF'
import sys
from ultralytics import YOLO
YOLO(f"{sys.argv[1]}.pt").export(format="onnx", imgsz=640, dynamic=False, simplify=True)
PYEOF
      ) > "$LOGDIR/onnx_export_${onnx_model_name}.log" 2>&1
    fi
    [ -f "$MODEL_ONNX" ] && pass "ONNX model ready: $MODEL_ONNX" || { fail "ONNX export failed for $onnx_model_name — see $LOGDIR/onnx_export_${onnx_model_name}.log"; continue; }

    for dev in CPU GPU NPU; do
      case "$dev" in CPU) ok=$OV_HAS_CPU ;; GPU) ok=$OV_HAS_GPU ;; NPU) ok=$OV_HAS_NPU ;; esac
      [ "$ok" -eq 1 ] || continue
      if [ "$ORT_OV" -eq 0 ] && [ "$dev" != "CPU" ]; then continue; fi

      info "ONNX Runtime ${onnx_model_name} via OpenVINO EP device_type=${dev} (raw tensor + video, no NMS)..."
      OUT=$(python3 - "$MODEL_ONNX" "${VIDEO_ARG:-}" "$dev" "$ORT_OV" "$OV_CACHE/ort_${dev}" <<'PYEOF' 2>&1
import sys, time, statistics, os
import numpy as np
import onnxruntime as ort
model_onnx, video_path, dev, use_ov, cache = sys.argv[1:6]
try:
    if use_ov == "1":
        os.makedirs(cache, exist_ok=True)
        providers = [("OpenVINOExecutionProvider", {"device_type": dev, "cache_dir": cache}), "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    t0 = time.perf_counter()
    sess = ort.InferenceSession(model_onnx, providers=providers)
    print(f"SESSION_CREATE_MS={(time.perf_counter()-t0)*1000:.1f}")
    print(f"ACTUAL_PROVIDERS={sess.get_providers()}")
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    if len(shape) == 4 and shape[2] == 1: shape = [1, 3, 640, 640]
    dummy = np.random.rand(*shape).astype(np.float32)
    t0 = time.perf_counter(); sess.run(None, {inp.name: dummy})
    print(f"FIRST_INFER_MS={(time.perf_counter()-t0)*1000:.2f}")
    times = []
    for _ in range(15):
        t0 = time.perf_counter(); sess.run(None, {inp.name: dummy}); times.append((time.perf_counter()-t0)*1000)
    print(f"STEADY_AVG_MS={statistics.mean(times):.2f}")
    print(f"STEADY_MIN_MS={min(times):.2f}")
    print(f"STEADY_MAX_MS={max(times):.2f}")
    if video_path:
        import cv2
        h, w = shape[2], shape[3]
        cap = cv2.VideoCapture(video_path); n = 0; ft = []; ts = time.perf_counter()
        while n < 300:
            ok, frame = cap.read()
            if not ok: break
            t = cv2.resize(frame, (w, h)).transpose(2, 0, 1)[np.newaxis].astype(np.float32) / 255.0
            t0 = time.perf_counter(); sess.run(None, {inp.name: t}); ft.append((time.perf_counter()-t0)*1000); n += 1
        tt = time.perf_counter() - ts; cap.release()
        if n:
            print(f"VIDEO_FRAMES={n}"); print(f"VIDEO_AVG_FPS={n/tt:.2f}"); print(f"VIDEO_AVG_MS={statistics.mean(ft):.2f}")
except Exception as e:
    print(f"ERROR={str(e).splitlines()[0][:300]}")
PYEOF
)
      echo "$OUT" > "$LOGDIR/ort_openvino_${dev}_${onnx_model_name}.log"
      label="onnxrt_openvino_${dev,,}"
      [ "$ORT_OV" -eq 0 ] && label="onnxrt_cpu"
      if echo "$OUT" | grep -q "^STEADY_AVG_MS="; then
        AVG=$(echo "$OUT" | grep "^STEADY_AVG_MS=" | cut -d= -f2)
        FIRST=$(echo "$OUT" | grep "^FIRST_INFER_MS=" | cut -d= -f2)
        SESS=$(echo "$OUT" | grep "^SESSION_CREATE_MS=" | cut -d= -f2)
        VFPS=$(echo "$OUT" | grep "^VIDEO_AVG_FPS=" | cut -d= -f2)
        if [ "$ORT_OV" -eq 1 ] && ! echo "$OUT" | grep "ACTUAL_PROVIDERS" | grep -q "OpenVINO"; then
          warn "[$label/$onnx_model_name] session silently fell back to CPUExecutionProvider — see $LOGDIR/ort_openvino_${dev}_${onnx_model_name}.log"
        else
          pass "[$label/$onnx_model_name] session ${SESS}ms | first-infer ${FIRST}ms | steady ${AVG}ms | video ${VFPS:-N/A} FPS (raw tensor, no NMS)"
          {
            echo "$label,${onnx_model_name},cold_load,ms,$SESS"
            echo "$label,${onnx_model_name},first_infer,ms,$FIRST"
            echo "$label,${onnx_model_name},steady_avg,ms,$AVG"
            [ -n "$VFPS" ] && echo "$label,${onnx_model_name},video,fps,$VFPS"
          } >> "$BENCH_CSV"
        fi
      else
        sev=fail; [ "$dev" = "NPU" ] && sev=warn
        $sev "[$label/$onnx_model_name] ONNX Runtime inference failed — $(echo "$OUT" | grep '^ERROR=' | cut -c1-200)"
      fi
    done
  done
fi

# ==================================================================
section "STEP 8: NPU diagnosis"
# ==================================================================
# Pull the Step 0 / Step 2 facts together into one verdict with the fix.

if [ "$OV_HAS_NPU" -eq 1 ]; then
  pass "NPU is usable: OpenVINO NPU plugin compiled and ran. See openvino_npu_* rows above for YOLO numbers."
  if grep -q "^\[WARN\] \[openvino_npu_.*yolo26" "$RESULTS_FILE"; then
    warn "YOLO26 on NPU did not run — Ultralytics only supports YOLO26-on-NPU on Core Ultra 200V/300 and newer; YOLO11 is the NPU model for $DETECTED_CHIP"
  fi
elif [ "$EXPECTED_NPU" = "none (CPU + Iris Xe/UHD GPU only)" ]; then
  info "This CPU generation has no NPU — nothing to fix. GPU + CPU are the available accelerators."
else
  echo "NPU expected on $DETECTED_CHIP ($EXPECTED_NPU) but not usable. Checking why:"
  if [ ! -e /dev/accel/accel0 ]; then
    if ! modinfo intel_vpu >/dev/null 2>&1; then
      warn "ROOT CAUSE: kernel $KVER has no intel_vpu module. Install a newer kernel (>= $MIN_KERNEL_NPU): sudo apt install linux-generic-hwe-24.04 (or OEM kernel), then reboot."
    elif ! ls /lib/firmware/updates/intel/vpu/*.bin /lib/firmware/intel/vpu/*.bin >/dev/null 2>&1; then
      warn "ROOT CAUSE: intel_vpu exists but NPU firmware is missing, so the device never bound. Install intel-fw-npu from the linux-npu-driver release, then: sudo rmmod intel_vpu; sudo modprobe intel_vpu"
    else
      warn "intel_vpu + firmware present but /dev/accel/accel0 never appeared — check: sudo dmesg | grep -i vpu (firmware/driver version mismatch is the usual cause; match intel-fw-npu to intel-level-zero-npu versions)"
    fi
  elif [ ! -r /dev/accel/accel0 ] || [ ! -w /dev/accel/accel0 ]; then
    warn "ROOT CAUSE: /dev/accel/accel0 exists but $USER cannot open it. Fix: sudo gpasswd -a \$USER render; sudo chown root:render /dev/accel/accel0; sudo chmod g+rw /dev/accel/accel0; then log out/in. Make it persistent with a udev rule: SUBSYSTEM==\"accel\", KERNEL==\"accel*\", GROUP=\"render\", MODE=\"0660\""
  elif ! ldconfig -p 2>/dev/null | grep -qE "libze_intel_npu|libze_intel_vpu"; then
    warn "ROOT CAUSE: kernel side is fine, the user-mode NPU driver is missing. Install from https://github.com/intel/linux-npu-driver/releases: intel-level-zero-npu, intel-driver-compiler-npu, intel-fw-npu (+ level-zero, libtbb12)."
  elif ! ldconfig -p 2>/dev/null | grep -q "libze_loader.so"; then
    warn "ROOT CAUSE: Level Zero loader (libze1 / level-zero) missing — the NPU driver is installed but nothing can load it."
  else
    warn "Everything looks installed but OpenVINO still does not list NPU. Likely a version mismatch between the openvino pip package and the NPU driver/firmware. Check: sudo dmesg | grep -i vpu, and the openvino_devices.log — match OpenVINO version to the driver release notes."
  fi
  echo "--- dmesg (intel_vpu) ---" | tee "$LOGDIR/dmesg_vpu.log"
  (dmesg 2>/dev/null || journalctl -k --no-pager 2>/dev/null) | grep -iE "vpu|npu|accel" | tail -20 | tee -a "$LOGDIR/dmesg_vpu.log"
fi

# ==================================================================
section "SUMMARY"
# ==================================================================

echo -e "\nFull results log: $RESULTS_FILE"
echo -e "Per-test logs in: $LOGDIR"
echo -e "Full benchmark numbers (CSV): $BENCH_CSV\n"

PASS_COUNT=$(grep -c "^\[PASS\]" "$RESULTS_FILE" || true)
FAIL_COUNT=$(grep -c "^\[FAIL\]" "$RESULTS_FILE" || true)
WARN_COUNT=$(grep -c "^\[WARN\]" "$RESULTS_FILE" || true)
echo -e "${GREEN}Passed: $PASS_COUNT${NC}  ${YELLOW}Warnings: $WARN_COUNT${NC}  ${RED}Failed: $FAIL_COUNT${NC}\n"
cat "$RESULTS_FILE"

table() {  # $1 = stage,metric filter ; $2 = title ; $3 = column header
  echo -e "\n${BLUE}--------------------------------------------------------------${NC}"
  echo -e "${BLUE}  $2${NC}"
  echo -e "${BLUE}--------------------------------------------------------------${NC}"
  printf "%-34s %-10s %12s\n" "BACKEND" "MODEL" "$3"
  grep ",$1," "$BENCH_CSV" | sort -t, -k2,2 -k1,1 | while IFS=, read -r backend model stage metric value; do
    printf "%-34s %-10s %12s\n" "$backend" "$model" "$value"
  done
}
table "video,fps" "VIDEO FPS (end-to-end for pytorch_*/openvino_*; raw tensor for ov_raw_*/onnxrt_*/benchmark_app_*) — higher is better" "VIDEO_FPS"
table "steady_avg,ms" "STEADY-STATE LATENCY (single image, ms/frame) — lower is better" "MS/FRAME"
table "first_infer,ms" "FIRST INFERENCE (includes GPU/NPU model compile, ms) — one-time cost" "MS"
table "cold_load,ms" "COLD LOAD (model constructor / session create, ms)" "MS"
table "compile,ms" "OPENVINO COMPILE TIME (raw runtime; *_cache_warm rows show model-cache benefit)" "MS"

echo -e "\n${BLUE}--------------------------------------------------------------${NC}"

# Recommendation: pick the fastest end-to-end video FPS among the
# Ultralytics-driven backends for the smallest model that ran.
ENV_FILE="$WORKDIR/yolo_intel_env.sh"
: > "$ENV_FILE"
echo "# Auto-generated by test_yolo_intel.sh on $(date)" >> "$ENV_FILE"
echo "# Source this before running YOLO: source $ENV_FILE" >> "$ENV_FILE"

BEST_LINE=$(grep -E "^(pytorch_cpu|pytorch_xpu|openvino_(cpu|gpu|npu)_(fp32|fp16|int8)),yolo(11|26)n,video,fps," "$BENCH_CSV" | sort -t, -k5,5 -gr | head -1)
BEST_BACKEND=$(echo "$BEST_LINE" | cut -d, -f1)
BEST_MODEL=$(echo "$BEST_LINE" | cut -d, -f2)
BEST_FPS=$(echo "$BEST_LINE" | cut -d, -f5)

case "$BEST_BACKEND" in
  openvino_gpu_*)  REC_DEVICE="intel:gpu" ;;
  openvino_npu_*)  REC_DEVICE="intel:npu" ;;
  openvino_cpu_*)  REC_DEVICE="intel:cpu" ;;
  pytorch_xpu)     REC_DEVICE="xpu:0" ;;
  *)               REC_DEVICE="cpu" ;;
esac
REC_PREC=$(echo "$BEST_BACKEND" | grep -oE "fp32|fp16|int8" || echo "")

if [ -n "$BEST_BACKEND" ]; then
  echo -e "${GREEN}RECOMMENDATION:${NC} fastest end-to-end nano-model backend on this machine: ${BEST_BACKEND} (${BEST_MODEL}, ${BEST_FPS} FPS)"
  {
    echo "export ULTRALYTICS_DEVICE=$REC_DEVICE"
    if [ -n "$REC_PREC" ]; then
      echo "# Export models with the matching precision, then run on the OpenVINO device:"
      case "$REC_PREC" in
        fp16) echo "#   yolo export model=yolo11n.pt format=openvino quantize=16   # (half=True on older ultralytics)" ;;
        int8) echo "#   yolo export model=yolo11n.pt format=openvino quantize=8 data=coco8.yaml   # (int8=True on older ultralytics)" ;;
        *)    echo "#   yolo export model=yolo11n.pt format=openvino" ;;
      esac
      echo "#   yolo predict model=yolo11n_openvino_model source=your_image.jpg device=\$ULTRALYTICS_DEVICE"
    else
      echo "#   yolo predict model=yolo11n.pt source=your_image.jpg device=\$ULTRALYTICS_DEVICE"
    fi
    echo "export OV_CACHE_DIR=$OV_CACHE   # OpenVINO model cache (skips GPU/NPU recompiles); pass as cache_dir / CACHE_DIR in your own code"
  } >> "$ENV_FILE"
  echo "Device to use with Ultralytics: device=$REC_DEVICE"
else
  echo -e "${YELLOW}RECOMMENDATION:${NC} no benchmark completed — fix the FAIL items above first."
  echo "# No backend completed a benchmark; nothing to recommend yet." >> "$ENV_FILE"
fi

if [ "$OV_HAS_GPU" -eq 1 ] && [ "$REC_DEVICE" != "intel:gpu" ]; then
  echo "Note: OpenVINO GPU works but wasn't the fastest for nano models — on small models CPU/NPU can win once pre/post-processing overhead is counted; check the medium-model rows, GPU usually pulls ahead there."
fi
if [ "$OV_HAS_NPU" -eq 1 ]; then
  echo -e "${GREEN}NPU:${NC} usable. Best for sustained low-power inference; use the FP16 (or INT8) IR and keep a CACHE_DIR to avoid the slow first compile."
fi

echo -e "${BLUE}--------------------------------------------------------------${NC}\n"
echo -e "${BLUE}Env file written to: $ENV_FILE${NC}"
echo "---"; cat "$ENV_FILE"; echo "---"

echo -e "\nTo use these every session, either:"
echo "  A) source it manually each time:   source $ENV_FILE"
echo "  B) make it permanent by appending to ~/.bashrc"
echo ""
if [ -t 0 ]; then
  read -r -p "Append the export lines to ~/.bashrc now? [y/N] " APPEND_CHOICE
else
  APPEND_CHOICE="n"
  info "Non-interactive shell detected — skipping the ~/.bashrc prompt. Run 'source $ENV_FILE' manually, or re-run interactively to be asked."
fi
if [[ "$APPEND_CHOICE" =~ ^[Yy]$ ]]; then
  {
    echo ""
    echo "# --- Added by test_yolo_intel.sh on $(date) ---"
    grep '^export' "$ENV_FILE"
    echo "# --- end test_yolo_intel.sh block ---"
  } >> "$HOME/.bashrc"
  pass "Appended export lines to ~/.bashrc — restart your shell or run: source ~/.bashrc"
else
  info "Skipped. Run 'source $ENV_FILE' before using YOLO, or append it to your shell profile yourself later."
fi
