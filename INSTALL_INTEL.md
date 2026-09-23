# Intel Core Ultra — the stack this suite needs, and the traps in it

`test_yolo_intel.sh` benchmarks *on top of* an existing Intel AI stack. This file
records what that stack has to be, how to build it, and the specific ways it goes
wrong. The failure modes below are not hypothetical: they are the same ones that
were found and fixed on the AMD sibling of this suite
([ryzen_yolo](https://github.com/ZephyrSai/ryzen_yolo)), and the Intel path has
the identical shape.

> **Verification status.** The *suite fixes* (single ONNX Runtime distribution,
> AutoUpdate off, reporting the device that actually ran) were validated on AMD
> hardware, where each bug was reproduced and then confirmed fixed. The *Intel
> install steps and version numbers* below are from Intel's own release
> documentation, checked against their repositories on 23 Sep 2026, but have
> **not** been run on a Core Ultra machine. Run §6 first; it tells you in one
> pass whether this document matches your box, and every number it prints is
> measured on your hardware rather than taken from here.

---

## 1. What has to be true

| Layer | What you need | Check |
|---|---|---|
| Kernel | 6.8+ Meteor Lake, **6.11+** Arrow/Lunar Lake, 6.17+ Panther Lake. Driver v1.38.0 is validated on Ubuntu 24.04 and 26.04 | `uname -r` |
| iGPU driver | `i915` (Meteor Lake) or `xe` (Lunar Lake and later) | `lsmod \| grep -E '^(i915\|xe)'` |
| GPU compute | Intel compute runtime — OpenCL ICD + Level Zero | `clinfo -l`, `ls /dev/dri/renderD*` |
| NPU kernel driver | `intel_vpu`, exposing `/dev/accel/accel0` | `lsmod \| grep intel_vpu` |
| NPU user driver | `intel-driver-compiler-npu`, `intel-fw-npu`, `intel-level-zero-npu` | `ls /dev/accel/accel0` |
| Permissions | your user in `render` (and `video`) | `id -nG` |
| OpenVINO | **match the NPU driver** - 2026.3.1 for driver v1.38.0 (§2.2) | `python -c "import openvino; print(openvino.__version__)"` |
| ONNX Runtime | **`onnxruntime-openvino`** 1.24.1 — and nothing else | see §3 |

Ground truth for "does the device work" is never `lsmod`; it is
`openvino.Core().available_devices` plus an actual compiled inference on each
device. The suite does exactly that in its device-visibility step.

---

## 2. Procedure (Ubuntu 24.04)

### 2.1 GPU compute runtime

Ubuntu's stock Mesa gives you display and media, **not** OpenCL/Level Zero
compute. Install Intel's runtime:

```bash
sudo apt update
sudo apt install -y intel-opencl-icd intel-level-zero-gpu level-zero clinfo
sudo usermod -aG render,video "$USER"    # log out and back in
clinfo -l                                 # expect an Intel Graphics platform
```

For a newer runtime than the distro ships (needed for Lunar/Panther Lake), take
the `.deb`s from [intel/compute-runtime releases](https://github.com/intel/compute-runtime/releases)
— 26.35.39758.10 as of 17 Sep 2026 — and `sudo dpkg -i *.deb` them together so
their inter-dependencies resolve in one transaction.

### 2.2 NPU (Intel AI Boost)

The kernel module is called `intel_vpu` — the NPU's old name was VPU. It is
in-tree from 6.8, so on a current Ubuntu kernel you only need the **user-mode**
driver, which is *not* packaged by Ubuntu:

```bash
# 1. remove anything from a previous attempt (mismatched versions are the
#    number one cause of "NPU listed but every inference fails")
sudo dpkg --purge --force-remove-reinstreq \
  intel-driver-compiler-npu intel-fw-npu intel-level-zero-npu 2>/dev/null || true

# 2. the compiler depends on TBB; skipping this gives a "missing shared object"
sudo apt update && sudo apt install -y libtbb12

# 3. install the release tarball's debs together, not one by one
wget https://github.com/intel/linux-npu-driver/releases/download/v1.38.0/linux-npu-driver-v1.38.0.20260910-34487311128-ubuntu2404.tar.gz
tar -xf linux-npu-driver-v1.38.0*.tar.gz
sudo dpkg -i *.deb

# 4. the device node is root:render by default
sudo chown root:render /dev/accel/accel0 && sudo chmod g+rw /dev/accel/accel0
sudo reboot
```

After the reboot:

```bash
ls -l /dev/accel/accel0                  # group render, group-writable
dmesg | grep -i vpu | tail               # firmware loaded, no errors
python -c "import openvino; print(openvino.Core().available_devices)"
# expect something like ['CPU', 'GPU', 'NPU']
```

**Match the versions — this is the most common NPU failure.** Intel validates
each NPU driver release against one specific OpenVINO and Level Zero pair. For
**v1.38.0** (10 Sep 2026) that is:

| Component | Version |
|---|---|
| NPU driver | v1.38.0.20260910 |
| OpenVINO | **2026.3.1** |
| Level Zero | v1.32.0 |
| NPU compiler | `npu_ud_2026_38_rc1` |
| Validated OS | Ubuntu 24.04 LTS and 26.04 LTS |

So pin OpenVINO to the driver, not to "latest":

```bash
pip install "openvino==2026.3.1"
```

A driver much older or newer than your OpenVINO is the usual reason the NPU
appears in `available_devices` and then fails to compile a model. If you upgrade
one, check the other release's notes and upgrade both.

### 2.3 Python environment

```bash
sudo apt install -y python3-venv python3-pip
python3 -m venv ~/yolo-intel-test/venv && source ~/yolo-intel-test/venv/bin/activate
pip install -U pip
pip install ultralytics "openvino==2026.3.1" "onnx>=1.12,<2" onnxslim

# PyTorch XPU (optional; only for the pytorch_xpu rows)
pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu

# ONNX Runtime: the OpenVINO EP build, and NOTHING ELSE (see §3)
pip uninstall -y onnxruntime onnxruntime-openvino
pip install onnxruntime-openvino
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# must contain OpenVINOExecutionProvider
```

### 2.4 Run the suite

```bash
./test_yolo_intel.sh            # full run
./test_yolo_intel.sh --quick    # nano models only
./test_yolo_intel.sh --int8     # adds INT8 IRs — the realistic NPU precision
```

---

## 3. The trap that silently fakes your results

`onnxruntime`, `onnxruntime-openvino`, `onnxruntime-qnn` and `onnxruntime-rocm`
are different distributions that all unpack into the **same `onnxruntime/`
directory**. Install two and the second one's `.so` files shadow the first's.
Nothing errors. `import onnxruntime` works. You simply get whichever build won.

The way it happens without you asking for it: Ultralytics' exporter runs a
requirement check for a distribution *literally named* `onnxruntime`. With
`onnxruntime-openvino` installed that check fails, so **AutoUpdate pip-installs
the plain CPU wheel over your OpenVINO build** in the middle of an export. From
that point every "OpenVINO EP" result is a CPU result.

On the AMD suite this produced ONNX numbers that were reported as GPU for weeks
and were entirely CPU. Two defences, both now in this suite:

```bash
export YOLO_AUTOINSTALL=False        # Ultralytics may not install anything
pip list | grep -ci '^onnxruntime'   # must be exactly 1
```

plus `assert_ort_runtime` in the script, which re-checks after the install phase
and again after exports, and warns loudly if `OpenVINOExecutionProvider`
disappears. Related: because AutoUpdate is off, `onnx` and `onnxslim` must be
installed explicitly, as §2.3 does.

**Never trust a reported device.** Ask the session what actually ran:

```python
sess = ort.InferenceSession(m, providers=["OpenVINOExecutionProvider", "CPUExecutionProvider"])
print(sess.get_providers()[0])       # the truth; may be CPUExecutionProvider
```

The same applies to OpenVINO itself: a compiled model exposes
`EXECUTION_DEVICES`, which is what to report — `AUTO` and `HETERO` will place
work somewhere other than you asked.

---

## 4. Verifying the silicon actually ran

`lib/accel_verify.py` reads the kernel's own accounting rather than the
runtime's claim:

```bash
python3 lib/accel_verify.py --list                       # what this box exposes
python3 lib/accel_verify.py --pid <pid> --seconds 5      # watch a running job
python3 lib/accel_verify.py --seconds 8 -- python infer.py --device npu
```

On Intel it reads:

- **iGPU** — `/proc/<pid>/fdinfo/*`, `drm-engine-render` / `drm-engine-compute`
  (i915 and xe both expose this). Per process and exact.
- **NPU** — `npu_busy_time_us` under the `intel_vpu` device in sysfs. Device-wide,
  so it cannot separate your process from another user of the NPU.

Exit status is 0 when an engine was measurably busy, 1 when nothing moved. A
"GPU" or "NPU" claim with exit status 1 is a claim you should not publish.

---

## 5. Measuring fairly

`lib/fair_compare.py` compares every backend that exists on the machine, and
`lib/fairness.py` holds the methodology. Three effects make the naive approach
give wrong answers, all measured on the AMD sibling box:

1. **Thermal drift** — the same workload measured 16.7 ms early in a session and
   24.4 ms later. Backends measured in sequence are not measured equally.
2. **Shared power budget** — CPU and iGPU draw from one package limit.
   Interleaving CPU and GPU measurements made the CPU 45% slower than measuring
   it alone. Intel's Core Ultra parts behave the same way, and the NPU's whole
   selling point is that it is the cheap unit — so measure it both ways and say
   which you are reporting.
3. **One-shot numbers** — a single mean over 15 runs was repeatable to ±40%
   between sessions and looked authoritative.

So: rounds, medians of medians, reported spread, and a loud flag when the spread
exceeds 15%. `--mode isolated` answers "which backend should I use"; `--mode
interleaved` answers "what happens when they run together".

```bash
python3 lib/fair_compare.py --model yolo11n --workdir ~/yolo-intel-test
python3 lib/fair_compare.py --model yolo11n --mode interleaved
```

---

## 6. First run on a new machine — the short version

```bash
uname -r; lsmod | grep -E '^(i915|xe|intel_vpu)'; id -nG | tr ' ' '\n' | grep -E 'render|video'
ls -l /dev/dri/renderD* /dev/accel/accel0 2>&1
clinfo -l 2>/dev/null | head -3
python3 -c "import openvino as ov; c=ov.Core(); print([(d, c.get_property(d,'FULL_DEVICE_NAME')) for d in c.available_devices])"
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
pip list | grep -i '^onnxruntime'        # exactly one line
python3 lib/accel_verify.py --list
./test_yolo_intel.sh --quick
```

If `available_devices` lacks `GPU`, §2.1 is incomplete. If it lacks `NPU`, §2.2
is. If `get_available_providers()` lacks `OpenVINOExecutionProvider`, §3 has
already happened to you.

---

## 7. NPU notes specific to YOLO

- The NPU wants **quantised, static-shape** models. FP32 IRs will often compile
  and run, but the NPU's advantage only appears at INT8 — run the suite with
  `--int8`, which builds INT8 IRs via post-training quantisation on `coco8`.
- Dynamic shapes are not supported; export with `dynamic=False` (the suite does).
- First inference on the NPU includes a model compile that can dominate a short
  benchmark. The suite separates cold-load, first-inference and steady-state for
  exactly this reason, and measures cold vs cached compile with OpenVINO's model
  cache.
- If the NPU is listed but every compile fails, suspect the version pairing in
  §2.2 before suspecting the model.

---

## 8. Files

- `INSTALL_INTEL.md` — this document
- `lib/accel_verify.py` — prove an accelerator was busy (kernel counters)
- `lib/fairness.py`, `lib/fair_compare.py` — measurement methodology
- `test_yolo_intel.sh` — the suite
- `bench_harness.py`, `ov_raw_bench.py`, `ov_export.py` — per-stage harnesses

---

## 9. Provenance of the version claims

Everything in §1 and §2 is from vendor sources, fetched and checked on
**23 Sep 2026**. It has **not** been executed on a Core Ultra machine.

| Claim | Source |
|---|---|
| NPU driver v1.38.0, OpenVINO 2026.3.1, Level Zero 1.32.0, compiler `npu_ud_2026_38_rc1`, Ubuntu 24.04/26.04 | [intel/linux-npu-driver v1.38.0 release notes](https://github.com/intel/linux-npu-driver/releases/tag/v1.38.0) |
| `libtbb12` required by `intel-driver-compiler-npu` | same release notes |
| Compute runtime 26.35.39758.10 (17 Sep 2026) | [intel/compute-runtime releases](https://github.com/intel/compute-runtime/releases) |
| `onnxruntime-openvino` 1.24.1 | [PyPI](https://pypi.org/project/onnxruntime-openvino/) |
| The ONNX Runtime shadowing behaviour and the Ultralytics AutoUpdate mechanism (§3) | reproduced and fixed on AMD hardware in [ryzen_yolo](https://github.com/ZephyrSai/ryzen_yolo) |

When you first run this on real hardware, §6 prints what your machine actually
has; if it disagrees with this table, believe your machine and send a PR.
