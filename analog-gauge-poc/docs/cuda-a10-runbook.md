# Ubuntu A10 CUDA 运行手册

本文件是 `codex/cuda-a10-runtime` 分支的 CUDA 安装、验证和回归入口。该分支只保存设备选择、CUDA 依赖、GPU 同步计时、运行环境证据和 GPU 验收；仪表类型、指针几何、OCR 解释和模型权重继续由 `main` 演进。

## 当前加速范围

- `--device cuda` 将 Ultralytics/PyTorch 的 YOLO 检测和分割放到 NVIDIA GPU。
- RapidOCR/ONNX Runtime 和 OpenCV 几何仍使用 CPU；这是故意的第一阶段边界。
- 报告的 `runtime.accelerator` 记录 PyTorch CUDA build、GPU 型号、显存占用和 ONNX Runtime providers。
- CUDA 不可用时，命令在加载模型前明确失败，不会静默回退 CPU。

## 安装

以 Ubuntu x86_64、Python 3.12、NVIDIA A10 和驱动支持 CUDA 12.8 为验证目标。使用独立虚拟环境，不覆盖 CPU 环境：

```bash
cd /mnt/workspace/industrial-gauge-reader/analog-gauge-poc
git switch codex/cuda-a10-runtime
git pull --ff-only

uv venv .venv-cuda --python 3.12
source .venv-cuda/bin/activate
uv pip install -r requirements-cuda.txt
```

`requirements-cuda.txt` 保持项目原有版本，只将 PyTorch/Torchvision 固定为官方 CUDA 12.8 wheel。第一阶段继续安装 CPU 版 `onnxruntime==1.29.0`；不安装 `onnxruntime-gpu`。

## 设备预检

```bash
nvidia-smi

.venv-cuda/bin/python - <<'PY'
import json
import torch
from src.inference_device import accelerator_fingerprint, ensure_device_available

ensure_device_available("cuda")
print(json.dumps(accelerator_fingerprint("cuda"), indent=2))
PY
```

同时满足以下条件才可继续：

- `torch.cuda.is_available()` 为 `true`；
- `torch_cuda_build` 为 `12.8`；
- `cuda_devices` 包含 `NVIDIA A10`；
- `onnxruntime_available_providers` 可以只有 `CPUExecutionProvider`，与当前 OCR 执行策略一致。

## 真实图片冒烟测试

每次运行使用独立时间戳目录：

```bash
run_stamp="$(date +%Y%m%dT%H%M%S-%N)"
run_root="output/cuda-a10/$run_stamp"
mkdir -p "$run_root"

.venv-cuda/bin/python batch_instrument_report.py \
  --device cuda \
  --pipeline-profile 448-highres-pad \
  --export-processing-stages \
  --output "$run_root/iter2/instrument-report.json" \
  input/iter2
```

命令必须返回 `0`，并在终端打印 JSON、HTML 和中间阶段图的绝对路径。然后检查 GPU 证据：

```bash
.venv-cuda/bin/python - "$run_root/iter2/instrument-report.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(json.dumps(payload["runtime"]["accelerator"], indent=2))
print(json.dumps(payload["automated_summary"], indent=2))
PY
```

`selected_device` 必须是 `cuda`，`cuda_memory_allocated_bytes` 必须大于零。另一个终端的 `nvidia-smi` 应能观察到 Python 进程使用 GPU。

## CPU/CUDA 非回归

对 `iter1`、`iter2`、`iter2-原图`和 `iter3` 分别使用 CPU 与 CUDA 生成报告。两边必须使用相同的：

- Git commit 和干净工作区；
- 原图 SHA-256；
- pipeline profile；
- 四个模型 SHA-256；
- 除 PyTorch CUDA build 外的识别依赖版本。

逐张比较类型、通道、状态、候选、指针方向、表盘框和关键中间图。GPU 数值只有浮点末位差异且不跨越仪表分度时，记录为数值漂移；任何状态、候选、几何分支或业务读数变化均按回归处理。

完成真实图片回归后再跑延迟基准：

```bash
sample_image="$(find input/iter2 -maxdepth 1 -type f \
  \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | sort | head -1)"

.venv-cuda/bin/python benchmark.py \
  "$sample_image" \
  --device cuda \
  --warmup 5 \
  --runs 20 \
  --output "$run_root/cuda-benchmark.jpg"
```

模型首次加载与预热后延迟分开记录。CUDA 计时必须包含显式同步，不得用异步 kernel 提交时间代替真实完成时间。

## 失败边界

- CUDA 不可用：修复 PyTorch wheel 或宿主机 NVIDIA 环境，不修改识别算法。
- CUDA 可用但读数不同：保留 CPU/CUDA JSON 和中间图，按首个分叉阶段诊断。
- CUDA 读数一致但速度不理想：先用阶段耗时确认是 YOLO、OCR 还是 OpenCV；只有 OCR 成为主要瓶颈时，才评估 `onnxruntime-gpu`。
- 未在真实 A10 上完成上述测试前，结论只能写“CUDA 代码路径已实现，尚未完成实机 GPU 验收”。
