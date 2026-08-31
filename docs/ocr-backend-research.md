# PaddleOCR / RapidOCR 方案调研（macOS Apple Silicon）

更新时间：2026-08-19

## 结论先行

- 你现在的 `VisionOCR` 方案是苹果系统 OCR，方向识别能力强但不是行业通用 OCR 栈。
- 如果要在 MacBook 上换成「通用 OCR」并尽量降低集成和部署摩擦：
  - `RapidOCR + onnxruntime` 为首选（模型可在运行时自动下载，支持 PP-OCRv4/PP-OCRv5/PP-OCRv6；能走 ONNX Runtime，Mac 上可用 CoreML 后端）。
  - `PaddleOCR` 也可用，但在当前 Apple Silicon 下主要依赖 PaddlePaddle + 预装模型，当前官方文档里对 Mac 的支持偏重 CPU，并存在历史文档与当前路径的兼容性歧义，落地风险更高。

## 1) 官方文档核验结果

### PaddleOCR / PaddlePaddle

- `paddleocr` 官方安装：基本路径是 `python -m pip install paddleocr`，如需全量能力可用 `python -m pip install "paddleocr[all]"`。
- 官方文档明确提到“推理时使用本地 `paddle_static` 需先安装 PaddlePaddle”；并且提供命令示例 `paddleocr ocr ...`。
- PaddlePaddle 官方 Mac 安装说明包含：
  - Mac 只支持 CPU 版本（未提供 macOS 下 GPU 版；GPU 编译页写明 macOS 下不支持 GPU 编译）。
  - Apple Silicon 建议本地编译并加 `-DWITH_ARM=ON`。
  - 有历史文档要求 Python 3.5~3.7 / x86_64 的要求项，与新版本文档（Python 3.9+）存在冲突，说明官方环境说明文档有分支且版本漂移，需要按当前你机器与当下 wheel 版本再做一次验装。
- PP-OCR 方向能力：命令行可显式控制 `--use_doc_orientation_classify`、`--use_doc_unwarping`、`--use_textline_orientation`。
- 模型下载：文档给出 `wget` 示例可下载 `det/rec/cls` 的推理模型到本地目录（可配合 `--*model_dir` 直接本地推理）。

### RapidOCR

- 官方文档提供了更直接的安装入口：`pip install rapidocr onnxruntime`。
- RapidOCR `rapidocr` 包默认不再捆绑 onnxruntime 依赖（从 2.0.6 起），但默认推理后端为 onnxruntime；如果已安装 ONNX Runtime 即可直接使用。
- 官方模型列表已覆盖 `PP-OCRv4` / `PP-OCRv5` / `PP-OCRv6` 的统一转换模型（PaddlePaddle/ONNX/MNN/TensorRT/PyTorch），模型托管在 ModelScope，且文档明确指出对应 `rapidocr` 版本会自动按参数下载。
- `rapidocr` 验证输出示例显示在安装后会下载并使用 `PP-OCRv6_det_small.onnx`、`PP-OCRv6_rec_small.onnx`（本机环境可复现）。
- `rapidocr_onnxruntime` 在 PyPI 显式标注兼容 `Python <3.13, >=3.6`，支持到 3.12。
- ONNX Runtime 官方（via RapidOCR 源码注释）指出 macOS 的标准 onnxruntime 包包含 CoreML，能让你避免额外手工做 NPU/ANE 适配。

## 2) 与你当前 PoC 的匹配性

当前 PoC 已把“读数流程”定义成：

1. 表盘检测（Level1）
2. 指针定位/角度（Level2）
3. 刻度映射读数（Level3）

其中第 3 步高度依赖刻度 OCR 的稳定性，建议更换为 RapidOCR 后再跑一次 20 次稳态压测。

## 3) 风险与建议

**建议直接试哪一个：**

1. 先上 `RapidOCR(onnxruntime)`：
   - 优点：安装链路短、模型可复用、和 ONNX 后端天然兼容，适合你 500ms PoC 的“快速可验”目标。
   - 可操作性：`pip install rapidocr onnxruntime` + 直接在数字识别前预处理裁剪图。
2. 仅在你必须与 PaddleOCR 特性（比如 PP-OCRv6 的一些参数）对齐时，再并行评估 `paddleocr`。

**风险点：**

- `PaddleOCR` 在 macOS 的底层依赖仍比 RapidOCR更重，且 Apple Silicon 下历史文档冲突较多（尤其 版本/架构/Python 兼容性），会把你首轮 PoC 的可跑性拉慢。
- 只换 OCR 引擎不等于自动读对所有模糊/斜拍图；后处理中仍需保证刻度圈展开、数字聚类和容错规则。

## 4) 推荐接下来马上做的改动（不影响你现有 detector 部分）

- 在当前 `vision_ocr.py` 外层保留一个新实现：`ocr_engine`。
- 尝试优先 `RapidOCR`，并保证：
  - 同一张图输出：是否读到刻度文本、文本置信度、bbox；
  - 与 Level 3 映射逻辑解耦（只负责提供文本候选和位置）；
  - benchmark 中独立计时（模型 warmup 5 次后再跑 20 次）。
- 通过后再决定是否再做 PaddleOCR 并行对照，不要先在 PaddleOCR 路线上耗费全部时间。
