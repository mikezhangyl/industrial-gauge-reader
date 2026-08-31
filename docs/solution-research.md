# Analog gauge 现成方案调研

调研日期：2026-08-18。本文只评估一手资料与权重可获得性，尚未在当前图片上完成真实推理或 latency benchmark。

## 结论

- **Mac 首试建议：`Synanthropic/reading-analog-gauge` 的两阶段 YOLO Pose 权重。** 它的整图表盘定位、透视校正和 `start / center / end / tip` 指针关键点都已有权重，依赖是较新的 Ultralytics + OpenCV，最容易在 Apple Silicon 上先验证 Level 1/2。
- **但它不是完全本地 Level 3：** 最终读数依赖 Google Cloud Vision OCR。若要求所有计算离线，本地链路仍需替换 OCR；若只给定量程做角度映射，必须把量程配置明确标成外部先验。
- **功能最完整的是 `ethz-asl/analog_gauge_reader`：** 它会从整图检测表盘、分割指针、OCR 刻度数字并拟合“角度 → 数值”，不要求预填量程；但是官方锁定的 Linux/x86 OpenMMLab 环境对 Apple Silicon 不友好，而且源码会在每张图处理时重复构造模型，不能原样用于 steady-state latency benchmark。
- **`shuyansy/Detect-and-read-meters` 权重仍可下载，但不适合作为 Mac 第一选择：** 代码显式围绕 CUDA、旧 YOLOv5/OpenCV 3 和 FP16 编写，CPU/MPS 需要先修兼容性。

## 候选对比

| 项目 | 预训练权重 | Level 1：整图找表 | Level 2：指针/角度 | Level 3：实际数值 | Apple Silicon 现实性 | 首轮建议 |
| --- | --- | --- | --- | --- | --- | --- |
| [`Synanthropic/reading-analog-gauge`](https://huggingface.co/spaces/Synanthropic/reading-analog-gauge/tree/main) | 有；37.7 MB 表盘/透视 Pose + 6.41 MB 指针 Pose | 有；整图预测表盘四角并做透视校正 | 有；输出 `start_kp / center / end_kp / tip` | 有条件；需 Google Cloud Vision OCR，至少识别到 2 个刻度数字 | 三者中最好；Ultralytics 8.1.2 + OpenCV 4.9，无 CUDA 硬编码；MPS 仍需实测 | **优先试 Level 1/2；Level 3 必须单列外部 OCR 延迟** |
| [`ethz-asl/analog_gauge_reader`](https://github.com/ethz-asl/analog_gauge_reader) | 有；检测、关键点、分割三份 Git LFS 权重 | 有；YOLOv8 从原图选最高置信表盘并裁剪 | 有；YOLOv8 分割指针，再以 ODR 拟合指针直线 | 有；MMOCR 读刻度数字/单位，再拟合角度到数值，不需预填量程 | 官方环境是 Python 3.8/PyTorch 2.0 + Linux x86 `mmcv` wheel；Mac 需改依赖和 device 路径 | 功能验证备选，不作为最快环境方案 |
| [`shuyansy/Detect-and-read-meters`](https://github.com/shuyansy/Detect-and-read-meters/tree/v2) | 有；检测约 173 MB + 识别约 330 MB，2026-08-18 实测下载端点返回 200 | 有；YOLOv5 从整图输出 `meter` crop | 有；联合网络预测 pointer/dial/text mask | 有条件；OCR 一个关键数字并结合两参考点算比例，无显式量程配置，但假设较强 | Python 3.7、OpenCV 3、CUDA 9/10；源码默认 CUDA 且 CPU 也强制 FP16 | 快速放弃条件应较低 |

## 一手证据与关键边界

### 1. `shuyansy/Detect-and-read-meters`

- README 明确给出三段流程：YOLO 检测表盘、对齐、pointer/dial/OCR 联合识别，并提供检测与识别权重：[项目 README](https://github.com/shuyansy/Detect-and-read-meters/blob/v2/README.md)。
- 权重链接目前可访问：[`best.pt` 检测权重](https://drive.google.com/file/d/1bHYpJro3ERmNTRO2JEo1inyU0_juqw5z/view) 和 [`textgraph_vgg_100.pth` 识别权重](https://drive.google.com/file/d/1sHmEEf9E0_kvL0LW1S5Y5jjFgjx_O5Dj/view)。本次通过文件下载端点核到文件名、HTTP 200 与约 173/330 MB 大小；没有 GitHub Release 兜底。
- [`get_meter_area.py`](https://github.com/shuyansy/Detect-and-read-meters/blob/v2/get_meter_area.py) 对整图执行 YOLOv5，取 `meter` 类 bbox 后裁剪，因此不需要固定 ROI。
- [`predict_online.py`](https://github.com/shuyansy/Detect-and-read-meters/blob/v2/predict_online.py) 把 crop 输入联合网络，得到 `pointer / dail / text / reco / std`；[`util/read_meter.py`](https://github.com/shuyansy/Detect-and-read-meters/blob/v2/util/read_meter.py) 用指针线、两个参考点和识别出的数字换算数值。它不是“理解任意刻度体系”的通用 OCR 标定：只有在关键数字、参考点和旋转方向都正确时 Level 3 才成立。
- Mac 阻碍来自源码而不只是 README 老旧：检测器在 `torch.cuda.is_available() == false` 时选择 CPU，但仍无条件 `model.half()`、输入也转 FP16；[`util/config.py`](https://github.com/shuyansy/Detect-and-read-meters/blob/v2/util/config.py) 只在 `cuda/cpu` 二选一，默认 `cuda=True`，没有 MPS 分支。需要兼容性补丁后才值得跑 benchmark。

### 2. `ethz-asl/analog_gauge_reader`

- 三份仓库权重是有效 Git LFS 对象，当前可直接下载：[`gauge_detection_model.pt`](https://media.githubusercontent.com/media/ethz-asl/analog_gauge_reader/main/models/gauge_detection_model.pt) 6,239,224 B、[`key_point_model.pt`](https://media.githubusercontent.com/media/ethz-asl/analog_gauge_reader/main/models/key_point_model.pt) 88,373,505 B、[`segmentation_model.pt`](https://media.githubusercontent.com/media/ethz-asl/analog_gauge_reader/main/models/segmentation_model.pt) 6,768,834 B。默认模型路径见 [`pipeline.py`](https://github.com/ethz-asl/analog_gauge_reader/blob/main/pipeline.py)。
- Level 1 是真实整图检测：[`gauge_detection/detection_inference.py`](https://github.com/ethz-asl/analog_gauge_reader/blob/main/gauge_detection/detection_inference.py) 调用 YOLOv8，并取最高置信 bbox。之后 pipeline 才裁剪、缩放。
- Level 2 同时使用关键点与分割：关键点建立刻度椭圆/零点，分割模型给出针像素，再由 [`segmentation/segmenation_inference.py`](https://github.com/ethz-asl/analog_gauge_reader/blob/main/segmentation/segmenation_inference.py) 做 ODR 直线拟合。
- Level 3 不需要预先写死量程：[`pipeline.py`](https://github.com/ethz-asl/analog_gauge_reader/blob/main/pipeline.py) 用 MMOCR 的 DB-r18 + ABINet 识别表盘数字，投影到刻度椭圆，再拟合角度与数值的线性关系。实际成功仍取决于能读到足够可靠的刻度数字；没有数字会明确失败。
- Apple Silicon 安装风险明确：[`pyproject.toml`](https://github.com/ethz-asl/analog_gauge_reader/blob/main/pyproject.toml) 把 `mmcv` 固定为 `cp38 manylinux1_x86_64` wheel，README 又要求 PyTorch 2.0.0、MMCV 2.0.0、MMOCR 1.0.0、Ultralytics 8.0.66；Poetry 配置不能原样安装到 macOS arm64。
- latency 统计前必须改加载边界：检测、关键点、分割与 OCR 类均在单图调用路径内构造。原样循环 20 次会把重复模型加载混进 latency，不能代表用户定义的 steady-state。

### 3. `Synanthropic/reading-analog-gauge`（更适合 Mac 的替代）

- Hugging Face Space 直接托管两份权重：[`corners-best.pt`](https://huggingface.co/spaces/Synanthropic/reading-analog-gauge/resolve/main/corners-best.pt?download=true) 37,732,202 B 与 [`keypoints-best.pt`](https://huggingface.co/spaces/Synanthropic/reading-analog-gauge/resolve/main/keypoints-best.pt?download=true) 6,409,410 B；文件页可见它们是 Ultralytics PoseModel checkpoint：[文件树](https://huggingface.co/spaces/Synanthropic/reading-analog-gauge/tree/main)。
- [`app.py`](https://huggingface.co/spaces/Synanthropic/reading-analog-gauge/blob/main/app.py) 的第一模型从整图找表盘四角并透视拉正，第二模型给出 `start_kp / center / end_kp / tip`，所以 Level 1/2 不依赖固定 ROI、中心或角度。
- Level 3 通过 Google Cloud Vision `text_detection` 读取刻度数字，再根据数字在圆周的位置与针尖角度插值；代码要求至少两个 OCR 数字。量程不是硬编码，但这一步需要 API key、网络和外部服务，必须把 API latency 与本地模型 latency 分开报告。
- [`requirements.txt`](https://huggingface.co/spaces/Synanthropic/reading-analog-gauge/blob/main/requirements.txt) 是 Ultralytics 8.1.2、OpenCV 4.9、SciPy 和 Pillow 等通用包，没有 CUDA wheel 或 x86 wheel 硬编码。PyTorch 官方支持在 Apple Silicon 使用 [`mps` device](https://docs.pytorch.org/docs/stable/notes/mps.html)；该项目默认 `model.predict(...)` 没有显式选择 MPS，因此 CPU 与 `device="mps"` 都要分别实测，不能先假定加速有效。
- 风险：Space 当前页面显示 `Runtime error`；作者也写明模型仅用合成数据训练，表盘应占图片大部分，显著不同的真实图片可能失败。这是必须通过当前图片实测的准确性风险，不应由 demo 图表现替代。

## 快速排除的方案

- [`DrawZeroPoint/VectorDetectionNetwork`](https://github.com/DrawZeroPoint/VectorDetectionNetwork)：作者提供了训练后权重，但 README 明确要求输入是上游 meter detector 输出的表盘 patch，而不是复杂整图；官方环境是 Ubuntu + NVIDIA Docker，并要求编译第三方模块。它只能补 Level 2，不能单独满足 Level 1/3，也不比上面三项更适合 Apple Silicon。
- [`SiddharthB7/vlm-analog-gauge-reader`](https://github.com/SiddharthB7/vlm-analog-gauge-reader)：代码栈较新且设计完整，但 README 明确写着模型权重和数据未提交、需另行分享；当前仓库没有可直接下载的训练后 gauge detection/pose 权重，因此不满足“第一轮直接 inference”。其 README 自报 OCR pipeline CPU latency 约 2.48 s/image，也不支持先验判断 `<500 ms`。

## 建议的自动尝试顺序

1. `Synanthropic` 两份 YOLO Pose 权重：先验证整图表盘和四关键点；CPU/MPS 各跑 warm-up 5 + 正式 20。外部 Google OCR latency 单独列项，不混入“本地模型 inference”指标，但完整 Level 3 的 total steady-state 必须包含它。
2. 若它因真实图域差异失败，尝试 `ethz-asl` 三份权重；先在当前 Python/PyTorch 上做最小兼容安装，不为复刻旧 Poetry 环境投入过多时间，并把模型移到循环外。
3. 仅在前两项不能形成有效读数、且愿意承担旧代码修补时再试 `shuyansy`。若卡在旧 OpenCV、CUDA/FP16 或 checkpoint 兼容，快速终止，不进入训练。

最终选择必须由当前原图的 Level 1/2/3 结果和 20 次 p50/p95 决定；上面的 README 能力描述不能替代本机 benchmark。
