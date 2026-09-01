# Analog Gauge Reading PoC (Apple Silicon)

不训练、不微调，以整张图片为输入，自动检测表盘、定位指针、理解线性刻度并输出实际读数。

## 本机环境

- MacBook Pro / Apple M2 Max / 12 核 / 32 GB / arm64
- macOS 26.5.2
- Python 3.12.12
- PyTorch 2.8.0，`MPS built=True`、`MPS available=True`
- OpenCV 4.12、ONNX Runtime 1.29、RapidOCR 3.9.2

## 当前方案

程序保留两个不训练的预训练方案作对照：

1. `Synanthropic/reading-analog-gauge` 两阶段 YOLO Pose。
2. ETHZ 预训练表盘检测/指针分割 + PP-OCRv6 small（RapidOCR/ONNX Runtime）+ 通用几何算法。

第二套完整链路：

```text
完整图片
→ ETHZ 表盘检测
→ 候选框几何验证
→ 无有效候选时启用与画面比例无关的多尺度重叠切割
→ 拒绝被人工切片边缘截断的候选并映射回原图
→ 检测失败时回退为通用圆形表盘检测
→ ETHZ 指针分割
→ 从指针最粗处估算轴心，使用可见圆弧拟合被遮挡的表盘椭圆
→ 椭圆仿射校正为圆形表盘
→ 全表盘 OCR 识别可见型号和名称，唯一匹配仪表类型知识包
→ 数字分支：刻度环 OCR → 鲁棒线性刻度拟合
→ 颜色分支：红绿分段 → 红绿交界零点 → 相对段值
→ 指针角度映射为原始视觉读数
→ 按 metadata 解释或归整为业务读数，并同时保留两者供审计
```

- Level 1：自动检测表盘，不接受固定 ROI。
- Level 2：自动估计表盘中心、指针方向和角度。
- Level 3：优先从 OCR 数字拟合量程；无数字但有红绿大段时输出可配置的相对段值。
- `Confidence` 是检测、分割、OCR 和拟合质量的保守组合，不是经过校准的业务正确率。

生产代码中没有写死测试图片的 ROI、中心、角度、刻度位置、量程或读数。

## 仪表类型专业知识

每个仪表类型使用一个独立知识包，不保存某张图片的具体读数：

```text
metadata/instrument-types/<type_id>/
├── metadata.json
└── 厂家说明书.pdf（能够获取并核验时的本地副本）
```

`metadata.json` 记录：

- 可见名称和型号标记；
- 一个或多个独立读数通道；
- 每个通道的显示方式、物理量、单位、量程和最小分度；
- 离散表盘的允许值及可审计的读数归整策略；
- 防止混淆不同显示区域的解释规则；
- 厂家资料、表盘文字或人工确认等证据来源。

例如，避雷器监测器有两个独立通道：下方指针是持续泄漏电流，上方机械计数窗是累计动作次数。`005` 应解释为动作 5 次，不能作为 `5 mA` 进入电流读数。

SHM-D 电动机构有三个独立通道：外圈长指针是当前分接位置，内圈是机构状态，机械计数窗是累计操作次数。位置标签是离散值，`9a`、`9b`、`9c` 不能合并成 `9`。厂家说明书和对应 `metadata.json` 位于同一类型目录，metadata 同时保存官方 URL、文档编号和本地文件 SHA-256。

JS-9 指针式放电计数器是离散的单圈数字盘，不按连续模拟刻度输出小数。表盘显示 `0` 时只记录当前可见数字 `0`；没有复位、进位或检修记录时，不据此推断设备全寿命累计动作总数。程序会保留几何算法得到的原始视觉读数，例如 `-0.05`，再按该类型 metadata 的允许值和最大归整距离输出业务读数 `0`。

查询已经收录的类型：

```bash
python -m src.instrument_metadata "JCQ-10/600Z 避雷器监测器"
python -m src.instrument_metadata "D96-V 同期电压表"
python -m src.instrument_metadata "SHM-D Motor drive unit"
python -m src.instrument_metadata "上海电瓷厂 JS-9 放电计数器"
```

加载器会自动合并各类型目录，验证 schema 版本、类型和通道 ID 唯一性、量程、允许值、归整规则以及证据来源；本地说明书存在时还会核对 SHA-256。识别链路通过 `InstrumentMetadataCatalog.find()` 匹配类型，再由 `InstrumentReadingInterpreter` 解释读数，不在 OCR 或通用几何代码中复制 JS-9 等具体仪表的专业规则。OCR 不能唯一匹配类型时，程序保留原始视觉结果，不擅自套用类型规则。

整图类型识别会合并表盘 OCR 和画面其他区域的型号、设备名称。对于同一画面中的多个仪表，`MetadataAwareImageAnalyzer` 对表盘候选去重并逐个读取；机械计数窗作为独立通道识别，原始 OCR、易混字符归整值和最终整数分别保留。双量程表在无法确认接线量程时输出候选值，不强行选择其中一个。

厂家说明书属于第三方资料。未确认再分发许可前，PDF 本地副本默认不进入 Git；可跟踪的 metadata 保留官方地址、文件名和哈希，既能在本机与专业知识同目录使用，也不会把第三方文档直接发布到公开仓库。厂家未公开说明书时，只记录已核验的官方产品页和现行标准，不把第三方销售页冒充厂家说明书。

用户明确提供的人工确认值与类型知识分开保存在本地 `observations/`。该目录默认不进入 Git；只有空目录占位文件会被跟踪，避免将用户现场读数发布到公开仓库。新图片自动报告不要求 observation。

对一批新图片生成多实例、多通道自动识别报告：

```bash
python batch_instrument_report.py image-1.jpg image-2.jpg image-3.jpg \
  --device cpu \
  --output output/user-instrument-batch.json
```

输出 `output/user-instrument-batch.json` 和同名 HTML。HTML 按图片展示内嵌原图、程序识别结果、状态、方法、原始显示和说明，不依赖原始临时图片路径，也不展示置信度、人工答案或比对结论。JSON 保留包括原始置信度在内的完整自动数据；本地存在同 SHA-256 observation 时，还可保留人工确认和比较字段用于审计。

新增现场图片、维护 metadata、核对自动结果、保持固定 HTML 格式和查找输出文件的完整步骤，统一以 [`docs/user-instrument-batch-runbook.md`](docs/user-instrument-batch-runbook.md) 为准。

## 五张图片的正式结果

每个设备均 warm-up 5 次、正式执行 20 次；模型加载时间排除在 steady-state latency 外。

| Input | Device | Detection | Pointer | Reading | p50 | p95 | Latency |
| --- | --- | --- | --- | ---: | ---: | ---: | --- |
| `gauge.png` | CPU | PASS | PASS | **2.62** | **224.64 ms** | **226.68 ms** | PASS |
| `gauge.png` | MPS | PASS | PASS | **2.62** | 231.97 ms | 236.17 ms | PASS |
| `gauge-2.png` | CPU | PASS | PASS | **2.40** | **206.71 ms** | **209.85 ms** | PASS |
| `gauge-2.png` | MPS | PASS | PASS | **2.40** | 211.87 ms | 213.91 ms | PASS |
| `gauge-3.jpg` | CPU | PASS | PASS | **3.12** | **419.87 ms** | **428.32 ms** | PASS |
| `gauge-3.jpg` | MPS | PASS | PASS | **3.12** | 480.25 ms | 504.83 ms | FAIL |
| `gauge-4.jpg` | CPU | PASS | PASS | **0.94 relative** | **281.33 ms** | **283.50 ms** | PASS |
| `gauge-4.jpg` | MPS | PASS | PASS | **0.94 relative** | 302.89 ms | 311.44 ms | PASS |
| `gauge-5.jpg` | CPU | PASS | PASS | **4.92** | **447.50 ms** | **455.75 ms** | PASS |
| `gauge-5.jpg` | MPS | PASS | PASS | **4.92** | 454.95 ms | 466.33 ms | PASS |

图片哈希：

- `gauge.png`: `5c5d2403e18528f199e1d5a4335b1aa55a58e1c22101ecbd22f8da24d131dd77`
- `gauge-2.png`: `f9d8d22dc81d4271258a00a7fc9f694a7409eae279310e99a9053e44e3c82544`
- `gauge-3.jpg`: `868e1afe635c6e6b248b8bc0cf15a2d26eff352eea901327fd1b87c791077460`
- `gauge-4.jpg`: `a21d76424dc72dde5dd6179229ba1795056381db19dc63a3aca4c69aa49bbca5`
- `gauge-5.jpg`: `6d32374ba31ea5460f14477d5ac1d7cc8b8d29aa79d4100e4ef9eeb2752b16a3`

第一张使用 OCR 刻度 `0–10`，刻度拟合 RMSE `0.097`；第二张使用 `0、1、3、4、7、8、9、10`，缺失数字由同一线性刻度拟合补足，伪识别 `-5、910` 被拒绝，RMSE `0.076`。

第三张的刻度半径明显更大，默认扇区数字不足后自动回退全表盘 OCR；使用 `0、2、3、4、7、8、9、10`，拒绝连着黑色刻度块误识别出的 `15、16`，RMSE `0.044`。宽三角指针使用分割掩膜尖端方向，结果 `3.117`（终端显示 `3.12`），画面标注为 `3.1`。

第四张没有数字量程。程序自动检测一段红色和三段绿色，以红绿交界为 `0`、红色为负方向、绿色为正方向；默认每个大段为 `1.0`，得到相对读数 `0.939`（终端显示 `0.94`），画面标注为 `0.92`。该结果的单位明确标为 `relative_segment`，不冒充物理量。

第五张表盘上半部被设备外壳遮挡，固定 `640` 整图检测会误报整张图片。程序自动启动三个重叠切片，拒绝贴着切片边缘的不完整候选，并利用指针轴心和可见圆弧恢复表盘几何；最终使用 OCR 刻度 `0、1、2、3、4、8、9、10`，读数为 `4.916`（终端显示 `4.92`），画面标注为 `5`。

第二张画面标注为 `2.4`，程序结果为 `2.403`（终端显示两位小数 `2.40`）；第一张画面标注为 `2.6`，程序结果为 `2.621`（终端显示 `2.62`）。画面标注没有进入算法输入，仅用于事后人工核对。

可视化：

- `output/result.jpg`、`output/result-2.jpg`：检测框、中心、指针、角度和读数。
- `output/rectified.jpg`、`output/rectified-2.jpg`：自动校正后的表盘。
- `output/scale-ring.jpg`、`output/scale-ring-2.jpg`：自动展开的刻度环。
- 第三张对应文件使用 `result-3.jpg`、`rectified-3.jpg`、`scale-ring-3.jpg`。
- 第四张对应文件使用 `result-4.jpg`、`rectified-4.jpg`、`scale-ring-4.jpg`。
- 第五张对应文件使用 `result-5.jpg`、`rectified-5.jpg`、`scale-ring-5.jpg`；MPS 版本带 `-mps` 后缀。

## 安装与运行

```bash
cd analog-gauge-poc
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
python benchmark.py input/gauge.jpg
```

默认行为：

- MPS 可用时依次测试 CPU 和 MPS。
- warm-up 至少 5 次，正式执行至少 20 次。
- 输出 mean、p50、p95、min、max。
- `p95 < 500 ms` 判定 latency PoC PASS。
- 模型加载、结果图写盘不计入 steady-state latency。

只跑单个设备：

```bash
python benchmark.py input/gauge.jpg --device cpu
python benchmark.py input/gauge.jpg --device mps
```

生成 `input/gauge*` 全量回归 HTML 报告：

```bash
python regression_report.py --device all --warmup 5 --runs 20
```

输出：

- `output/regression-report.html`：自包含 HTML，含原图缩略图、识别可视化、Level 1/2/3 和各项延迟；
- `output/regression-report.json`：同次执行的可审计原始数据；匹配到类型时同时记录原始视觉读数、业务读数、类型 ID、通道 ID 和解释方法；
- `output/regression-report-assets/`：CPU/MPS 标注图片。

模型按设备各加载一次，加载时间单列且不计入 steady-state；每张图片仍独立 warm-up 至少 5 次、正式执行至少 20 次。

彩色分段默认每个大段为 `1.0`。现场标定为每段 `2.0` 时：

```bash
python benchmark.py input/gauge-4.jpg --device cpu --units-per-major-segment 2
```

同一几何位置会从相对读数约 `0.94` 转换为约 `1.88`。

## 延迟口径

```text
T_steady = T_preprocess + T_inference + T_postprocess
```

- preprocess：每次从完整图片路径解码。
- inference：表盘检测、指针分割、椭圆校正，以及需要时的 PP-OCRv6 批量识别；MPS 前后显式同步。
- postprocess：OCR 刻度拟合，或红绿分段、零点和相对段值计算。
- CPU 当前比 MPS 略快，因为 ONNX OCR 使用 CPU，MPS 仅承载两个 YOLO 模型。

## 验证

```bash
python -m unittest discover -s tests -v
uvx ruff check src benchmark.py regression_report.py batch_instrument_report.py tests
uvx ty check src benchmark.py regression_report.py batch_instrument_report.py
```

## 模型来源

- [Synanthropic/reading-analog-gauge](https://huggingface.co/spaces/Synanthropic/reading-analog-gauge/tree/main)
- [ethz-asl/analog_gauge_reader](https://github.com/ethz-asl/analog_gauge_reader)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [RapidOCR](https://github.com/RapidAI/RapidOCR)

OCR 完全在本机执行，不需要云服务或 API key。Apple Vision 旧实验代码仍保留作历史对照，但 benchmark 已不再使用。
