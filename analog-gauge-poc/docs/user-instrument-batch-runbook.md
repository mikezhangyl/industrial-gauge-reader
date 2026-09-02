# 仪表自动识别批次运行手册

本文件是现场仪表新图片处理和批次报告格式的唯一权威规范。新图片默认没有人工答案；HTML 只展示本次程序识别结果。除非用户明确要求改变格式，后续 AI Agent 持续生成相同结构的自包含 HTML 和同名 JSON。

## 完成条件

一批新图片只有同时满足以下条件才算完成：

1. 每张输入图片在 JSON 中有一个 `record`，在 HTML 中有一个对应图片卡片，顺序与命令行输入一致。
2. HTML 可以脱离原始临时图片打开，并只展示原图及本次程序产生的结果、状态、方法、原始显示和说明。
3. JSON 包含 `automated_summary`，并分别统计匹配到的类型 metadata、程序通道、成功、候选、未识别和图片级失败。
4. 未唯一匹配类型 metadata 时继续执行通用指针读取，并将类型和未知单位明确标记；只有表盘检测等视觉分析失败时才记录图片级失败，不补入人工猜测值。
5. `output/` 中同时存在同名 `.json` 和 `.html`，HTML 中内嵌图片数与输入图片数一致。
6. 每条 JSON 记录包含原图 SHA-256、原图/分析图尺寸、检测框和运行环境指纹，可用于两台机器对比。
7. 相关测试、Ruff、类型检查和 `git diff --check` 通过。交付结论写成“当前验证未发现回归”。

## 数据职责

### 类型知识：`metadata/instrument-types/`

一个仪表类型对应一个目录。`metadata.json` 保存型号标记、识别特征、读数通道、单位、量程、最小分度、离散允许值、解释规则和证据来源，不保存某张照片的具体读数。

能够核验厂家说明书时，将本地副本与 `metadata.json` 放在同一类型目录，并在 metadata 中记录官方 URL、文档编号、本地文件名和 SHA-256。厂家 PDF 默认被 Git 忽略；公开仓库只保存允许公开的知识和证据索引。

专业知识示例：避雷器监测器上方 `005` 是机械动作计数窗，表示累计动作 5 次；下方指针才是持续泄漏电流。这类规则属于类型 metadata。

### 可选标注：`observations/`

新图自动报告不要求 observation。只有用户明确提供确认值并要求标注回归时，才在 `observations/` 保存按图片 SHA-256 关联的人工确认。

JSON 可以同时保留自动值、确认值和比较结果用于审计；HTML 始终只展示程序结果。历史 observation 中的 `automated_result` 只是旧记录，本次运行结果以新生成的 JSON 为准。

现场图片、observation 和生成报告默认不提交到公开仓库。

## 新图片处理步骤

### 1. 准备环境

```bash
cd /path/to/industrial-gauge-reader/analog-gauge-poc
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

已有 `.venv` 且依赖未变化时可直接激活使用。

### 2. 核对类型知识

先检查 `metadata/instrument-types/` 是否已有匹配类型。新类型以现有目录为结构示例新增 `metadata.json`；已有类型只补充经过核验的通道、解释规则或证据。

程序利用整图 OCR 尝试唯一匹配类型 metadata。metadata 是可选的结果增强：命中后补充仪表类型、业务通道、单位、量程和专业解释；没有唯一匹配时必须回退到通用指针读取，保留能够从表盘视觉直接得到的读数，并把类型或单位标为未知。不能仅因缺少 metadata 将图片判为失败。

### 3. 运行固定自动报告

推荐把同一批原始现场照片直接放在一个目录中。输入照片必须未画框、未画指针箭头、未写入旧读数；不得把 `output/result*.jpg` 或其他程序可视化派生图改名后重新放入输入目录。程序只读取该目录第一层的常见图片文件，忽略 `.DS_Store`、隐藏文件和非图片文件，并按文件名自然排序。相对路径和绝对路径都支持。

程序会利用整图 OCR 检查本项目生成的 `Pointer angle / Sweep position / Reading` 覆盖层，并兼容常见 OCR 拼写误差。一旦命中，程序仍生成带明确失败原因的 JSON/HTML，但不检测表盘、不输出读数，并返回退出码 `2`。执行者必须换回原始照片，不能忽略退出码或调整算法掩盖输入错误。

Iteration 2 的标准命令：

```bash
.venv/bin/python batch_instrument_report.py input/iter2
```

通用目录命令：

```bash
.venv/bin/python batch_instrument_report.py /absolute/or/relative/image-directory
```

程序自动执行以下输入标准化，不修改原始文件：

- 按 EXIF 修正旋转方向；
- 转换为 RGB；
- 最长边超过 1920 像素时等比例缩小；
- 用无损 PNG 作为本次分析输入；
- 在 JSON 中同时记录原图和标准化图片的 SHA-256 与尺寸。

也可以显式传入多个图片文件，此时保持命令行顺序：

```bash
.venv/bin/python batch_instrument_report.py \
  --device cpu \
  --output output/user-instrument-batch.json \
  "/absolute/path/to/image-1.jpg" \
  "/absolute/path/to/image-2.jpg"
```

目录输入的默认固定输出位于 `output/<输入目录名>/`。例如 `input/iter2` 生成：

- `output/iter2/instrument-report.json`：机器可读结果、输入哈希、检测框、环境和模型指纹。
- `output/iter2/instrument-report.html`：内嵌统一大小图片预览、检测框和程序识别结果。

命令结束时必须把输入目录、JSON 和 HTML 的绝对路径打印到终端。执行者以终端打印路径为准，不猜测输出位置。

显式图片列表继续默认输出 `output/user-instrument-batch.json` 和同名 HTML。

需要保留多个批次时，只改变 `--output` 的 JSON 文件名；HTML 自动使用相同文件名主干。例如 `--output output/2026-09-02-substation-a.json` 会同时生成 `output/2026-09-02-substation-a.html`。

`output/` 被 Git 忽略。需要长期归档时，将同名 JSON 和 HTML 一起复制到经用户确认的安全位置。

### 4. 核对程序结果

```bash
jq '.automated_summary' output/iter2/instrument-report.json
open output/iter2/instrument-report.html
```

依次检查：

- `instrument_types_recognized` 表示匹配到类型 metadata 的图片数；未命中 metadata 的通用读数仍应出现在程序通道中；
- `recognized`、`ambiguous`、`not_recognized` 和 `analysis_failures`；
- `records[].channels[].automated` 中的值、候选、状态、方法、原始 OCR 和说明；置信度只在 JSON 中保留供技术审计；
- `records[].detections` 是否包含程序检测到的仪表框；
- 对内外同心显示区域，`records[].detections[].bbox` 是 HTML 绿色展示框，必须覆盖实际读取的外圈标签；`detector_bbox` 保留模型原始框，`bbox_method` 说明展示框是否经过几何扩展；
- `runtime.git_commit`、依赖版本和四个模型 SHA-256 是否与对比机器一致；
- HTML 中每张原图和对应程序通道是否完整可见。

程序输出候选值时，HTML 保留全部候选。例如照片无法确认双量程实际接线时，可以显示 `1/2 V`，不擅自二选一。

### 5. 回归验证

```bash
.venv/bin/python -m unittest discover -s tests -v
uvx ruff check src benchmark.py regression_report.py batch_instrument_report.py tests
uvx ty check src benchmark.py regression_report.py batch_instrument_report.py
git diff --check
```

修改报告 HTML 时，还必须运行 `tests/test_batch_instrument_report.py`，并在浏览器中打开真实生成的 HTML，确认所有输入图片和程序结果均可见。

## 可选：标注回归

已有用户确认 observation、且任务明确要求比较自动结果与确认值时，可以额外使用 `--require-pointer-match`。该参数只影响退出码和 JSON 中的 `pointer_acceptance`，不会把人工答案显示到 HTML。

没有已确认指针通道时，使用 `--require-pointer-match` 会返回非零退出码，避免把无标注批次误报为“全部吻合”。

## 固定 HTML 报告契约

`batch_instrument_report.py::render_html()` 负责报告结构。格式固定为：

1. 顶部标题为“仪表自动识别报告”，说明结果来自本次程序运行。
2. 摘要区展示图片数、识别到的仪表类型数、程序通道数、识别成功数、候选结果数、未识别数和图片级失败数。
3. 每张输入图片按命令行顺序生成一个卡片，左侧或上方展示原图，右侧或下方展示程序数据表。
4. 每个程序通道的主行展示实例、通道 ID、程序识别结果和状态；下方详情行展示识别方法、原始显示和说明。
5. 图片先按展示框并集加边距裁剪，再放入统一的 960×640 预览画布；检测到的仪表使用绿色框标出。普通仪表的展示框等于模型检测框；当程序通过内圈定位并读取同心外圈标签时，展示框必须扩展到包含外圈标签，同时在 JSON 保留原始模型框。没有检测框时展示完整标准化图片。
6. 图片以 data URI 嵌入 HTML，使报告不依赖输入目录或 `/var/folders/...` 等临时文件路径。
7. `recognized`、`ambiguous` 和 `not_recognized` 使用清晰、可区分的状态样式；窄屏自动改为单列。
8. 图片级识别失败仍保留原图卡片和失败原因。

HTML 不展示置信度、observation、人工确认值、自动/人工比对、最终复核值或最终值来源。这些技术或可选审计字段只保留在 JSON。

格式变更需要同时更新 `render_html()`、针对性回归测试和本文件，并在真实图片生成的报告上完成浏览器检查。用户未明确要求变更时，维持本契约。

## 交付格式

向用户交付时直接提供：

- HTML 报告的绝对路径链接；
- 同批 JSON 的绝对路径链接；
- 程序识别到的类型数、通道数、成功数、候选数、未识别数和图片级失败数；
- 本次实际执行的测试和检查结果。

以本次 JSON 的 `automated_summary` 和 `records[].channels[].automated` 为事实来源。

## 跨机器复现检查

另一台机器出现不同结果时，先比较以下字段，再修改算法：

1. 两边输入文件的 `records[].image_sha256` 是否完全相同；
2. `runtime.git_commit` 是否相同，工作区是否无未提交行为改动；
3. `runtime.packages` 中 Python、OpenCV、ONNX Runtime、RapidOCR、PyTorch 和 Ultralytics 版本是否相同；
4. `runtime.models` 中四个模型的 SHA-256 是否相同；
5. 两边是否都使用目录标准命令，而不是临时脚本或人工整理表格。
6. 输入是否为原始照片，而不是已经包含绿色框、红色箭头或 `Pointer angle / Sweep position / Reading` 的程序派生图。

只有上述条件相同仍产生差异时，才归类为跨平台数值或算法稳定性问题。
