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
- 另生成最长边不超过 1920 像素的等比例定位图；
- 实验 profile 需要时，另保留完整分辨率、方向已校正的临时细节图；
- 定位图和可选细节图均使用无损 PNG，批次结束后随临时目录清理；
- 已定位表盘的临时裁剪同样使用无损 PNG，避免二次 JPEG 压缩改变细指针分割结果；
- 在 JSON 中同时记录原图和标准化图片的 SHA-256 与尺寸。

默认 `448-highres-pad` profile 用定位图检测和分割指针，并把候选与掩膜映射回完整分辨率细节图做几何、OCR 和刻度解释；非正方形裁剪先保持比例补边，再缩放到 448×448。旧 `448` 只作为比较基线。JSON 的 `analysis_dimensions` 是定位图尺寸，`detail_dimensions` 是实际细节来源尺寸。

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

需要审阅图像处理链路时，显式增加阶段导出参数；普通批次默认不生成大量中间 PNG：

```bash
.venv/bin/python batch_instrument_report.py \
  --pipeline-profile 448-highres-pad \
  --export-processing-stages \
  --output output/stage-review/instrument-report.json \
  input/iter2-原图
```

程序会在报告目录下创建唯一的 `processing-stages/<run-id>/<图片>/`，按读取分组保存无损 PNG，包括：EXIF 校正原图、低分辨率定位图、可选高分辨率细节图、表盘候选框、分割裁剪、补边/未补边画布、模型实际输入、模型原始掩膜、回映射掩膜、几何裁剪、椭圆拟合、校正表盘、校正掩膜和展开刻度环。某一步只有在当前实际执行路径产生对应数组时才会出现，不伪造未执行阶段。

`records[].processing_stages[]` 记录每张阶段图相对路径、真实像素尺寸、宽高比、变换操作、来源阶段以及是否保持宽高比。HTML 内嵌等比例 `contain` 缩略图，不裁剪、不拉伸；点击缩略图可打开原尺寸 PNG。命令结束时另打印 `Processing stages:` 绝对路径。阶段导出仅旁路复制实际数组，不得改变识别分支或读数。

### 4. 核对程序结果

涉及完整圆形外壳、圆环、椭圆拟合或姿态校正时，必须先完整阅读并执行 [`dial-geometry-validation.md`](dial-geometry-validation.md)。该规范负责候选质量、外圈选择、下游交叉验证和几何回归；最终读数相同不能代替几何验收。

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
- 对 JCQ 类带隐藏轴心和调零螺钉的矩形刻度窗，先用完整外圈椭圆校正斜拍姿态，再由真实指针延长线与“中间刻度—调零螺钉中心线”的交点确定隐藏轴心；最右端刻度不参与轴心求解，完整刻度扇面只复核量程映射。外圈、两线或刻度扇面任一项未通过合理性检查时必须拒绝专用路径，不能输出伪精确结果；
- `runtime.git_commit`、依赖版本和四个模型 SHA-256 是否与对比机器一致；
- HTML 中每张原图和对应程序通道是否完整可见。

程序输出候选值时，HTML 保留全部候选。例如照片无法确认双量程实际接线时，可以显示 `1/2 V`，不擅自二选一。

### 5. 回归验证

```bash
.venv/bin/python -m unittest discover -s tests -v
uvx ruff check src benchmark.py regression_report.py batch_instrument_report.py compare_pipeline_profiles.py tests
uvx ty check src benchmark.py regression_report.py batch_instrument_report.py compare_pipeline_profiles.py
git diff --check
```

默认批次固定使用已回归的 `448-highres-pad` profile。后续尺寸实验不能直接覆盖默认值；使用命名 profile，并将表盘工作画布与模型推理输入分别记录在 JSON 顶层的 `pipeline_profile`：

```bash
.venv/bin/python batch_instrument_report.py \
  --pipeline-profile 640 \
  --output output/size-experiment.json \
  input/iter2
```

需要在相同输入、模型和运行环境下做完整 A/B 时，使用项目内比较程序；不要另写临时汇总脚本：

```bash
.venv/bin/python compare_pipeline_profiles.py \
  --profiles 448 640 \
  --output-dir output/640-adaptation \
  input/iter1 input/iter2 input/iter2-原图 input/iter3
```

需要同时审阅两条 profile 的中间图时，可在比较命令中增加 `--export-processing-stages`，该参数会原样传给每个批次运行。

程序会分别调用固定批次入口，并生成：

- `output/640-adaptation/profile-comparison.json`
- `output/640-adaptation/profile-comparison.html`
- `output/640-adaptation/<profile>/<batch>/instrument-report.{json,html}`

比较报告只说明两个程序 profile 的输出是否变化，不把任一方当成人工真值。只有有 observation 的通道才能判断正确率变化；其余变化必须标为漂移、覆盖增减或状态变化，不能宣称改善。当前默认是 `448-highres-pad`；任何后续 profile 仍不得在未通过全部保留批次回归前替换默认值。

双分辨率实验 profile：

- `448-highres`：低分辨率定位图负责表盘检测和已验证的指针分割；指针掩膜映射到高分辨率表盘裁剪后，再做椭圆校正、OCR 和几何读取。
- `448-highres-seg`：诊断用，指针分割也读取高分辨率裁剪。
- `448-highres-pad`：当前默认。在 `448-highres` 基础上将非正方形分割输入保持比例补边，并对彩色刻度同时复核低分辨率几何分支，按分支内几何一致性选择结果。

2026-09-03 修复候选完整性、外圈几何、类型签名和双分辨率彩色刻度后，对 `iter1`、`iter2`、`iter2-原图`、`iter3` 共 19 张图、36 个通道重新执行真实 A/B：`448-highres-pad` 相对 `448` 为 0 个覆盖下降、0 个状态变化。逐张审阅最终候选框和指针箭头后，它修复了原图 KEOI 被反向读取为 9.49 的问题，并保留 JCQ 三联表、SHM-D 双指针、JS-9、D96-V 和矩形多表盘结果，因此升为默认。比较报告位于 `output/profile-final-comparison-v2/profile-comparison.{json,html}`（本地隐私输出，不进入 Git）。

macOS 上底层 ONNX Runtime 曾在批次已经导出后于解释器清理阶段触发 `SIGABRT`。批次入口现在先显式关闭 OCR session、完成文件写入并刷新标准输出，再跳过这一段不安全的原生析构；业务错误返回码仍照常保留。比较程序仍兼容旧入口：只对导出后 `SIGABRT` 做受控容错，并且必须确认本次 JSON/HTML 完整、profile 正确、图片顺序和记录数匹配；任何其他非零退出或陈旧报告立即失败。

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
9. 只有显式启用 `--export-processing-stages` 时，每张图片卡片才增加可折叠的处理阶段画廊；缩略图内嵌，原尺寸 PNG 作为同目录 sidecar 保留，JSON 记录其真实尺寸和几何变换。

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
