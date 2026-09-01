# 仪表自动识别批次运行手册

本文件是现场仪表新图片处理和批次报告格式的唯一权威规范。新图片默认没有人工答案；HTML 只展示本次程序识别结果。除非用户明确要求改变格式，后续 AI Agent 持续生成相同结构的自包含 HTML 和同名 JSON。

## 完成条件

一批新图片只有同时满足以下条件才算完成：

1. 每张输入图片在 JSON 中有一个 `record`，在 HTML 中有一个对应图片卡片，顺序与命令行输入一致。
2. HTML 可以脱离原始临时图片打开，并只展示原图及本次程序产生的结果、状态、方法、原始显示和说明。
3. JSON 包含 `automated_summary`，并分别统计识别到的仪表类型、程序通道、成功、候选、未识别和图片级失败。
4. 未唯一匹配仪表类型或未识别通道时，HTML 和 JSON 明确显示失败，不补入人工猜测值。
5. `output/` 中同时存在同名 `.json` 和 `.html`，HTML 中内嵌图片数与输入图片数一致。
6. 相关测试、Ruff、类型检查和 `git diff --check` 通过。交付结论写成“当前验证未发现回归”。

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

程序依靠整图 OCR 唯一匹配类型 metadata。没有唯一匹配时，报告应显示图片级失败；先补齐可核验的类型知识，再重新运行。

### 3. 运行固定自动报告

保持图片按用户提供顺序传入。新图默认命令不使用人工验收参数：

```bash
.venv/bin/python batch_instrument_report.py \
  --device cpu \
  --output output/user-instrument-batch.json \
  "/absolute/path/to/image-1.jpg" \
  "/absolute/path/to/image-2.jpg"
```

默认固定输出：

- `output/user-instrument-batch.json`：机器可读的本次运行数据和可选审计字段。
- `output/user-instrument-batch.html`：只展示程序识别结果的自包含图文报告。

需要保留多个批次时，只改变 `--output` 的 JSON 文件名；HTML 自动使用相同文件名主干。例如 `--output output/2026-09-02-substation-a.json` 会同时生成 `output/2026-09-02-substation-a.html`。

`output/` 被 Git 忽略。需要长期归档时，将同名 JSON 和 HTML 一起复制到经用户确认的安全位置。

### 4. 核对程序结果

```bash
jq '.automated_summary' output/user-instrument-batch.json
open output/user-instrument-batch.html
```

依次检查：

- `instrument_types_recognized` 是否覆盖预期图片；
- `recognized`、`ambiguous`、`not_recognized` 和 `analysis_failures`；
- `records[].channels[].automated` 中的值、候选、状态、方法、原始 OCR 和说明；置信度只在 JSON 中保留供技术审计；
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
5. 图片以 data URI 嵌入 HTML，使报告不依赖 `/var/folders/...` 等临时文件路径。
6. `recognized`、`ambiguous` 和 `not_recognized` 使用清晰、可区分的状态样式；窄屏自动改为单列。
7. 图片级识别失败仍保留原图卡片和失败原因。

HTML 不展示置信度、observation、人工确认值、自动/人工比对、最终复核值或最终值来源。这些技术或可选审计字段只保留在 JSON。

格式变更需要同时更新 `render_html()`、针对性回归测试和本文件，并在真实图片生成的报告上完成浏览器检查。用户未明确要求变更时，维持本契约。

## 交付格式

向用户交付时直接提供：

- HTML 报告的绝对路径链接；
- 同批 JSON 的绝对路径链接；
- 程序识别到的类型数、通道数、成功数、候选数、未识别数和图片级失败数；
- 本次实际执行的测试和检查结果。

以本次 JSON 的 `automated_summary` 和 `records[].channels[].automated` 为事实来源。
