# AI Agent 项目入口

## 现场仪表图片

仓库内的 [`skills/onboard-industrial-gauges/SKILL.md`](skills/onboard-industrial-gauges/SKILL.md) 是新仪表接入 Skill 的版本化源文件。安装后调用 `$onboard-industrial-gauges`；该 Skill 负责读取范围确认、失败分层、metadata/程序/训练决策和真实图片验收。

处理新的现场仪表图片、补充仪表类型知识、修改指针识别逻辑，或生成批次结果前，先完整阅读并执行 [`analog-gauge-poc/docs/user-instrument-batch-runbook.md`](analog-gauge-poc/docs/user-instrument-batch-runbook.md)。该运行手册是批次输入、自动识别报告、输出位置和验收条件的唯一权威文档。

任务只有在运行手册的完成条件全部满足后才算完成。交付时应直接给出本次生成的 HTML 和 JSON 文件路径，并说明程序识别成功、候选、未识别和图片级失败的数量。

用户现场图片、`observations/` 和 `output/` 是本地隐私数据；保持现有 `.gitignore` 边界。仪表类型专业知识写入 `metadata/instrument-types/`。新图自动报告不要求人工 observation；只有用户提供确认值并要求标注回归时，才将确认值写入 `observations/`。

## 其他子项目

摄像头闹钟任务遵循 [`camera-clock-poc/README.md`](camera-clock-poc/README.md)。仓库许可和公开发布边界见根目录 [`README.md`](README.md)。
