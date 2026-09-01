# Industrial Gauge Reader PoC

面向 Apple Silicon Mac 的工业指针仪表和摄像头闹钟读数实验项目。目标是优先使用预训练模型与通用计算机视觉算法，验证完整画面中的仪表定位、指针识别、刻度解释、连续跟踪和延迟表现。

## 项目结构

### `analog-gauge-poc`

静态工业仪表图片 PoC：

- 从整张图片自动寻找表盘。
- 识别中心、指针和角度。
- 结合 OCR、刻度和颜色区域推算读数。
- 查询带证据来源的仪表类型知识，区分同一仪表上的多个读数通道，并将原始视觉读数解释为可审计的业务读数。
- 支持批量回归测试和 HTML 报告。

使用说明见 [`analog-gauge-poc/README.md`](analog-gauge-poc/README.md)。

处理用户确认的现场仪表图片或生成固定格式图文报告时，AI Agent 必须遵循 [`analog-gauge-poc/docs/user-instrument-batch-runbook.md`](analog-gauge-poc/docs/user-instrument-batch-runbook.md)。

### `camera-clock-poc`

MacBook 摄像头连续跟踪 PoC：

- 使用预训练模型寻找闹钟。
- 自动校正表盘透视和方向。
- 根据连续运动识别秒针。
- 记录逐帧结果、报警状态、截图和中文报告。
- 区分可复用的仪表能力和闹钟演示专属逻辑。

使用说明见 [`camera-clock-poc/README.md`](camera-clock-poc/README.md)，阶段总结见 [`camera-clock-poc/NEXT_STEPS.md`](camera-clock-poc/NEXT_STEPS.md)。

## 本仓库不包含的文件

为了控制仓库体积并保护现场隐私，以下内容不会提交：

- 下载的预训练模型权重。
- Python 虚拟环境和缓存。
- 摄像头 session、逐帧图片和生成报告。
- 用户提供的工业现场图片。
- 本地录制的视频。
- 未确认再分发许可的第三方厂家说明书本地副本。

运行项目前需要按照各子项目 README 下载或准备相应模型和输入文件。

## 许可证

本项目采用双许可模式：

- 默认按 [GNU Affero General Public License v3.0（AGPL-3.0）](LICENSE) 开源。学习、研究和商业使用都可以，但必须完整遵守 AGPL-3.0，包括向网络用户提供对应源码等义务。
- 如果希望将本项目用于闭源产品、企业内部私有系统、闭源 SaaS、客户交付或其他不公开完整对应源码的场景，必须事先取得单独的书面商业许可。
- 第三方依赖、工具和预训练模型遵循各自的许可证与使用条款。

商业许可不能替代 Ultralytics Enterprise 或其他第三方授权。详细说明见 [`COMMERCIAL_LICENSE.md`](COMMERCIAL_LICENSE.md)、[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) 和 [`docs/model-weight-licensing.md`](docs/model-weight-licensing.md)。
