# 模型权重许可证核查

核查日期：2026-08-31

> 本文是基于上游公开条款的工程合规梳理，不构成法律意见。商业发布前应由法务复核实际部署方式和最终合同。

## 结论

本仓库现已采用 AGPL-3.0，与 Ultralytics 的开源许可路线方向一致。当前 PoC 可以继续用于研究、实验和符合 AGPL-3.0 的商业使用，但**不能仅凭本仓库的许可证，宣称整套方案可以闭源商用**。

主要原因是：五个 `.pt` 权重全部通过 Ultralytics YOLO 工具链运行，其中四个第三方权重虽然随 MIT 仓库发布，但 Ultralytics 当前官方许可页又声明，使用其代码、架构、训练管线或训练所得模型，需要选择 AGPL-3.0 合规或 Enterprise 许可证。两层声明存在需要法务确认的边界。

| 权重 | 上游项目声明 | Ultralytics 层 | 闭源商用判断 |
| --- | --- | --- | --- |
| `corners-best.pt` | Synanthropic Space：MIT | Ultralytics Pose，官方称训练所得模型默认受 AGPL-3.0 约束 | 高风险，先确认或采购 Enterprise |
| `keypoints-best.pt` | Synanthropic Space：MIT | 同上 | 高风险，先确认或采购 Enterprise |
| `ethz-gauge-detection.pt` | ETHZ 仓库：MIT | YOLOv8 检测模型 | 高风险，存在 MIT 与 Ultralytics 条款叠加 |
| `ethz-segmentation.pt` | ETHZ 仓库：MIT | YOLOv8 分割模型 | 高风险，存在 MIT 与 Ultralytics 条款叠加 |
| `yolo11n.pt` | Ultralytics 官方权重 | AGPL-3.0 或 Enterprise | 高风险；闭源产品通常需要 Enterprise |
| RapidOCR PP-OCRv6 ONNX | RapidOCR/PaddleOCR：Apache-2.0 | 不依赖 Ultralytics | 相对低风险，保留许可证和通知即可 |

## 1. 本仓库实际使用的文件

### 手工保存的五个 PyTorch 权重

权重路径和 SHA-256：

| 本地文件 | SHA-256 | 来源 |
| --- | --- | --- |
| `analog-gauge-poc/models/corners-best.pt` | `a88502e86a40941aec69fe4d48e03c675a9381500fbf4c1ca8e3d1a89db089a9` | Synanthropic |
| `analog-gauge-poc/models/keypoints-best.pt` | `8b9b6ac6b3e5dd73a4e00af18365b13e0607cb8e4b00cfd9189e005b86103124` | Synanthropic |
| `analog-gauge-poc/models/ethz-gauge-detection.pt` | `eb496ed0c007b890fe7d26da69a777076cb8f28f0166f2a5905e971d85d15db8` | ETHZ-ASL |
| `analog-gauge-poc/models/ethz-segmentation.pt` | `28efb6aa2ebd9032b55ed29636c937b6410c869e3775435bde2c2ac167536cbf` | ETHZ-ASL |
| `camera-clock-poc/models/yolo11n.pt` | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` | Ultralytics 官方 release `v8.3.0` |

前四个下载地址和校验值固定在 [`analog-gauge-poc/src/model_store.py`](../analog-gauge-poc/src/model_store.py)。本地 `yolo11n.pt` 的 SHA-256 与 [Ultralytics Assets `v8.3.0` 官方文件](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt)一致。

### RapidOCR 随包安装的 ONNX 权重

本项目安装的 `rapidocr==3.9.2` 默认使用：

- `PP-OCRv6_det_small.onnx`
- `PP-OCRv6_rec_small.onnx`
- `ch_ppocr_mobile_v2.0_cls_mobile.onnx`

这些文件随 RapidOCR 包提供，不在本仓库的 `models/` 目录中，但仍属于实际推理链的一部分。

## 2. Synanthropic 两个 Pose 权重

Hugging Face Space 的元数据明确标记 `license: mit`，同一文件树同时包含 MIT `LICENSE`、`corners-best.pt` 和 `keypoints-best.pt`，没有发现针对权重单独设置的附加条款：

- [Space API 元数据和文件清单](https://huggingface.co/api/spaces/Synanthropic/reading-analog-gauge)
- [MIT LICENSE](https://huggingface.co/spaces/Synanthropic/reading-analog-gauge/blob/main/LICENSE)
- [`corners-best.pt`](https://huggingface.co/spaces/Synanthropic/reading-analog-gauge/blob/main/corners-best.pt)
- [`keypoints-best.pt`](https://huggingface.co/spaces/Synanthropic/reading-analog-gauge/blob/main/keypoints-best.pt)
- [`requirements.txt` 固定 Ultralytics 8.1.2](https://huggingface.co/spaces/Synanthropic/reading-analog-gauge/blob/main/requirements.txt)

按 Space 自身的仓库级声明，MIT 允许使用、修改、分发和商业使用，但再分发时必须保留版权和许可文本。

风险在于这两份文件是 Ultralytics Pose 模型。Ultralytics 当前许可页声明，其代码、模型架构、训练管线以及训练/微调产生的模型，需要按 AGPL-3.0 开源整个项目，或者取得 Enterprise 许可证。由于上游同时给出 MIT，实际商业边界存在冲突或叠加解释，不能由我们单方面认定 MIT 已完全解除 Ultralytics 层的要求。

**工程判断：**非商业本地 PoC 可继续；闭源商业部署前，应向 Synanthropic 和 Ultralytics 取得书面确认，或者购买适用的 Ultralytics Enterprise 许可。

## 3. ETHZ 检测和分割权重

ETHZ-ASL 仓库根目录采用 MIT，三个模型文件直接保存在同一仓库中，没有发现模型文件单独的附加许可证：

- [ETHZ MIT LICENSE](https://github.com/ethz-asl/analog_gauge_reader/blob/main/LICENSE.md)
- [模型目录](https://github.com/ethz-asl/analog_gauge_reader/tree/main/models)
- [项目依赖 Ultralytics 8.0.66](https://github.com/ethz-asl/analog_gauge_reader/blob/main/pyproject.toml)
- [YOLOv8 表盘检测代码](https://github.com/ethz-asl/analog_gauge_reader/blob/main/gauge_detection/detection_inference.py)
- [YOLOv8 指针分割代码](https://github.com/ethz-asl/analog_gauge_reader/blob/main/segmentation/segmenation_inference.py)

MIT 本身允许商业使用和再分发，但要求保留版权与许可文本。不过，这两份权重分别由 Ultralytics YOLOv8 检测和分割接口加载，因此同样会遇到 Ultralytics 当前声明的 AGPL/Enterprise 层。

**工程判断：**权重作者侧的条款比不明来源权重清晰，但闭源商业部署风险不能标成“低”；建议按高风险处理，直到 Enterprise 许可或书面确认覆盖这一组合。同时还应独立核查训练数据权利，MIT 软件许可证不自动保证训练数据的商业权利。

## 4. `yolo11n.pt`

本地权重与 Ultralytics 官方 `v8.3.0` release 文件哈希完全一致，因此来源明确。Ultralytics 官方给出两条路径：

1. AGPL-3.0：可以使用，包括商业使用，但需要满足强开源义务；官方当前解释要求公开完整项目及相关组件。
2. Enterprise：用于不希望履行 AGPL 开源义务的内部工具、私有系统、SaaS、商业产品、硬件或边缘设备。

一手来源：

- [Ultralytics 官方 Licensing 页面](https://www.ultralytics.com/license)
- [Ultralytics 仓库 License 说明](https://github.com/ultralytics/ultralytics#-license)
- [AGPL-3.0 原文](https://github.com/ultralytics/ultralytics/blob/main/LICENSE)

**工程判断：**公开且完整遵守 AGPL 的方案可以继续使用；闭源商业产品、企业内部私有部署或客户交付，应先取得适用的 Enterprise 许可证。我们自己的商业授权不能代替 Ultralytics 授权。

## 5. RapidOCR / PP-OCRv6 权重

本地安装包元数据表明 `rapidocr==3.9.2` 为 Apache-2.0；RapidOCR 和 PaddleOCR 官方仓库也都使用 Apache-2.0：

- [RapidOCR LICENSE](https://github.com/RapidAI/RapidOCR/blob/main/LICENSE)
- [PaddleOCR LICENSE](https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE)

Apache-2.0 允许商业使用、修改和再分发，并提供明确的专利授权。再分发时需要附带许可证、保留归属/NOTICE，并对修改文件作显著说明。

**工程判断：**当前 OCR 权重是许可证风险最低的一组，可以作为商业方案的优先保留组件；正式打包时仍要生成第三方许可证和 NOTICE 清单。

## 6. 对当前公开仓库的影响

当前 GitHub 仓库没有提交任何 `.pt` 或 ONNX 权重，只提交了：

- 下载地址与 SHA-256；
- `ultralytics`、`rapidocr` 等依赖声明；
- 调用这些依赖的项目代码。

这减少了直接再分发第三方权重的义务，但不消除运行和部署组合后的许可证要求。尤其是：

- 本仓库的 AGPL-3.0 只覆盖我们有权许可的源码；
- 它不能覆盖或重新授权第三方权重；
- 运行或部署完整组合时，应对受 AGPL 覆盖的完整对应源码履行网络源码提供等义务，并同时保留 MIT、Apache-2.0 等第三方许可证与通知；
- 如果不希望履行 AGPL-3.0 义务，需要分别取得本项目和相关第三方所需的商业许可。

## 7. 建议采取的动作

### 当前非商业 PoC

- 可以继续本地运行。
- 保留本文件、下载 URL、SHA-256 和上游许可证快照。
- 不把第三方权重直接提交到我们的 GitHub release 或安装包。

### 闭源商业 PoC 或生产部署

1. 先向 Ultralytics 申请 Enterprise 授权范围与报价，并明确询问是否覆盖：
   - 官方 `yolo11n.pt`；
   - Synanthropic 使用 Ultralytics 训练的 Pose 权重；
   - ETHZ 使用 Ultralytics 训练的检测/分割权重；
   - 内部部署、客户现场部署和边缘设备。
2. 保存 Synanthropic 与 ETHZ 的 MIT LICENSE，并向权重作者确认模型权重和训练数据可用于计划中的商业场景。
3. 为 RapidOCR/PaddleOCR 准备第三方许可证和 NOTICE。
4. 在上述问题确认前，不对外承诺“购买我们的商业授权后即可合法使用完整方案”。

### 如果不购买 Ultralytics Enterprise

- 路线 A：整个组合项目按 AGPL-3.0 的要求公开，建立完整对应源码发布和网络用户源码获取机制，并保留所有第三方许可证与通知。
- 路线 B：替换 Ultralytics 推理库和所有由 Ultralytics 工具链产生的模型，使用许可证边界明确的商业友好模型/框架，再重新做准确率和延迟回归。

## 8. 上游版本快照

- Synanthropic Space SHA：`8222483d7c98faf32581dc70f166150ea4bc6ecf`
- ETHZ-ASL `main`：`518cdfaab1e9c7cdabb1ef957e45311fe3f6a47a`
- Ultralytics `main`：`fa34184a5080c81fff453670394e13303ac781b2`
- RapidOCR `main`：`0e629c8be05635035c01a829d10a91bbcd56a27a`
- PaddleOCR `main`：`2661c7c0ef5c613e8f93c6e93b2e052399f0f854`

许可证和商业政策可能变化，正式决策前应重新核查最新版本。
