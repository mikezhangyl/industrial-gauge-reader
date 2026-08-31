# 第三方组件与模型通知

本项目源码采用 AGPL-3.0，但运行时会使用第三方库、工具和预训练模型。它们不因本项目的许可证或商业许可而被重新授权。

## 当前主要组件

| 组件或权重 | 上游声明 | 本仓库处理方式 |
| --- | --- | --- |
| Synanthropic `corners-best.pt`、`keypoints-best.pt` | 上游 Space 标记 MIT；模型由 Ultralytics Pose 工具链产生 | 权重不提交；保留 MIT 声明，并按 Ultralytics AGPL-3.0/Enterprise 边界核查 |
| ETHZ `ethz-gauge-detection.pt`、`ethz-segmentation.pt` | 上游仓库 MIT；模型使用 Ultralytics YOLOv8 | 权重不提交；保留 MIT 声明，并按 Ultralytics AGPL-3.0/Enterprise 边界核查 |
| Ultralytics `yolo11n.pt` 与推理代码 | AGPL-3.0 或 Enterprise | 权重不提交；开源路线遵守 AGPL-3.0，闭源路线另行取得 Enterprise |
| RapidOCR / PaddleOCR 与 PP-OCRv6 ONNX 权重 | Apache-2.0 | 随依赖安装；再分发时保留许可证、归属和 NOTICE |

## 一手来源

- [Synanthropic Space 与文件清单](https://huggingface.co/api/spaces/Synanthropic/reading-analog-gauge)
- [Synanthropic MIT LICENSE](https://huggingface.co/spaces/Synanthropic/reading-analog-gauge/blob/main/LICENSE)
- [ETHZ analog_gauge_reader MIT LICENSE](https://github.com/ethz-asl/analog_gauge_reader/blob/main/LICENSE.md)
- [Ultralytics Licensing](https://www.ultralytics.com/license)
- [RapidOCR LICENSE](https://github.com/RapidAI/RapidOCR/blob/main/LICENSE)
- [PaddleOCR LICENSE](https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE)

更完整的权重来源、哈希和风险判断见 [`docs/model-weight-licensing.md`](docs/model-weight-licensing.md)。

> 本文件是工程合规记录，不构成法律意见。正式商业发布前应由法务根据实际组合和部署方式复核。
