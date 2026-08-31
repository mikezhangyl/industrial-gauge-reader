# MacBook 摄像头闹钟 PoC

这是一个与 `analog-gauge-poc` 完全隔离的演示项目。目标是在 MacBook 摄像头的完整画面中用预训练 COCO 模型寻找闹钟，再用连续帧确认约每秒顺时针转动 `6°` 的秒针，记录秒数，并在 `50 <= 秒数 < 60` 时演示报警。

它验证摄像头取流、连续识别、时间记录、事件截图和报告链路，不作为工业压力表准确率证明。

## 安装

```bash
cd /Users/mikezhang/Coding/AI-Learning/industrial-gauge-reader/camera-clock-poc
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

## 运行

```bash
python camera_poc.py --source 0 --sample-fps 2
```

- 首次运行时，macOS 会请求摄像头权限；请允许终端或 Codex 使用摄像头。
- `q`：结束实验并生成报告。
- `s`：立即保存当前截图。
- 没有闹钟时应输出“未检测到闹钟”，不能报警。
- 闹钟刚进入画面时通常需要约 1～2 秒积累运动证据；确认前显示“未知”，不会猜读数。
- 一旦锁定秒针，单帧模糊或漏线会短时预测保持最多 1.25 秒；画面会明确显示“短时保持”，置信度同步下降。
- 秒针跟踪会在启动阶段比较普通灰度与局部对比度两路运动证据，自动锁定更适合当前表盘材质的一路；备用通道接手时还会检查角度是否与上一读数连续，避免反光造成读数跳变。
- 程序先用表盘外圈和指针轴心恢复近大远小的透视变形，再用刻度线确定网格、用 PP-OCRv6 识别的钟面数字确定 `12/3/6/9` 的真实方向，最后才计算指针角度。
- COCO 模型有时会把手持闹钟标成 `frisbee`。程序会把 `clock/frisbee` 都当作候选，但 `frisbee` 必须通过表盘数字验证或紧邻已确认闹钟的连续位置，不会盲目当成闹钟。
- 数字参照在同一个表盘姿态下会被锁定，不会每帧重复 OCR；表盘位置或角度明显变化时才重新校准。实时画面和报告会显示透视校正及数字参照数量。
- 识别永远使用清晰原图。隐私虚化只是识别成功后的展示处理：保留闹钟清晰、虚化背景。如果没找到闹钟，画面不虚化并保存清晰原图，以便判断漏检原因。

自动停止示例：

```bash
python camera_poc.py --source 0 --sample-fps 2 --duration 120
python camera_poc.py --source 0 --headless --max-samples 10
```

如果闹钟秒针已经与 Mac 系统秒数对齐，可开启绝对误差对照：

```bash
python camera_poc.py --source 0 --sample-fps 2 --compare-system-time --clock-offset-seconds 0
```

如果闹钟固定快或慢几秒，用 `--clock-offset-seconds` 填写这个固定差值。报告中的“秒数误差p95”才是绝对准确度指标；只看连续运动不能证明读数准确。

每次实验输出到独立目录：

```text
sessions/YYYYMMDD-HHMMSS/
├── records.jsonl
├── frames/           # 成功帧虚化背景；漏检帧保留清晰原图
├── screenshots/      # 与逐帧归档采用相同规则的未标注帧和标注帧
└── report.html
```

逐帧归档默认开启；如只想做轻量演示，可使用 `--no-archive-all-raw-frames` 关闭。

隐私虚化可用 `--no-privacy-blur` 显式关闭。默认的“诊断优先”模式会保存漏检帧的清晰背景，运行前应确认实验环境可以被记录。旧会话不会被自动改写。

## 代码边界

- `src/camera_clock_poc/reusable/`：摄像头、时间、日志等现场可复用候选。
- `src/camera_clock_poc/clock_demo/`：秒针、秒数换算和50～59秒报警，只用于演示。
- 本项目不导入或修改 `analog-gauge-poc/src/`。

详细规则见 `docs/code-boundaries.md`。
