# 代码边界

## 不可违反的约束

1. `camera-clock-poc` 不导入、不修改 `analog-gauge-poc/src/`。
2. 闹钟专属代码不得进入 `reusable/`。
3. `reusable/` 不得依赖 `clock_demo/`。
4. 候选复用算法只有经过现有六张工业仪表图片回归后，才能进入工业项目。

## 分类

| 路径 | 分类 | 内容 |
| --- | --- | --- |
| `reusable/types.py` | `REUSABLE` | 帧、观察结果和 reader 接口 |
| `reusable/capture.py` | `REUSABLE` | 最新帧摄像头取流，不堆积旧帧 |
| `reusable/session.py` | `REUSABLE` | JSONL、隐私处理后逐帧归档、事件未标注帧/标注帧和实验目录 |
| `reusable/privacy.py` | `REUSABLE` | 只在仪表检测成功后虚化背景；不参与识别，漏检帧保留清晰原图便于诊断 |
| `clock_demo/object_detector.py` | `CLOCK_DEMO_ONLY` | COCO `clock/frisbee` 只用于生成候选；误标为飞盘的候选还必须通过钟面数字或连续位置验证，不能替代工业仪表检测器 |
| `clock_demo/reader.py` | `REUSABLE_CANDIDATE` | 用表盘外圈和轴心完整恢复透视、用刻度线稳定方向，并按运动证据在普通灰度和局部对比度间选择后再做时序跟踪；其中秒针每秒 `6°` 的规则仅属于闹钟 |
| `clock_demo/scale_reference.py` | `CLOCK_DEMO_ONLY` | 用 PP-OCRv6 识别 `1～12` 并恢复钟面固定方向；OCR 坐标和刻度结合的思路可复用，但钟面数字映射不能直接进工业项目 |
| `clock_demo/alarm.py` | `CLOCK_DEMO_ONLY` | 50～59秒报警 |
| `clock_demo/overlay.py` | `CLOCK_DEMO_ONLY` | 闹钟秒数中文可视化 |
| `clock_demo/report.py` | `CLOCK_DEMO_ONLY` | 闹钟实验中文报告 |

## Reader 接缝

摄像头实验只依赖一个小接口：

```python
class FrameReader(Protocol):
    def read(self, frame: np.ndarray, captured_at: datetime) -> Observation: ...
```

未来可以用 `IndustrialGaugeReader` 替换 `ClockSecondHandReader`，摄像头、日志和实验运行逻辑不需要知道内部识别方式。

本轮为了消除“把人脸当闹钟”的误检，使用预训练 `YOLO11n` 的 COCO `clock` 类。这一门禁是演示专属，不进入 `analog-gauge-poc`。
