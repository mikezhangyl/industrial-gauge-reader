# Iteration 001：分层定位剪枝与连续直针回退

状态：`COMPLETE`

## Observe

输入：`input/gauge-6.jpg`

SHA-256：`795964ce05e9942cab3d4893a68c9a1dde1c25aca5b3e2f3e2ef00121bebaaa2`

人工核验信息：画面外部红框显示 `0.18`，仅用于最终准确性核对，禁止作为表盘刻度或程序输入。

当前真实模型单次结果：

```text
Level 1: PASS
Bounding box: [901, 362, 1051, 473]
Level 2: FAIL
Level 3: FAIL
Failure: Pointer segmenter returned no valid mask
Total: 3933 ms
```

分层检测探针：

| 层级 | 切片数 | 检测耗时 | 观察 |
| --- | ---: | ---: | --- |
| 粗层 | 2 | 81 ms | 已找到两个正确表盘候选，约 `[882,338,1055,481]` |
| 细层 | 33 | 3652 ms | 框更紧，但没有解决指针分割失败 |

当前流程把“指针分割失败”等同于“表盘未定位”，因此在粗层已经找到表盘后仍继续细分。这是定位与读取耦合造成的性能问题。

诊断放大图显示：

- 表盘中心存在明显红色圆形轴心；
- 左上方存在一条从轴心附近向刻度环延伸的连续细直线；
- 中心下方存在粗机械结构，不能简单选择最黑或最长方向；
- PP-OCRv6 当前只能得到不一致的低可信候选，尚不足以建立数值量程。

## Hypothesize

### H1：定位与指针分割耦合导致无效细分

如果粗层候选通过表盘几何验证后立即结束定位，而把指针失败交给独立回退，那么 `gauge-6` 不会进入 33 个细切片，单次耗时应显著下降。

### H2：中心约束的连续直线可以恢复 Level 2

如果从彩色圆形轴心出发，只接受接近中心、连续、笔直、较细且延伸到刻度环的径向线，那么左上指针应成为唯一或最高分候选，粗机械结构应被降权。

### H3：当前 OCR 不足以直接完成 Level 3

如果透视校正和多种增强后仍无法得到至少两个位置一致的数字锚点，那么本轮必须保持 `Reading: N/A`，不能读取外部红框 `0.18` 或凭单个字符硬猜量程。

### H4：刻度序列约束可以成为下一轮入口

如果后续 OCR 能保留 `0.1、0.2` 或 `2、3` 等低可信候选及坐标，再与长刻度的等角间距联合拟合，则可以把缺失的 `0.3` 标记为 `inferred`，而不是伪装成 OCR 明文。

## Act

本轮只实现：

1. 将表盘候选是否成立与指针分割是否成功解耦；
2. 粗层出现已验证候选后停止继续细分；
3. 指针分割失败时，增加彩色轴心与连续直线的几何回退；
4. 保存指针候选、中心、角度和置信度到可视化；
5. 不实现数字序列补全，留给后续迭代。

## Verify

本轮验收标准：

- `gauge-6`：Level 1 PASS；
- `gauge-6`：Level 2 PASS，并且可视化指向人工可见的左上细指针；
- `gauge-6`：如果没有可靠刻度锚点，Level 3 必须保持 FAIL；
- 不使用外部红框 `0.18`；
- 不进入 33 个细切片，CPU 单次延迟目标先恢复到 `< 500 ms`；
- 前五张图读数保持约 `2.62、2.40、3.12、0.94 relative、4.92`；
- 单元测试、Ruff 和类型检查通过；
- 生产代码中不得出现 `gauge-6` 的坐标、角度或最终读数。

## Result

### 实现结果

- 新增通用 `colored-hub+line-segment` 指针回退：先寻找表盘中心附近的彩色紧凑轴心，再用 LSD 线段检测、径向一致性、连续性和候选分离度选择指针；
- 回退不限定红色轴心，也未写入本图坐标、角度或读数；
- 表盘检测候选不再因为指针模型无 mask 而直接判负；
- `gauge-6` 在整图检测候选上已完成几何回退，实测执行的切片层级为 `[]`，没有进入原来的 2 个粗切片或 33 个细切片。

### `gauge-6` 真实链路结果

```text
Level 1: PASS
Bounding box: [878, 339, 1061, 488]
Level 2: PASS
Pointer angle: 300.15 deg
Pointer method: colored-hub+line-segment+edge-ellipse+affine-rectification
Level 3: FAIL
Reading: N/A
Failure: OCR found only 0 numeric scale candidates
```

可视化人工检查：框位于真实表盘，中心位于红色轴心，指针线沿左上方细直针；没有沿下方粗机械结构，也没有使用画面外部红框 `0.18`。

### 正式 steady-state benchmark

每个设备 warm-up 5 次，正式执行 20 次；模型加载时间未计入 steady-state。

| 设备 | Level 1 | Level 2 | Level 3 | p50 | p95 | min | max | latency |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| CPU | PASS | PASS | FAIL | 313.74 ms | 319.22 ms | 309.89 ms | 322.26 ms | PASS |
| MPS | PASS | PASS | FAIL | 319.27 ms | 345.86 ms | 311.61 ms | 356.94 ms | PASS |

本图的 MPS 没有带来收益；CPU 的 p50 和 p95 都略低。两个设备均满足 `p95 < 500 ms`。

### 六张真实图片回归

| 图片 | Level 1 | Level 2 | Level 3 | 读数 | CPU 单次耗时 |
| --- | --- | --- | --- | ---: | ---: |
| `gauge.png` | PASS | PASS | PASS | 2.6207 | 297.13 ms |
| `gauge-2.png` | PASS | PASS | PASS | 2.4004 | 270.46 ms |
| `gauge-3.jpg` | PASS | PASS | PASS | 3.1169 | 459.73 ms |
| `gauge-4.jpg` | PASS | PASS | PASS | 0.9393 relative | 306.11 ms |
| `gauge-5.jpg` | PASS | PASS | PASS | 4.9159 | 478.70 ms |
| `gauge-6.jpg` | PASS | PASS | FAIL | N/A | 309.59 ms |

这些是同一个常驻 reader 的单次功能回归结果，不替代上面的 20 次正式 benchmark。

### 自动检查

```text
unittest: 15 passed
Ruff: passed
ty: passed
production hard-code grep: no matches
```

证据文件：

- `output/result-6-cpu.jpg`
- `output/result-6-mps.jpg`
- `output/rectified-6-cpu.jpg`
- `output/scale-ring-6-cpu.jpg`

## Reflect

- H1 成立：原始约 3933 ms 降至 CPU p50 313.74 ms；新链路甚至在整图候选上结束，未执行任何切片。
- H2 成立：通用彩色轴心与连续径向直线恢复了左上细指针，Level 2 通过。
- H3 成立：OCR 没有得到可靠刻度锚点，程序保持 `Reading: N/A`，所以“程序链路成功”不等于“业务读数正确”。
- H4 尚未验证，进入下一轮候选。

下一轮建议建立 `Iteration 002`，只处理“低可信 OCR 候选保留 + 长短刻度检测 + 单调等距序列联合拟合”。推断数字必须标记为 `inferred`，原始 OCR 明文标记为 `explicit`；若证据不足，继续输出 `N/A`。
