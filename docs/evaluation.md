# C-MET 结果评估

评估分为三层，不能混用。

## 1. 视频技术有效性

命令：

```bash
python scripts/evaluate_videos_basic.py \
  --video-dir /path/to/videos \
  --out-csv /path/to/technical.csv \
  --report /path/to/technical.json \
  --strict
```

检查内容：

- 视频能被 `ffprobe` 读取。
- 分辨率为 256x256。
- FPS 为 25，允许 0.05 误差。
- 时长大于 0。
- 有音频流。

这一层只能证明文件有效，不能证明情绪、身份、同步或视觉质量达到论文结果。

## 2. 定性评估

每个结果至少观察：

- 目标情绪是否明显。
- 身份是否稳定。
- 头部姿态是否延续中性驱动视频。
- 嘴型是否与中性驱动音频一致。
- 是否出现脸崩、闪烁、局部纹理漂移或夸张异常。

建议同时展示：

- 中性输入。
- 基础情绪输出，例如 happy、angry。
- 扩展情绪输出，例如 sarcastic、romantic。
- 失败案例。

扩展情绪没有对应 ground-truth 视频，论文使用用户研究，不应套用基础情绪分类器准确率。

## 3. 论文五项指标

论文 Table 1 使用：

```text
AITV
FID
FVD
SyncNet confidence
Emotion-FAN accuracy
```

汇总命令：

```bash
python scripts/evaluate_paper_metrics.py \
  --benchmark-manifest benchmark_runs/.../manifests/dataset/mead.csv \
  --sample-metrics external/mead_sample_metrics.csv \
  --global-metrics external/mead_global_metrics.json \
  --progress benchmark_runs/.../progress/dataset/mead.jsonl \
  --dataset mead \
  --emotion-protocol dataset \
  --experiment paper_main \
  --out-dir reports/strict_paper_metrics \
  --strict
```

上例中的 `dataset` 是情绪协议目录。若运行 `official-static`，manifest、progress、视频和外部指标必须全部切换到 `official-static`，不能跨协议拼接。

逐样本 CSV 必须包含：

```text
sample_id,sync_confidence,predicted_emotion
```

全局 JSON 必须包含：

```json
{
  "aitv": 2.5,
  "fid": 90.0,
  "fvd": 320.0
}
```

## 4. 论文主方法目标值

### MEAD

| 指标 | 论文值 |
|---|---:|
| AITV | 2.643 |
| FID | 90.804 |
| FVD | 329.862 |
| Sync confidence | 7.9996 |
| Emotion accuracy | 55.91% |

### CREMA-D

| 指标 | 论文值 |
|---|---:|
| AITV | 1.561 |
| FID | 50.028 |
| FVD | 309.828 |
| Sync confidence | 6.2887 |
| Emotion accuracy | 43.47% |

## 5. 损失消融目标值

论文 Table 6，MEAD：

| 实验 | FID | FVD | Sync confidence | Emotion accuracy |
|---|---:|---:|---:|---:|
| `Lrecon` | 88.951 | 325.926 | 7.9892 | 49.43% |
| `Lrecon + Lcnt` | 88.082 | 321.961 | 8.0018 | 53.46% |
| 完整损失 | 90.804 | 329.862 | 7.9996 | 55.91% |

三组的 AITV 都是 2.643。

## 6. AITV 的计时边界

`run_benchmark_inference.py` 的 persistent 后端记录：

- 模型已加载。
- 单样本 emotion direction 计算。
- 表情预测。
- 渲染。
- 视频封装。

它不包含模型初始化时间。只有当硬件、视频长度分布、预热、同步策略和作者协议一致时，才能和论文 AITV 直接比较。

## 7. Benchmark 情绪协议对结果的影响

- `dataset`：MEAD 按目标 `level_1/2/3`、CREMA-D 按 `HI/LO/MD/XX` 构建 10-shot 情绪语音特征，保留 benchmark 强度。
- `official-static`：使用官方公开演示池，基础情绪来自静态演示资源，不随 `test.csv` 强度变化。

论文写明随机抽取 10 条 neutral 和 10 条 emotional speech，计算语义向量均值，并由语音本身编码情绪强度；但作者没有公开 Table 1 的批量推理脚本。因此默认 `dataset` 是可审计重建协议，`official-static` 是官方演示资源对照，二者都不能未经作者确认就称为精确作者协议。

## 8. 为什么仓库不直接计算“论文指标”

作者没有公开：

- FID/FVD 的精确帧采样、特征模型版本和聚合代码。
- SyncNet 版本、裁剪与同步预处理细节。
- 分别在两个 benchmark 上微调的 Emotion-FAN checkpoint。

因此仓库提供输入清单、稳定 sample ID、覆盖率检查、目标值和汇总层，但不伪造外部评估器。使用替代实现时，应写成“自定义协议结果”，不能写成“严格复现论文数值”。
