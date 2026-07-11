# C-MET 论文指标输入协议

仓库提供 benchmark 生成、稳定 sample ID、覆盖率检查和指标汇总层。外部评估器必须读取同一 checkpoint、同一情绪协议的 manifest，才能正确合并结果。

## 1. 先固定情绪协议

笔记本变量：

```python
BENCHMARK_EMOTION_PROTOCOL = "dataset"
```

可选值：

- `dataset`：依据论文的 10 条 neutral speech 与 10 条 emotional speech、官方 `Dataset.get_raw_e2v` 和 `test.csv` 重建。MEAD 按测试身份、情绪、`level_1/2/3` 建池；CREMA-D 按情绪、`HI/LO/MD/XX` 建池。
- `official-static`：复用官方仓库公开的静态演示池，不能表达 `test.csv` 的逐样本强度，仅作为官方演示资源对照。

作者没有公开 Table 1 的批量推理脚本。`dataset` 是可审计的强度感知重建协议，`official-static` 是公开演示协议；未经作者确认，两者都不能写成“精确作者 benchmark 协议”。

## 2. Benchmark manifest

`run_benchmark_inference.py` 生成：

```text
benchmark_runs/<checkpoint_source>/<checkpoint_tag>/
  manifests/<emotion_protocol>/mead.csv
  manifests/<emotion_protocol>/crema-d.csv
  progress/<emotion_protocol>/mead.jsonl
  progress/<emotion_protocol>/crema-d.jsonl
  videos/<emotion_protocol>/mead/*.mp4
  videos/<emotion_protocol>/crema-d/*.mp4
```

关键字段：

```text
sample_id
dataset
source_video
source_audio
source_image
gt_video
target_emotion
intensity
output_video
emotion_protocol
neutral_pool
emotion_pool
```

sample ID 包含协议名，例如：

```text
mead_dataset_000001_...
mead_official-static_000001_...
```

任何外部评估器都应把 `sample_id` 原样写回结果 CSV，不要自行重新编号。

汇总脚本会要求 manifest 只包含一个合法的 `emotion_protocol`，并核验命令行 `--emotion-protocol` 与 progress 中记录的协议。任一处不一致都会停止，不生成报告。

## 3. SyncNet 与情绪分类结果

逐样本文件：

```text
sample_id,sync_confidence,predicted_emotion
mead_dataset_000001_...,7.95,angry
```

要求：

- 每个 manifest ID 恰好一行。
- 不允许未知 ID 或重复 ID。
- `predicted_emotion` 会统一大小写和 MEAD/CREMA-D 缩写。
- Emotion-FAN 必须是与论文一致、分别在 benchmark 上微调的模型，才可称为论文 Accemo。

## 4. AITV、FID、FVD

全局文件：

```json
{
  "aitv": 2.643,
  "fid": 90.804,
  "fvd": 329.862
}
```

必须同时记录：

- checkpoint 和情绪协议。
- GPU 型号与推理精度。
- 模型是否预加载、是否预热。
- AITV 是否包含视频封装和磁盘 I/O。
- 视频数量与长度分布。
- FID/FVD 特征模型版本、抽帧和聚合策略。

## 5. 仓库内部 AITV

persistent 后端的 `inference_seconds`：

- 不包含模型初始化。
- 包含单样本 emotion direction、表情预测、渲染和视频封装。
- 使用同一模型进程连续处理 benchmark。

可用 `--use-progress-aitv` 汇总为本机诊断值，但作者没有公开精确计时协议，不能自动认定与论文 AITV 等价。

## 6. 严格汇总

以 `dataset` 协议的 MEAD 为例：

```bash
python scripts/evaluate_paper_metrics.py \
  --benchmark-manifest benchmark_runs/.../manifests/dataset/mead.csv \
  --sample-metrics external_paper_metrics/.../dataset/mead_sample_metrics.csv \
  --global-metrics external_paper_metrics/.../dataset/mead_global_metrics.json \
  --progress benchmark_runs/.../progress/dataset/mead.jsonl \
  --dataset mead \
  --emotion-protocol dataset \
  --experiment paper_main \
  --out-dir reports/strict_paper_metrics/<checkpoint_tag>/dataset \
  --strict
```

输出：

```text
paper_main_mead_paper_metrics.json
paper_main_mead_paper_metrics.csv
paper_main_mead_emotion_accuracy.csv
paper_main_mead_paper_metrics.md
```

若切换为 `official-static`，命令中的 manifest、sample metrics、global metrics、progress 和输出目录必须全部一起切换，不能混用。

## 7. 公开材料限制

官方没有发布精确 FID/FVD/SyncNet 实现、Table 1 批量推理脚本和微调 Emotion-FAN，因此仓库不会生成虚假的论文指标。替代评估器可以用于内部比较，但报告标题应写“自定义协议”，并列出与论文的差异。
