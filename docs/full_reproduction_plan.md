# C-MET 完整复现执行计划

本计划对应 `notebooks/C-MET_Full_Reproduction_Colab.ipynb`。目标不是缩小版 demo，而是在公开材料允许的范围内执行 C-MET 主方法的完整数据、训练、推理和评估链路。

## 1. 开始前

你需要：

- Colab A100 40GB/80GB，高内存运行时。
- 可写的 Google Drive；默认公开流式流程下，约 180GB 可用空间可以开始，但需持续观察实际占用。
- 把 MEAD 官方 Part0 公共文件夹添加为 MyDrive 快捷方式；CREMA-D 由脚本按 benchmark 清单自动下载。
- 从第 0 格开始按顺序运行，不跳过环境或数据门禁。

官方论文使用单张 RTX 3090 24GB；A100 40GB 满足显存要求。完整流程不能在 8GB 4060 或 MPS Mac 上等价运行。

## 2. 路径约定

```text
MyDrive/C-MET-full/
  dataset/MEAD/FPS25/
  dataset/CREMA_D/FPS25/
  official_model_files/
  cache/cmet_training_cache.tar
  reproduction_runs/
  qualitative_runs/
  benchmark_runs/
  reports/

MyDrive/MEAD/                  # 官方 Part0 文件夹快捷方式，不占用你的 Drive 配额
```

默认配置：

```python
DATA_SOURCE = "public"
MEAD_SHARED_ROOT = MY_DRIVE / "MEAD"
```

若你已经自行取得完整原始数据，才改为 `DATA_SOURCE="manual"`，使用 `raw/MEAD` 和 `raw/CREMA-D`。

官方仓库固定为：

```text
0ca437cf7a8129c6a5dca1e2667a588410822bbe
```

## 3. 数据准备

### 3.1 一次性来源预检

先在浏览器打开：

```text
https://drive.google.com/drive/folders/1GwXP-KpWOxOenOxITTsURJZQ_1pkd4-j
```

选择“整理 -> 添加快捷方式”，放到 MyDrive 根目录并保持名称 `MEAD`。然后运行：

```python
RUN_MEAD_SOURCE_CHECK = True
```

预检必须一次性找到 C-MET 官方 train/test 的 47 个身份和对应 `video.tar`，但不会复制、解压或预处理。这样不会跑到第几十个身份才发现快捷方式不完整。

### 3.2 Smoke test

依次打开：

```python
RUN_MEAD_PREP_SMOKE = True
RUN_CREMAD_PREP_SMOKE = True
```

每个数据集只处理 2 条视频。公开 MEAD smoke 会保留第一个身份的本地 tar，紧接着运行 full 时可直接复用。通过标准：

- 官方 EDTalk 裁脸器能找到人脸轨迹。
- 输出 MP4 为 256x256、25 FPS。
- 输出 WAV 为 16 kHz、单声道。
- 报告中 `failed == 0`。

### 3.3 Full preprocess

Smoke 成功后打开：

```python
RUN_MEAD_PREP_FULL = True
RUN_CREMAD_PREP_FULL = True
RUN_FULL_MEDIA_GATE = True
```

默认官方裁脸模式会让裁后 MP4 和 WAV 从同一个临时时间片段导出，避免旧流程中“视频已裁短、音频仍来自完整原片”的错位。输出目录中的 `.cmet_prepare_state.json` 记录 schema v2 升级进度；Colab 断线后直接重跑同一开关，只处理尚未升级或尚未完成的条目。只有你已经人工确认旧媒体严格对齐时，才在命令行使用 `--trust-existing-media` 跳过迁移。

公开 MEAD 的执行方式：

- 每次只把一个身份的约 19GB `video.tar` 顺序复制到 `/content/cmet_public_data`。
- 只提取 `front`、官方 Common/Generic 编号，处理完成后删除该身份的临时 tar 和原片。
- 每个身份的完成状态写入 Drive；已完成的身份会检查 670 对 MP4/WAV 后再跳过，缺一条只修复该身份。
- `.part` 可以在同一 Colab 运行时内继续复制；若运行时被彻底删除，本地 `.part` 会丢失，但此前完成并写入 Drive 的身份不会重做。

公开 CREMA-D 的执行方式：

- 从官方 Git LFS 镜像只拉取 `test.csv` 使用的 2069 个唯一 `.flv`。
- 镜像固定到 `d15eeed6a139e9724483ed9a2fc4643f88708b79`。
- LFS 路径按 200 个一批下载，断线重跑只补处理后仍缺失的文件。
- full 成功后删除 `/content` 中的 Git LFS 源仓库，只保留 Drive 中处理后的 MP4/WAV。

MEAD 只处理官方 train/test 身份及 Common/Generic 编号。官方划分实际为：

```text
train: 43 个身份
test: 4 个身份
```

官方 benchmark 实际为：

```text
MEAD: 1143 条样本
CREMA-D: 1546 条样本
```

## 4. 特征抽取

按顺序执行：

```python
RUN_MEAD_E2V_SMOKE = True
RUN_CREMAD_E2V_SMOKE = True
RUN_EDTALK_SMOKE = True
```

确认两条数据成功后，再运行：

```python
RUN_MEAD_E2V_FULL = True
RUN_CREMAD_E2V_FULL = True
RUN_EDTALK_FULL = True
RUN_FEATURE_GATE = True
```

MEAD emotion2vec 特征同时用于训练和 MEAD benchmark；CREMA-D emotion2vec 特征用于保留其情绪与 `HI/LO/MD/XX` 强度。两套 full 都完成后再运行 benchmark 特征门禁。

产物合同：

```text
001.mp4
001.wav
001_ED_exp.npy       # (T, 10)
001_ED_pose.npy      # (T, 6)
001_ED_lip.npy       # (T, 20)
emotion2vec+large_features/001.npy  # (1024,)
```

`RUN_FEATURE_GATE` 会分别执行：

- `training`：检查完整媒体和三类 EDTalk/e2v 特征。
- `model`：按官方 DataLoader 实际读取的编号子集检查训练合同。

只有两份报告都为 `ready: true` 才进入训练。

emotion2vec NPY 会在 shape/有限值校验后原子替换；EDTalk 的 exp/pose/lip 三件套使用事务标记。若 Colab 在三件套替换中断线，下一次门禁会报 `incomplete_edtalk_features`，直接重跑 `RUN_EDTALK_FULL` 即可整组修复。

## 5. 训练缓存

先打开：

```python
BUILD_TRAIN_CACHE = True
```

缓存只包含训练实际读取的：

- `*_ED_exp.npy`。
- `emotion2vec+large_features/*.npy`。
- 零字节 MP4 路径标记。

官方训练用 MP4 路径建立样本索引，不解码视频，因此不需要把全部视频复制到 Colab 本地盘。

缓存先写隐藏临时 tar，成员数和可读性验证通过后才替换正式缓存；断线不会破坏上一份可用 tar。

每次新运行时打开：

```python
EXTRACT_TRAIN_CACHE = True
RUN_LOCAL_CACHE_GATE = True
```

训练从 `/content/cmet_training_data/MEAD/FPS25` 读取，checkpoint 和日志写回 Drive。

## 6. 训练

### 6.1 Dry-run

笔记本会自动生成四份 dry-run manifest：

- 主实验 smoke：2 step、1 个验证 batch、每步 checkpoint。
- 主实验：20 万 step。
- `Lrecon`：20 万 step。
- `Lrecon + Lcnt`：20 万 step。

Dry-run 使用独立目录，不覆盖正式训练状态。

### 6.2 Smoke train

首次运行：

```python
RUN_TRAIN_SMOKE = True
```

中断后：

```python
RESUME_TRAIN_SMOKE = True
```

通过标准：

- 完成 2 个优化 step。
- 至少执行 1 个验证 batch。
- 生成非空 checkpoint。

### 6.3 主实验

首次运行：

```python
RUN_MAIN_TRAIN = True
```

Colab 中断后重新挂载 Drive、克隆、安装、打补丁、解压缓存，再打开：

```python
RESUME_MAIN_TRAIN = True
```

主实验配置：

```text
max_steps = 200000
lambda_cnt = 0.1
lambda_dir = 0.05
mode = mean
num_feats = 10
direction = average
ID = same
train = bidir
balance = focal_mse
```

checkpoint 先写入同目录 `.tmp`，成功后原子替换正式文件；自动续训会忽略零字节 checkpoint。

为防止 Drive 被写满，完整实验默认：

```text
每 1000 step 保存一次
永久保留最近 3 份
永久保留 50000/100000/150000/200000 step 里程碑
```

清理只发生在新 checkpoint 原子写入成功之后，因此不会先删掉唯一可恢复点。

### 6.4 论文损失消融

```python
RUN_ABLATION_RECON_ONLY = True
RUN_ABLATION_RECON_CNT = True
```

对应论文 Table 6：

```text
ablation_recon_only: lambda_cnt=0.0, lambda_dir=0.0
ablation_recon_cnt:  lambda_cnt=0.1, lambda_dir=0.0
paper_main:          lambda_cnt=0.1, lambda_dir=0.05
```

三组都固定为 20 万 step，保存在独立目录。

## 7. 推理

完整复现选择：

```python
CHECKPOINT_SOURCE = "self_trained"
```

默认只有达到 20 万 step 的自训练 checkpoint 才允许进入正式推理。中间 checkpoint 仅用于诊断时，可显式打开：

```python
ALLOW_PARTIAL_SELF_TRAINED_CHECKPOINT = True
```

### 7.1 定性结果

```python
RUN_QUAL_BASIC = True
RUN_QUAL_EXTENDED = True
RUN_QUAL_TECH_CHECK = True
```

生成 7 种基础情绪和 6 种扩展情绪。模型每组只加载一次，JSONL 记录断点进度。

### 7.2 Benchmark

先保持默认协议：

```python
BENCHMARK_EMOTION_PROTOCOL = "dataset"
```

该协议按论文的 10 条 neutral speech 和 10 条 emotional speech 重建：MEAD 按测试身份、情绪和 `level_1/2/3` 建池，CREMA-D 按情绪和 `HI/LO/MD/XX` 建池。若要做官方公开演示资源对照，再改为：

```python
BENCHMARK_EMOTION_PROTOCOL = "official-static"
```

`official-static` 使用官方静态演示池，会忽略 `test.csv` 的逐样本强度。作者没有公开 Table 1 的批量推理脚本，因此两者都必须在报告中写明协议名；`dataset` 是依据论文与官方数据读取逻辑重建的强度感知协议，不能未经作者确认声称与作者内部脚本逐项相同。

先跑：

```python
RUN_MEAD_BENCH_SMOKE = True
RUN_CREMAD_BENCH_SMOKE = True
```

成功后：

```python
RUN_MEAD_BENCH_FULL = True
RUN_CREMAD_BENCH_FULL = True
RUN_BENCH_TECH_CHECK = True
```

每个 checkpoint 使用独立输出目录。脚本会在模型加载前预检全部所选输入、权重和情绪池；中断后读取 JSONL，跳过已完成样本。文件末尾被断线截断的一条 JSON 会被忽略，中间损坏仍会严格报错。

读取末尾断线残片时，脚本会先原子裁掉残片再追加新记录。最终 MP4 也先在隐藏路径完成音视频封装，并通过分辨率、FPS、时长和音视频流检查后才替换正式文件。

同一 checkpoint 下，两套协议也完全隔离：

```text
videos/<emotion_protocol>/<dataset>/
manifests/<emotion_protocol>/<dataset>.csv
progress/<emotion_protocol>/<dataset>.jsonl
```

## 8. 评估

### 8.1 技术检查

`evaluate_videos_basic.py` 检查：

- MP4 能否被 ffprobe 解码。
- 分辨率是否为 256x256。
- FPS 是否为 25。
- 时长是否大于 0。
- 是否含音频流。

技术检查不是论文指标。

### 8.2 部分指标汇总

```python
RUN_PARTIAL_METRIC_SUMMARY = True
```

可以汇总常驻模型后端记录的本机 AITV 和输出覆盖率，但不能称为作者论文 AITV，除非硬件、计时边界和协议一致。

### 8.3 严格论文指标

外部评估器应生成：

```text
external_paper_metrics/<checkpoint_tag>/<emotion_protocol>/mead_sample_metrics.csv
external_paper_metrics/<checkpoint_tag>/<emotion_protocol>/mead_global_metrics.json
external_paper_metrics/<checkpoint_tag>/<emotion_protocol>/crema-d_sample_metrics.csv
external_paper_metrics/<checkpoint_tag>/<emotion_protocol>/crema-d_global_metrics.json
```

外部评估器必须读取同一协议的 manifest，并把其中的 `sample_id` 原样写回，不能把 `dataset` 与 `official-static` 的结果混合。

其中：

```text
sample_metrics.csv: sample_id,sync_confidence,predicted_emotion
global_metrics.json: aitv,fid,fvd
```

然后打开：

```python
RUN_STRICT_PAPER_METRICS = True
```

脚本会检查情绪协议、清单覆盖率、重复 ID、未知 ID、五项指标完整性，并和论文目标值生成差值表。

## 9. 作者未公开造成的边界

当前无法仅靠官方仓库严格完成：

- 精确 FID/FVD/SyncNet 采样实现。
- 微调后的 MEAD/CREMA-D Emotion-FAN。
- PD-FGC + C-MET 完整分支。
- Qwen2.5-Omni 消融。
- 连续情绪编辑。
- 统一 baseline 复现。
- 用户研究原始投票和抽样清单。

这些是公开材料缺失，不是通过增加 GPU 就能解决。

## 10. 最终交付

- 环境、媒体、特征、缓存门禁报告。
- 主实验 20 万 step checkpoint 与 TensorBoard。
- 两组损失消融 20 万 step checkpoint。
- 13 类定性视频。
- MEAD 1143 条、CREMA-D 1546 条 benchmark 视频与进度。
- 技术检查 CSV/JSON。
- 获得作者评估器后生成的严格论文指标表。
