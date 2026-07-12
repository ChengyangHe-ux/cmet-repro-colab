# C-MET 中文全流程复现

论文：**Cross-Modal Emotion Transfer for Emotion Editing in Talking Face Video**，CVPR 2026。

本仓库提供中文 Colab 复现入口和可靠性工具，不复制官方源码、不上传受许可限制的数据，也不存放大模型权重。Colab 会固定克隆官方 C-MET commit：

```text
0ca437cf7a8129c6a5dca1e2667a588410822bbe
```

## 直接开始

- [完整复现 Colab](https://colab.research.google.com/github/ChengyangHe-ux/cmet-repro-colab/blob/main/notebooks/C-MET_Full_Reproduction_Colab.ipynb)
- [官方权重演示 Colab](https://colab.research.google.com/github/ChengyangHe-ux/cmet-repro-colab/blob/main/notebooks/C-MET_Colab_Demo.ipynb)

完整复现默认使用公开数据入口。你只需手工完成一次 MEAD Drive 授权：

1. 打开 [MEAD 官方公共文件夹](https://drive.google.com/drive/folders/1GwXP-KpWOxOenOxITTsURJZQ_1pkd4-j)，将你有权访问的视频 Part 添加为 Drive 快捷方式。
2. 在“我的云端硬盘”建立 `MEAD` 聚合目录：可直接包含身份目录，也可保留 `Part0/身份`、`Part1/身份` 结构。
3. 在完整 Notebook 中保持 `DATA_SOURCE = "public"`，打开 `RUN_MEAD_SOURCE_CHECK` 做 47 个官方身份的一次性来源预检。预检会一次列出所有缺失身份。
4. 只有来源预检无报错后，再依次运行 MEAD/CREMA-D smoke、full、媒体门禁，之后再做特征与训练。

> 官方公共目录可能按 Part 分散数据。某个身份目录存在但没有 `video.tar` / `video_*.tar` 时，必须补充包含该身份视频分卷的 Part；重跑脚本不会自动获得缺失数据。

CREMA-D 不需要手工上传。脚本从[官方 Git LFS 镜像](https://gitlab.com/cs-cooper-lab/crema-d-mirror)分批下载 C-MET `test.csv` 实际引用的 2069 个唯一视频。
镜像固定到 commit `d15eeed6a139e9724483ed9a2fc4643f88708b79`。

完整笔记本共有 26 个阶段，按顺序覆盖：

```text
GPU 与真实 Drive 门禁
-> 固定官方代码与依赖环境
-> MEAD / CREMA-D 原始数据预处理
-> emotion2vec+large 与 EDTalk 特征抽取
-> 完整数据和模型读取合同检查
-> Drive 训练缓存与 Colab 本地解压
-> 2 step 训练/验证/checkpoint 冒烟测试
-> 20 万 step 主实验与自动续训
-> 两组论文损失消融
-> 13 类基础/扩展情绪定性推理
-> MEAD 1143 条与 CREMA-D 1546 条 benchmark
-> 视频技术检查与论文指标汇总接口
```

所有长任务默认关闭。每次只把当前步骤的 `RUN_*` 开关改为 `True`，跑完后再进入下一步。

## 算力与存储

- 论文训练硬件：单张 RTX 3090 24GB。
- 推荐 Colab：A100 40GB/80GB，并打开高内存运行时。
- 4060 8GB：适合读代码和小规模排错，不适合官方 batch size 的完整训练。
- 32GB M5 Mac：适合论文学习、代码阅读和轻量检查；官方 CUDA 主流程不能等价训练。
- 默认公开流式流程不会把约 900GB 的 MEAD 原始 tar 复制进你的 Drive；每次只在 Colab 本地盘暂存一个身份的 `video.tar` 或 `video_*.tar` 分卷。
- 约 180GB 可用 Drive 是合理起点，但最终占用取决于视频压缩率与生成结果数量；每个大阶段后都要查看 Notebook 打印的剩余空间。
- 若要永久保存完整 MEAD/CREMA-D 原始仓库、所有中间 checkpoint 和全部评估缓存，才建议准备 1TB 以上空间。

训练不会直接从 Drive 读取数万个小文件。笔记本先构建精简训练缓存，再解压到 `/content` 本地盘；checkpoint、TensorBoard、进度和报告仍写回 Drive。

主实验和两组消融仍每 1000 step 保存一次用于断线恢复，但会自动只保留最近 3 份、每 5 万 step 里程碑和最终 checkpoint，避免三组 20 万步训练写满 Drive。

## Colab 终端推荐跑法

如果 Notebook 页面里临时单元格被改乱，优先用 Colab 终端跑脚本。先确认已经挂载 Google Drive，然后使用续跑控制器：

```bash
cd /content/cmet-repro-colab
git pull --ff-only
bash scripts/colab_resume_data_pipeline.sh
```

该命令会读取 `MyDrive/C-MET-full` 的状态文件：首次运行执行来源预检和小样本 smoke；smoke 通过后，再运行同一条命令会自动进入完整预处理；完整数据已完成时会提示进入特征抽取阶段。查看状态可执行：

```bash
bash scripts/colab_resume_data_pipeline.sh status
```

若希望 smoke 成功后不中断、直接继续完整预处理：

```bash
CMET_CONTINUE_FULL=1 bash scripts/colab_resume_data_pipeline.sh
```

底层仍可分别调用 `colab_run_stage1_data_smoke.sh` 和 `colab_run_stage2_full_preprocess.sh`。这些脚本会自动固定官方 C-MET commit、检查 `MyDrive/MEAD` 聚合目录、支持顶层身份与 `Part*/身份`、支持 `video.tar` 和 `video_*.tar` 分卷，并把状态报告写到 `MyDrive/C-MET-full/reports`。

## 已实现的主方法链路

- 数据发现、官方 EDTalk 裁脸、256x256、25 FPS、16 kHz 单声道 WAV。
- MEAD 官方公共目录全身份预检、逐身份流式复制/解包、身份级 Drive 状态；CREMA-D 按清单分批 Git LFS 下载并只修复缺失文件。
- 裁脸后的 MP4 与 WAV 从同一个时间片段导出；预处理 schema v2 会自动重做旧版可能音画错位的产物，并用状态文件断点继续。
- MEAD 官方 train 43 个身份、test 4 个身份的数据门禁。
- emotion2vec+large 与 EDTalk 表情/姿态/唇形特征抽取、shape 和 NaN/Inf 检查；NPY 先写临时文件，EDTalk 三件套用事务标记防止断线混入新旧特征。
- 官方训练路径修复、20 万 step 上限、独立实验目录、checkpoint、自动续训和 TensorBoard。
- checkpoint 临时文件写入后原子替换，避免断线留下半个 `.pth`。
- 主实验以及论文 Table 6 的 `Lrecon`、`Lrecon + Lcnt` 两组消融。
- 模型常驻内存的定性与 benchmark 推理、稳定 sample ID、可自动修复末尾断线残片的 JSONL 进度。
- 训练缓存、训练 manifest、checkpoint、benchmark manifest 和最终 MP4 均先生成临时文件，验证成功后再替换正式产物。
- benchmark 在加载模型前检查所选视频、音频、权重和情绪特征池。
- 生成视频分辨率、FPS、时长、音频流等严格技术检查。
- AITV、FID、FVD、SyncNet confidence、Emotion-FAN 准确率的覆盖率和论文目标值汇总接口。

## Benchmark 情绪协议

笔记本默认使用：

```python
BENCHMARK_EMOTION_PROTOCOL = "dataset"
```

- `dataset`：依据论文的 10 条 neutral speech 与 10 条 emotional speech、官方 `Dataset.get_raw_e2v` 和 `test.csv` 重建。MEAD 按测试身份、目标情绪和 `level_1/2/3` 取特征；CREMA-D 按目标情绪和 `HI/LO/MD/XX` 取特征，因此保留语音强度信息。
- `official-static`：复用官方仓库公开的静态演示语音池。基础情绪池来自官方发布的演示资源，不能表达 `test.csv` 中每条样本的强度，只适合作为官方演示协议对照。

作者没有公开 Table 1 的批量推理脚本，因此 `dataset` 是论文依据下的可审计重建，不应未经作者确认就写成“精确作者 benchmark 协议”。两套协议的 sample ID、视频、manifest、progress 和报告均按协议目录隔离，不能混合汇总。

## 必须诚实区分的范围

仓库可以完整执行 C-MET 主方法的数据、训练、推理和结果汇总链路，但官方没有公开以下精确材料：

- FID/FVD/SyncNet 的完整采样与实现细节。
- 分别在 MEAD、CREMA-D 上微调的 Emotion-FAN checkpoint。
- EAT、EAMM、EDTalk、FLOAT 的统一 baseline 环境。
- PD-FGC 完整源码、权重和推理入口。
- Qwen2.5-Omni 音频编码器消融代码。
- 连续情绪编辑脚本、用户研究原始数据与抽样清单。

因此，只有在获得作者对应实现和权重后，才能把五项数值称为“严格论文指标复现”。仓库不会用基础视频检查或自定义分类器冒充论文结果。

## 运行产物

默认保存在：

```text
MyDrive/C-MET-full/
  raw/                         # 仅 manual 模式使用的原始数据
  dataset/                     # 处理后的 MEAD / CREMA-D
  official_model_files/        # 官方 EDTalk 与 C-MET 权重
  cache/                       # 精简训练缓存
  reproduction_runs/           # manifest、checkpoint、TensorBoard
  qualitative_runs/            # 13 类定性视频
  benchmark_runs/              # 按 checkpoint 和协议隔离的视频、manifest、JSONL 进度
  reports/                     # 数据门禁、技术检查、指标表
```

推理结果按 checkpoint 文件名隔离，避免中间 checkpoint 的旧视频被 20 万 step checkpoint 误判为已完成。

benchmark 目录进一步按协议隔离：

```text
benchmark_runs/<checkpoint_source>/<checkpoint_tag>/
  videos/<emotion_protocol>/<dataset>/
  manifests/<emotion_protocol>/<dataset>.csv
  progress/<emotion_protocol>/<dataset>.jsonl
```

## 仓库结构

```text
notebooks/
  C-MET_Full_Reproduction_Colab.ipynb
  C-MET_Colab_Demo.ipynb
configs/
  colab_requirements.txt
  experiments.json
  paper_targets.json
scripts/
  colab_resume_data_pipeline.sh
  data_pipeline_status.py
  install_colab_dependencies.py
  download_pretrained_weights.py
  verify_colab_environment.py
  prepare_public_datasets.py
  prepare_datasets.py
  extract_emotion2vec_features.py
  extract_edtalk_features.py
  validate_full_dataset.py
  build_training_cache.py
  run_training.py
  cmet_inference_runtime.py
  run_qualitative_inference.py
  run_benchmark_inference.py
  evaluate_videos_basic.py
  evaluate_paper_metrics.py
docs/
  full_reproduction_plan.md
  dataset_structure.md
  evaluation.md
  metrics_protocol.md
  troubleshooting.md
tests/
```

## 本地检查

```bash
PYTHONPATH=scripts python3 -m unittest discover -s tests -v
PYTHONPYCACHEPREFIX=/tmp/cmet-pycache python3 -m compileall -q scripts tests
git diff --check
```

这些是离线结构和逻辑检查，不代表本机已经替你完成 A100 全量训练。A100、真实 MEAD/CREMA-D 和论文外部评估器仍需在 Colab 中执行。

## 资料

- [官方项目页](https://chanhyeok-choi.github.io/C-MET/)
- [官方代码](https://github.com/ChanHyeok-Choi/C-MET)
- [论文](https://arxiv.org/abs/2604.07786)
- [官方模型](https://huggingface.co/coldhyuk/C-MET)
