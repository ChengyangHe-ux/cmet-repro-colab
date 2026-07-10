# C-MET 复现 Colab

这是这篇论文的复现仓库：

**Cross-Modal Emotion Transfer for Emotion Editing in Talking Face Video**  
Choi et al., CVPR 2026

这个仓库只放复现流程和工具脚本，不存 C-MET 权重检查点、EDTalk 权重、MEAD 数据，也不直接复制官方 C-MET 源码。Colab 笔记本会在运行时克隆官方仓库。

## 快速开始

如果你只想先看官方预训练权重推理演示，打开：

[演示版 Colab](https://colab.research.google.com/github/ChengyangHe-ux/cmet-repro-colab/blob/main/notebooks/C-MET_Colab_Demo.ipynb)

如果你要做完整全流程复现，打开：

[完整复现 Colab](https://colab.research.google.com/github/ChengyangHe-ux/cmet-repro-colab/blob/main/notebooks/C-MET_Full_Reproduction_Colab.ipynb)

运行环境建议：

```text
A100 GPU > L4 GPU > T4 GPU
```

如果 Colab 有高 RAM 选项，也一起打开。

Colab 默认 Python 版本可能比作者环境新，所以笔记本不会盲目执行官方完整 `requirements.txt`。演示版安装推理依赖；完整复现版安装训练和预处理核心依赖，并在运行前修复这些常见兼容问题：可选依赖导入、`librosa`/NumPy、`moviepy`/Python 3.12、PyTorch 权重检查点加载默认值，以及 torchvision 缺少旧视频 I/O 接口。

## 当前两个目标

### 目标 1：演示版复现

先跑官方权重检查点推理，生成这两个视频：

```text
C-MET/res/ChatGPT_man3_happy.mp4
C-MET/res/ChatGPT_man3_sarcastic.mp4
```

`sarcastic` 适合汇报展示，因为论文重点之一就是扩展情绪。

### 目标 2：完整全流程复现

完整复现要覆盖：

```text
下载完整 MEAD / CREMA-D
-> 裁剪、抽音频、统一 25 FPS
-> 抽 emotion2vec+large 音频特征
-> 抽 EDTalk 表情/姿态/唇形特征
-> 从头训练 C-MET connector
-> 用自训练 checkpoint 批量推理
-> 做指标评估和消融实验
```

## 仓库结构

```text
notebooks/
  C-MET_Colab_Demo.ipynb
  C-MET_Full_Reproduction_Colab.ipynb
docs/
  full_reproduction_plan.md
  dataset_structure.md
  metrics_protocol.md
  reproduction_plan.md
  troubleshooting.md
  evaluation.md
experiments/logs/
  experiment_template.md
scripts/
  check_env.py
  patch_cmet_colab_full.py
  validate_full_dataset.py
  extract_edtalk_features.py
  batch_inference.py
  evaluate_videos_basic.py
```

## 为什么不直接放官方 C-MET 源码

官方 C-MET 仓库已经提供推理和训练代码。这个仓库负责把 Colab 全流程串起来：

```text
本仓库 -> Colab 笔记本 -> 克隆官方 C-MET -> 打补丁 -> 数据检查 -> 特征抽取 -> 训练 -> 推理 -> 评估
```

这样 GitHub 仓库更干净，也能避免误传模型权重或生成视频。

## 面向汇报的里程碑

- [ ] 讲清楚论文 Figure 3。
- [ ] 跑通官方 happy 演示。
- [ ] 跑通官方 sarcastic 演示。
- [ ] 运行 Colab 笔记本里的评估单元，并保存帧截图表。
- [ ] 记录硬件和依赖问题。
- [ ] 说明为什么完整训练需要更强显存，最好 24GB 以上。
- [ ] 准备完整 MEAD / CREMA-D 数据。
- [ ] 跑完整复现 Colab。
- [ ] 保存训练曲线、checkpoint、批量生成结果和指标 CSV。

## 链接

- 官方项目页：https://chanhyeok-choi.github.io/C-MET/
- 官方代码：https://github.com/ChanHyeok-Choi/C-MET
- 论文：https://arxiv.org/abs/2604.07786
- Hugging Face 模型：https://huggingface.co/coldhyuk/C-MET
