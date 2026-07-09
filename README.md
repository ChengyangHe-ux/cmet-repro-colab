# C-MET 复现 Colab

这是这篇论文的轻量级复现仓库：

**Cross-Modal Emotion Transfer for Emotion Editing in Talking Face Video**  
Choi et al., CVPR 2026

这个仓库只放复现流程，不存 C-MET 权重检查点、EDTalk 权重、MEAD 数据，也不直接复制官方 C-MET 源码。Colab 笔记本会在运行时克隆官方仓库，然后跑官方推理演示。

## 快速开始

打开 Colab 笔记本：

[在 Colab 中打开](https://colab.research.google.com/github/ChengyangHe-ux/cmet-repro-colab/blob/main/notebooks/C-MET_Colab_Demo.ipynb)

运行环境建议：

```text
A100 GPU > L4 GPU > T4 GPU
```

如果 Colab 有高 RAM 选项，也一起打开。

Colab 笔记本不直接安装官方完整 `requirements.txt`，因为 Colab 默认 Python 版本可能比作者环境新。我们只安装推理需要的最小依赖，并在推理前修复这些常见兼容问题：可选依赖导入、`librosa`/NumPy、`moviepy`/Python 3.12、PyTorch 权重检查点加载默认值，以及 torchvision 缺少旧视频 I/O 接口。

## 第一目标

先不要训练。第一步只跑官方权重检查点推理，生成这两个视频：

```text
C-MET/res/ChatGPT_man3_happy.mp4
C-MET/res/ChatGPT_man3_sarcastic.mp4
```

`sarcastic` 适合汇报展示，因为论文重点之一就是扩展情绪。

## 仓库结构

```text
notebooks/
  C-MET_Colab_Demo.ipynb
docs/
  reproduction_plan.md
  troubleshooting.md
  evaluation.md
experiments/logs/
  experiment_template.md
scripts/
  check_env.py
```

## 为什么不直接放官方 C-MET 源码

官方 C-MET 仓库已经提供推理和训练代码。为了快速完成汇报版复现，我们用这个流程：

```text
本仓库 -> Colab 笔记本 -> 克隆官方 C-MET -> 安装依赖 -> 跑演示
```

这样 GitHub 仓库更干净，也能避免误传模型权重或生成视频。

## 面向汇报的里程碑

- [ ] 讲清楚论文 Figure 3。
- [ ] 跑通官方 happy 演示。
- [ ] 跑通官方 sarcastic 演示。
- [ ] 运行 Colab 笔记本里的评估单元，并保存帧截图表。
- [ ] 记录硬件和依赖问题。
- [ ] 说明为什么完整训练需要更强显存，最好 24GB 以上。

## 链接

- 官方项目页：https://chanhyeok-choi.github.io/C-MET/
- 官方代码：https://github.com/ChanHyeok-Choi/C-MET
- 论文：https://arxiv.org/abs/2604.07786
- Hugging Face 模型：https://huggingface.co/coldhyuk/C-MET
