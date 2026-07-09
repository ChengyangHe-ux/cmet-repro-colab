# C-MET Reproduction Colab

Lightweight reproduction workspace for:

**Cross-Modal Emotion Transfer for Emotion Editing in Talking Face Video**  
Choi et al., CVPR 2026

This repository is intentionally small. It does not store C-MET checkpoints, EDTalk weights, MEAD data, or the official C-MET source. The Colab notebook clones the official repository at runtime and runs the official inference demo.

## Quick Start

Open the notebook:

[Open in Colab](https://colab.research.google.com/github/ChengyangHe-ux/cmet-repro-colab/blob/main/notebooks/C-MET_Colab_Demo.ipynb)

Runtime recommendation:

```text
A100 GPU > L4 GPU > T4 GPU
```

Turn on high RAM if available.

The notebook intentionally uses a minimal inference dependency set instead of the full official `requirements.txt`, because Colab's default Python may be newer than the version used by the paper authors.

## First Goal

Before training anything, run official checkpoint inference and generate:

```text
C-MET/res/ChatGPT_man3_happy.mp4
C-MET/res/ChatGPT_man3_sarcastic.mp4
```

The sarcastic example is useful for presentation because the paper emphasizes extended emotions.

## Repository Layout

```text
notebooks/
  C-MET_Colab_Demo.ipynb
docs/
  reproduction_plan.md
  troubleshooting.md
experiments/logs/
  experiment_template.md
scripts/
  check_env.py
```

## Why This Repo Does Not Vendor C-MET

The official C-MET repository already provides inference and training code. For a fast report-oriented reproduction, the cleanest workflow is:

```text
this repo -> Colab notebook -> clone official C-MET -> install dependencies -> run demo
```

This keeps GitHub small and avoids accidentally committing model weights or generated videos.

## Presentation-Oriented Milestones

- [ ] Explain Figure 3 in the paper.
- [ ] Run the official happy demo.
- [ ] Run the official sarcastic demo.
- [ ] Record hardware and dependency issues.
- [ ] Summarize why full training needs stronger GPU memory, ideally around 24GB or more.

## Links

- Official project: https://chanhyeok-choi.github.io/C-MET/
- Official code: https://github.com/ChanHyeok-Choi/C-MET
- Paper: https://arxiv.org/abs/2604.07786
- Hugging Face model: https://huggingface.co/coldhyuk/C-MET
