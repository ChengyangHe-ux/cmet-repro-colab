# C-MET Scientific Mini Reproduction Colab

这是一个与完整复现工程分离的 **科学小规模 C-MET 复现项目**。目标是在 Colab GPU 上用几小时完成一条可展示、可检查、可比较、可审计的闭环，而不是只证明代码“能跑”。

## 为什么不是简单 demo

默认 `scientific` 配置会同时完成：

```text
固定官方 C-MET commit
→ 记录 Python / Torch / CUDA / GPU / 依赖版本
→ 记录官方 checkpoint 与输入素材 SHA256
→ 使用官方示例身份图、语音、姿态视频和情绪特征池
→ 生成 happy / sad / angry 主实验
→ 生成零情绪方向控制组
→ 运行特征样本数消融
→ 运行随机种子敏感性实验
→ 检查每个 MP4 的音频流、视频流和时长
→ 计算方向范数、视频时序 MAD、相对控制组像素 MAD
→ 自动生成 JSON 记录和 Markdown 科学报告
```

完整规格见 [`REPRODUCTION_SPEC.md`](REPRODUCTION_SPEC.md)。

## 默认实验矩阵

| 实验 | 目的 |
|---|---|
| `main_happy`、`main_sad`、`main_angry` | 验证官方模型对多个目标情绪的主链路 |
| `baseline_zero_happy` | 将情绪方向缩放为 0，作为控制组 |
| `ablation_single_happy` | 把 3 个情绪特征样本改成 1 个，观察样本数影响 |
| `sensitivity_seed_123_happy` | 改变随机抽样种子，观察结果稳定性 |

默认共生成 6 个视频。

## 运行时间

- `PROFILE="demo"`：只跑主情绪，通常约 1～3 小时。
- `PROFILE="scientific"`：加入控制、消融和敏感性实验，通常约 2～6 小时。
- 首次依赖和权重下载约 20～60 分钟。
- 网络、Colab GPU 和 Drive I/O 会影响实际时间。

推荐 A100、L4 或显存不低于 16GB 的 GPU。

## Colab 使用

打开：

```text
notebooks/C-MET_Mini_Reproduction_Colab.ipynb
```

只运行顶部的一键单元。默认参数：

```python
PROFILE = "scientific"
EMOTIONS = "happy,sad,angry"
NUM_SAMPLES = 3
SEED = 42
SENSITIVITY_SEED = 123
RUN_NAME = "scientific_mini"
```

结果保存到：

```text
MyDrive/C-MET-mini/results/<run-name>/
```

断线后重新运行同一单元即可；已经生成且通过检查的视频会自动跳过。

## 输出文件

```text
results/<run-name>/
├── main_happy.mp4
├── main_sad.mp4
├── main_angry.mp4
├── baseline_zero_happy.mp4
├── ablation_single_happy.mp4
├── sensitivity_seed_123_happy.mp4
├── environment.json
├── protocol.json
├── progress.jsonl
├── summary.json
├── analysis.json
├── human_evaluation_template.csv
└── report.md
```

其中：

- `environment.json` 记录环境版本、GPU、官方 commit、checkpoint 哈希和输入素材哈希；
- `protocol.json` 记录研究问题、实验矩阵和限制；
- `analysis.json` 记录方向范数、视频诊断和相对控制组差异；
- `human_evaluation_template.csv` 用于人工盲评情绪、身份保持、唇形同步和伪影；
- `report.md` 自动整理研究问题、假设、实验表格、结论判定规则和复现边界。

## 复现边界

这是官方 checkpoint 的 **推理级科学小规模复现**，不是论文级训练复现。它不会：

- 完整预处理 MEAD / CREMA-D；
- 重新训练 20 万 step；
- 声称复现作者未公开的精确 FID/FVD/SyncNet/Emotion-FAN 协议；
- 把像素 MAD 当作情绪识别准确率。

像素 MAD 只能证明不同条件下输出发生了变化。情绪是否准确，仍需要人工盲评或独立情绪分类器进一步验证。

## 代码结构

```text
Notebook 一键入口
└── scripts/run_colab_mini.py
    ├── 固定官方仓库、补丁和权重
    └── scripts/run_scientific_mini.py
        ├── 构建主实验、控制组、消融和敏感性实验
        ├── 复用一次模型加载生成多个视频
        ├── 记录环境与实验 manifest
        └── scripts/analyze_mini_results.py
            ├── 视频技术检查
            ├── 时序与像素差异分析
            └── 自动生成 report.md
```

## 当前托管方式

独立项目发布在 `ChengyangHe-ux/cmet-repro-colab` 的 `mini-repro` 隔离分支。该分支是独立根提交，与完整复现 `main` 不共享项目文件。
