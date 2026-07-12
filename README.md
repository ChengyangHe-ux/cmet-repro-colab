# C-MET Mini Reproduction Colab

这是一个与完整复现工程分离的 **小规模 C-MET 推理复现项目**。默认目标是在 Colab GPU 上用几小时完成一条可展示、可检查的闭环，而不是训练论文中的 20 万 step 模型。

## 默认复现内容

```text
固定官方 C-MET commit
→ 安装 Colab 依赖并打兼容补丁
→ 下载官方 Audio2Lip / EDTalk / C-MET connector 权重
→ 使用官方仓库自带的身份图、语音、姿态视频与 emotion2vec 特征池
→ 生成 happy / sad / angry 三种情绪视频
→ 用 ffprobe 检查音视频流和时长
→ 保存 progress.jsonl 与 summary.json
```

默认不需要完整 MEAD，也不需要 `MyDrive/MEAD` 快捷方式。

## 时间与资源

- 推荐 GPU：A100、L4 或显存不低于 16GB 的运行时。
- 首次安装和权重下载：约 20～60 分钟。
- 三种情绪推理：通常约 30～120 分钟。
- 现实总时间：约 1～3 小时，网络或 Colab 负载较慢时可能更久。

## 复现边界

该项目验证的是官方模型的 **推理链路与情绪编辑效果**，不是论文级训练复现。它不会：

- 完整预处理 MEAD / CREMA-D；
- 重新训练 20 万 step；
- 复现作者未公开的精确 FID/FVD/SyncNet/Emotion-FAN 协议。

## Colab

打开 `notebooks/C-MET_Mini_Reproduction_Colab.ipynb`，只运行顶部的一键单元。默认参数无需修改。

结果保存到：

```text
MyDrive/C-MET-mini/results/<run-name>/
```

断线后重新运行同一个单元即可；已经生成且通过检查的视频会自动跳过。
