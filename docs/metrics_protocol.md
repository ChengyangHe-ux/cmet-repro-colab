# C-MET 论文级指标复现协议

基础视频检查只能说明文件有效，不能说明论文指标复现。完整复现至少需要下面几类指标。

## 1. 唇音同步

目标：证明嘴型跟音频同步。

建议指标：

- SyncNet confidence。
- LSE-C。
- LSE-D。

输入：

```text
生成视频 mp4
对应驱动音频 wav
```

输出：

```text
video, lse_c, lse_d, sync_confidence
```

## 2. 身份保持

目标：证明生成视频仍然像 source identity。

建议指标：

- ArcFace cosine similarity。

做法：

- 从 source image 提取 ArcFace embedding。
- 从生成视频均匀抽帧。
- 对每帧提取 ArcFace embedding。
- 计算 source 和生成帧 embedding 的平均 cosine similarity。

## 3. 情绪准确率

目标：证明目标情绪真的迁移到了脸上。

建议指标：

- facial emotion classifier accuracy。
- 每个目标情绪的分类概率均值。

注意：

- sarcastic、romantic 这类扩展情绪不一定有标准人脸分类器标签。
- 对扩展情绪可以做定性展示和用户研究式评分。

## 4. 视频质量

目标：评估生成质量。

可选指标：

- FID。
- FVD。
- LPIPS。

注意：

- FID/FVD 需要大量样本，不适合只用几个 demo 视频算。
- 需要和论文同样的数据 split、同样采样策略，否则不能直接说复现了论文数值。

## 5. 消融实验表

每个实验至少记录：

```text
实验名
训练命令
checkpoint
训练 step
MSE / EC / Vel / Dir / Cnt
唇音同步指标
身份保持指标
情绪准确率
代表视频路径
```

## 6. 当前仓库已经提供的指标

当前脚本：

```text
scripts/evaluate_videos_basic.py
```

它只检查：

- 分辨率。
- FPS。
- 时长。
- 是否有音频。
- 编码格式。

下一步要补的是 SyncNet、ArcFace 和情绪分类器三个脚本。
