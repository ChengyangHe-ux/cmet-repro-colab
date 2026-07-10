# C-MET 完整全流程复现计划

这份计划针对“不缩水”的复现目标：不是只跑官方演示，而是尽量复现论文从数据处理、训练、推理到评估的完整链路。

## 0. 当前状态

已经完成：

- 官方预训练权重推理演示。
- happy 和 sarcastic 两个生成视频。
- Colab 兼容补丁。
- 基础视频技术检查。

还没完成：

- 完整 MEAD / CREMA-D 数据预处理。
- 从头训练 C-MET connector。
- 用自训练权重批量推理。
- 论文级定量指标。
- 消融实验。
- 对比基线。

## 1. 硬件和存储

推荐配置：

- GPU：A100 40GB/80GB 优先，4090 24GB 可以尝试，T4/L4 不建议跑完整训练。
- 内存：64GB 以上更稳。
- 存储：至少 1TB，最好 2TB。MEAD、CREMA-D、裁剪视频、音频、特征、checkpoint 会占很多空间。
- 运行位置：Colab Pro/Pro+ 或自己的 CUDA 服务器。

Mac 的作用：

- 读代码、写报告、整理仓库、做轻量检查。
- 不建议用 Mac 跑官方训练，因为官方代码有大量 `.cuda()` 路径。

## 2. 完整复现阶段

### 阶段 A：准备数据

目标：拿到完整 MEAD 和 CREMA-D。

需要做：

- 下载 MEAD。
- 下载 CREMA-D。
- 记录下载链接、版本、目录结构。
- 保留原始数据，不要直接在原始目录上改。

产物：

```text
Google Drive/C-MET-full/raw/MEAD/
Google Drive/C-MET-full/raw/CREMA-D/
```

### 阶段 B：裁剪和统一帧率

目标：把视频处理成官方训练需要的 talking-face 视频。

要求：

- front view。
- 25 FPS。
- 人脸裁剪稳定。
- 每条视频有对应 `.wav`。

产物示例：

```text
dataset/MEAD/FPS25/M005/front/happy/level_3/001.mp4
dataset/MEAD/FPS25/M005/front/happy/level_3/001.wav
```

### 阶段 C：抽 emotion2vec+large 音频特征

目标：为每条 `.wav` 生成语音情绪向量。

命令：

```bash
python extract_e2v+L.py --data_root ./dataset/MEAD/FPS25
```

产物：

```text
emotion2vec+large_features/001.npy
```

### 阶段 D：抽 EDTalk 表情、姿态、唇形特征

目标：为每条视频生成训练 C-MET 需要的视觉表达特征。

仓库脚本：

```bash
python repro_tools/extract_edtalk_features.py \
  --cmet-root . \
  --data-root ./dataset/MEAD/FPS25 \
  --batch-size 100
```

产物：

```text
001_ED_exp.npy
001_ED_pose.npy
001_ED_lip.npy
```

### 阶段 E：检查数据是否可训练

命令：

```bash
python repro_tools/validate_full_dataset.py \
  --cmet-root . \
  --mead-root ./dataset/MEAD/FPS25 \
  --strict
```

通过标准：

- train/test split 的 ID 都存在。
- `.mp4`、`.wav`、`emotion2vec+large_features/*.npy`、`*_ED_exp.npy`、`*_ED_pose.npy`、`*_ED_lip.npy` 都存在。

### 阶段 F：从头训练 C-MET connector

官方入口：

```bash
python train.py --config ./configs/train.yaml
```

建议第一轮完整训练使用官方默认：

```bash
python train.py \
  --config ./configs/train.yaml \
  --mode mean \
  --num_feats 10 \
  --direction average \
  --ID same \
  --feature_type ED \
  --audio_encoder emotion2vec+large \
  --train bidir \
  --lambda_cnt 0.1 \
  --lambda_dir 0.05 \
  --balance focal_mse
```

训练时记录：

- GPU 型号。
- batch size。
- 总 step。
- checkpoint 路径。
- TensorBoard 曲线。
- `running_MSE_loss`、`running_Cnt_loss`、`running_Dir_loss`。
- `eval_MSE_loss`、`eval_EC_loss`、`eval_Vel_loss`、`eval_Dir_loss`、`eval_Cnt_loss`。

### 阶段 G：用自训练 checkpoint 批量推理

命令：

```bash
python repro_tools/batch_inference.py \
  --cmet-root . \
  --checkpoint ./checkpoints/你的训练目录/xxx_checkpoint_step000xxxxxx.pth \
  --out-dir ./res/full_repro
```

产物：

```text
res/full_repro/ChatGPT_man3_happy.mp4
res/full_repro/ChatGPT_man3_sarcastic.mp4
...
```

### 阶段 H：评估

基础技术检查：

```bash
python repro_tools/evaluate_videos_basic.py \
  --video-dir ./res/full_repro \
  --out-csv ./res/full_repro/video_basic_metrics.csv
```

论文级指标需要继续补：

- 唇音同步：SyncNet / LSE-C / LSE-D。
- 身份保持：ArcFace cosine similarity。
- 情绪准确率：人脸情绪分类器。
- 视频质量：FID / FVD / LPIPS。

### 阶段 I：消融实验

至少跑这些组：

```text
默认完整模型
lambda_cnt=0
lambda_dir=0
train=unidir
ID=diff
direction=raw
direction=first
direction=max
except_emotions happy
except_emotions sad
```

每组都要保存：

- 训练命令。
- checkpoint。
- TensorBoard 曲线。
- 生成视频。
- 指标 CSV。

## 3. 仓库里的执行入口

完整 Colab：

```text
notebooks/C-MET_Full_Reproduction_Colab.ipynb
```

工具脚本：

```text
scripts/patch_cmet_colab_full.py
scripts/validate_full_dataset.py
scripts/extract_edtalk_features.py
scripts/batch_inference.py
scripts/evaluate_videos_basic.py
```

## 4. 最终汇报材料

完整复现最终应该能交付：

- 复现环境表。
- 数据处理流程图。
- 训练曲线。
- 自训练 checkpoint 的生成视频。
- 论文指标复现表。
- 消融实验表。
- 失败案例分析。
- 和官方预训练权重 demo 的对比。
