# C-MET 数据结构说明

完整复现最容易卡在数据结构。训练脚本默认读取：

```text
./dataset/MEAD/FPS25
```

并根据官方 split 文件读取 ID：

```text
./dataset/MEAD/train.txt
./dataset/MEAD/test.txt
```

## 1. MEAD 目标结构

训练代码期望的结构是：

```text
dataset/MEAD/FPS25/
  M005/
    front/
      happy/
        level_1/
          001.mp4
          001.wav
          001_ED_exp.npy
          001_ED_pose.npy
          001_ED_lip.npy
          emotion2vec+large_features/
            001.npy
        level_2/
        level_3/
      neutral/
        level_1/
```

注意：

- 非 neutral 情绪有 `level_1`、`level_2`、`level_3`。
- neutral 只使用 `level_1`。
- 视频和音频文件名要能对应，比如 `001.mp4` 和 `001.wav`。
- EDTalk 特征放在视频同目录。
- emotion2vec+large 特征放在 `emotion2vec+large_features/`。

## 2. 官方 split

官方 MEAD 训练 ID 数量：

```text
train: 42
test: 3
```

完整复现要按官方 split 走，不要随便改 train/test，否则指标不能对齐。

## 3. 检查命令

在官方 C-MET 根目录运行：

```bash
python repro_tools/validate_full_dataset.py \
  --cmet-root . \
  --mead-root ./dataset/MEAD/FPS25 \
  --strict
```

如果 `--strict` 失败，先修数据，不要急着训练。

## 4. Google Drive 建议结构

建议把大文件放到 Drive：

```text
MyDrive/C-MET-full/
  raw/
    MEAD/
    CREMA-D/
  dataset/
    MEAD/FPS25/
    CREMA_D/FPS25/
  checkpoints/
  tensorboard_runs/
  res/
```

在 Colab 里用软链接接到官方 C-MET 目录：

```bash
ln -s /content/drive/MyDrive/C-MET-full/dataset ./dataset
ln -s /content/drive/MyDrive/C-MET-full/checkpoints ./checkpoints
ln -s /content/drive/MyDrive/C-MET-full/res ./res
```

这样断开运行时后，数据和结果不会丢。
