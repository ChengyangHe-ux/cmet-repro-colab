# C-MET 数据结构与官方合同

## 0. 公开来源与临时目录

默认 Notebook 不把完整原始数据复制到用户 Drive：

```text
MyDrive/MEAD/                         # 官方 MEAD 快捷方式聚合根目录
  M003/video.tar                     # 支持：顶层身份目录
  Part1/W021/video_1.tar             # 支持：Part*/身份嵌套目录
/content/cmet_public_data/MEAD/       # 当前 MEAD 身份临时 tar 与原片
/content/cmet_public_data/CREMA-D/    # CREMA-D Git LFS 临时仓库
MyDrive/C-MET-full/dataset/           # 最终处理数据与特征
```

MEAD 先一次扫描聚合根目录，对官方 47 个身份做来源预检，再逐身份处理。同一身份可以分散在多个 Part，同名 tar 复制到 Colab 时会自动编号避免覆盖。每个完整身份应有 670 对 MP4/WAV：neutral 40 条，加 7 个非中性情绪 x 3 个 level x 30 条。身份完成状态记录在：

```text
MyDrive/C-MET-full/reports/mead_public_stream_state.json
```

CREMA-D 从官方 Git LFS 镜像读取，按 `test.csv` 去重后下载 2069 个 `.flv`，处理后保持扁平目录。full 成功后临时 Git 仓库可以删除。

## 1. MEAD 训练目录

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

媒体要求：

- 256x256。
- 25 FPS。
- WAV 为 16 kHz、单声道。
- 非 neutral 使用 `level_1`、`level_2`、`level_3`。
- neutral 只使用 `level_1`。

特征要求：

```text
ED_exp:  (T, 10)
ED_pose: (T, 6)
ED_lip:  (T, 20)
emotion2vec+large: (1024,)
```

## 2. 官方划分

官方固定 commit 的真实列表长度：

```text
dataset/MEAD/train.txt: 43 个身份
dataset/MEAD/test.txt: 4 个身份
```

旧统计 `42/3` 来自文件末尾没有换行时使用 `wc -l`，不是实际身份数。

## 3. 官方训练实际读取编号

官方 DataLoader 用 MP4 路径建立样本索引：

```text
neutral: 1-3、31-40
其他情绪: 1-3、21-30
```

`mode=mean` 的 EDTalk 表情特征池：

```text
neutral: 1-10、31-40
其他情绪: 1-10、21-30
```

emotion2vec 池：

```text
neutral: 1-40
其他情绪: 1-30
```

因此仓库把数据检查分为：

- `training`：完整媒体和全部三类 EDTalk/e2v 特征。
- `model`：官方训练真实读取的最小合同。

## 4. CREMA-D 目录

官方测试清单使用扁平目录：

```text
dataset/CREMA_D/FPS25/
  1001_DFA_NEU_XX.mp4
  1001_DFA_NEU_XX.wav
  1001_DFA_HAP_XX.mp4
  1001_DFA_HAP_XX.wav
  emotion2vec+large_features/
    1001_DFA_NEU_XX.npy
    1001_DFA_HAP_XX.npy
```

预处理脚本保留原始 actor、sentence、emotion、intensity 文件名。

CREMA-D emotion2vec 文件 shape 为 `(1024,)`。benchmark 的 `dataset` 协议会按文件名中的情绪和 `HI/LO/MD/XX` 强度分组，因此不能在预处理后改名。

## 5. Benchmark 数量

CSV 第一行是表头，实际样本数为：

```text
MEAD test.csv: 1143
CREMA_D test.csv: 1546
```

## 6. Drive 与本地训练缓存

完整处理数据保存在 Drive：

```text
MyDrive/C-MET-full/dataset/MEAD/FPS25
```

训练缓存解压为：

```text
/content/cmet_training_data/MEAD/FPS25
```

缓存中的 MP4 是零字节路径标记，因为官方 connector 训练只用路径筛选样本，不解码视频。不要对本地缓存运行媒体检查。

## 7. 预处理版本与断点状态

使用官方裁脸模式时，处理后数据根目录包含：

```text
.cmet_prepare_state.json
```

当前 schema 为 v2。它保证裁后 MP4 和 WAV 从同一个裁后片段导出，并记录旧版数据升级进度。零字节媒体不算完成；每个 MP4/WAV 先写临时文件，成功后再原子替换。断线后重跑同一 full 开关即可继续，不要删除状态文件或从头覆盖全部数据。

公开流式流程在全局 schema 状态之外还检查每个身份/文件的实际 MP4、WAV 和事务标记。即使状态写着完成，只要媒体缺失、为空或存在 `.media_pair.in_progress`，重跑时仍会自动修复。

成对替换期间会创建隐藏事务标记 `.编号.media_pair.in_progress`。如果进程在两次替换之间断线，门禁会报 `incomplete_media_pair`，下一次预处理会强制从同一裁后片段重建 MP4/WAV。

EDTalk 三件套替换期间会创建 `.编号_ED_features.in_progress`。标记存在时，即使三个 NPY 都能读取，也不会被视为完整，必须重跑 EDTalk 特征抽取。

只有在确认现有媒体已经严格音画对齐时，才可显式使用 `--trust-existing-media`；否则保留默认迁移行为。

## 8. Benchmark 情绪特征合同

默认 `dataset` 协议要求：

- MEAD：官方 test 身份的 neutral `level_1`，以及目标情绪对应的 `level_1/2/3`，每个池至少 10 个有效 `(1024,)` 特征。
- CREMA-D：neutral `XX`，以及目标情绪对应的 `HI/LO/MD/XX`，每个实际使用的池至少 10 个有效 `(1024,)` 特征。

`official-static` 协议改用官方 `audios/MEAD/<emotion>/emotion2vec+large_features` 演示池，不读取 benchmark 数据集自身的强度特征。

## 9. 检查命令

完整数据：

```bash
python scripts/validate_full_dataset.py \
  --cmet-root /content/C-MET \
  --mead-root /content/drive/MyDrive/C-MET-full/dataset/MEAD/FPS25 \
  --scope train \
  --split both \
  --stage training \
  --check-arrays \
  --strict
```

两套 benchmark 的媒体和 emotion2vec：

```bash
python scripts/validate_full_dataset.py \
  --cmet-root /content/C-MET \
  --mead-root /content/drive/MyDrive/C-MET-full/dataset/MEAD/FPS25 \
  --cremad-root /content/drive/MyDrive/C-MET-full/dataset/CREMA_D/FPS25 \
  --scope benchmark \
  --benchmark-dataset both \
  --stage features \
  --check-arrays \
  --strict
```

训练缓存：

```bash
python scripts/validate_full_dataset.py \
  --cmet-root /content/C-MET \
  --mead-root /content/cmet_training_data/MEAD/FPS25 \
  --scope train \
  --split both \
  --stage model \
  --check-arrays \
  --strict
```
