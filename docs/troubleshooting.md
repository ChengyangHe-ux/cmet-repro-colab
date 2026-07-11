# C-MET Colab 排错手册

先确认错误属于哪个阶段：Drive、依赖、数据、特征、训练、推理或评估。不要从第一格反复重跑所有长任务。

## 1. Drive 授权失败

### `credential propagation was unsuccessful`

这是 Colab 浏览器和 Google 授权问题，不是 C-MET、CUDA 或数据问题。

按顺序处理：

1. 浏览器只登录一个 Google 账号。
2. 允许第三方 Cookie、弹窗和重定向。
3. 断开并删除 Colab 运行时。
4. 重新连接后只运行 Drive 挂载格。

不要手动创建 `/content/drive/MyDrive`。新版笔记本会验证 `/content/drive` 是否是真实挂载点，并在失败时立即停止，防止把临时盘误当 Drive。

### Drive 可见但不可写

环境自检会写入并删除 `.cmet_write_test`。失败时检查：

- Drive 容量是否已满。
- 当前 Google 账号是否拥有目标目录权限。
- 共享盘路径是否被误写成 `MyDrive`。

## 2. GPU 与显存

### `torch.cuda.is_available()` 为 `False`

在 Colab 中选择：

```text
运行时 -> 更改运行时类型 -> GPU
```

重新连接后从 GPU 检查格开始。

### 显存低于 20GB

完整训练建议切换 A100。论文使用 24GB RTX 3090。8GB 4060 不适合官方 batch size 的完整训练。

### `CUDA out of memory`

按阶段处理：

- EDTalk 特征抽取：把 `--batch-size 100` 改为 32、16 或 8。
- 持久化推理：降低 `EXPRESSION_BATCH_SIZE` 和 `RENDER_BATCH_SIZE`。
- 训练：先确认 A100；必要时通过 `run_training.py --batch-size` 和 `--batch-size-val` 降低，但要记录偏离论文/官方配置。
- 重启运行时，避免前一个模型仍占显存。

## 3. pip 与依赖冲突

### 不要升级 pip、Torch 或 CUDA

完整笔记本使用：

```bash
python scripts/install_colab_dependencies.py
```

它不会执行 `pip --upgrade`，也不会重装 Torch。

### `setuptools>=82` 或 `pandas 3.x`

仓库固定：

```text
setuptools<82
pandas==2.2.2
```

如果此前手动升级过环境，删除运行时并重新开始比继续补包更可靠。

### `ModuleNotFoundError`

不要直接安装官方完整 `requirements.txt`。先重跑仓库依赖安装格，再运行环境自检。环境自检会明确指出哪个核心模块无法导入。

### `numpy` 没有 `complex` / `float` / `int`

兼容补丁会修改官方 `src/audio.py`，并固定 `librosa==0.10.2.post1`。重新运行补丁格，不要手动降级整个 NumPy 栈。

### `_pickle.UnpicklingError: Weights only load failed`

新版 PyTorch 默认改变了 `torch.load` 行为。补丁会对官方可信 checkpoint 显式使用 `weights_only=False`。确认权重来自官方 Hugging Face 仓库后，重新打补丁。

### `torchvision.io.read_video` 或 `write_video` 不存在

补丁用 `imageio + ffmpeg` 替换官方旧视频 I/O。重新运行补丁，不要为了恢复旧 API 强行重装 Colab 自带 torchvision。

## 4. 数据预处理

### MEAD 官方快捷方式不可见

默认路径是：

```text
/content/drive/MyDrive/MEAD
```

处理顺序：

1. 打开 `https://drive.google.com/drive/folders/1GwXP-KpWOxOenOxITTsURJZQ_1pkd4-j`。
2. 对顶层 `MEAD` 文件夹执行“整理 -> 添加快捷方式”，不要逐个身份添加。
3. 把快捷方式放到 MyDrive 根目录；若名称或位置不同，只修改 Notebook 的 `MEAD_SHARED_ROOT`。
4. 重新挂载 Drive，运行 `RUN_MEAD_SOURCE_CHECK=True`。

不要使用匿名 `gdown` 批量下载。大文件可能触发 Google `Too many users have viewed or downloaded this file recently`，而且完整复制原始 tar 会远超当前 Drive 空间。

### MEAD 来源预检缺少某个身份

来源预检会检查 C-MET 官方 47 个身份。特殊身份 `M026-2`、`M032-2`、`M042-1`、`W021-1` 会映射到公共盘中的基础身份目录。若仍报缺失，通常说明添加的是某个子目录而不是顶层 Part0 文件夹。

### MEAD tar 复制中断

- 只停止单元格、运行时仍存在：再次运行同一开关，会从 `/content/cmet_public_data/.../video.tar.part` 继续。
- 运行时被彻底删除：当前身份的本地 `.part` 会消失，需要重新复制该身份约 19GB；已经写入 Drive 并标记完成的其他身份不会重做。
- 报本地盘不足：至少需要“tar 剩余大小 + 8GB”可用空间。重启运行时清理 `/content`，不要删除 Drive 中的处理结果和状态报告。

### CREMA-D Git LFS 下载失败

Notebook 会安装 `git-lfs`，按 200 个路径一批下载。网络中断后直接重跑 `RUN_CREMAD_PREP_FULL=True`，脚本会检查 Drive 中已有 MP4/WAV，只补缺失文件。不要删除 `/content/cmet_public_data/CREMA-D`，除非仓库损坏且错误明确说它不是 Git 仓库。

### 没有发现 MEAD 视频

检查原始路径是否含：

```text
身份/front/情绪/level_1|level_2|level_3/编号.视频后缀
```

身份名需要类似 `M003`、`W015`、`M026-2`。只处理官方 train/test 身份及 Common/Generic 编号。

### 没有发现 CREMA-D 视频

文件名应类似：

```text
1001_DFA_HAP_XX.flv
```

不要在上传前改掉 actor、sentence、emotion、intensity 字段。

### 官方裁脸器找不到人脸轨迹

- 确认输入是正脸或近正脸 talking-face 视频。
- 先查看失败报告中的源视频。
- 如果原始数据已经按 EDTalk 协议稳定裁脸，可把 `CROP_MODE="none"`，但必须随后通过 256x256/25 FPS 媒体门禁。

### ffmpeg 路径含空格时报错

把 Drive 根目录保持为简单路径，例如：

```text
MyDrive/C-MET-full
```

不要把原始数据放进含复杂引号或特殊符号的目录。

### 旧版预处理结果可能音画错位

旧版官方裁脸流程可能把 MP4 从裁后临时片段导出，却把 WAV 从完整原视频导出。新版 schema v2 会在输出根目录写入 `.cmet_prepare_state.json`，并自动重做尚未升级的音视频对。

- 断线后直接重跑同一个 `RUN_*_PREP_FULL` 开关，脚本只继续未升级条目。
- 不要删除状态文件，也不要把零字节 MP4/WAV 当成已完成。
- 只有你已经逐条确认旧媒体严格对齐时，才使用 `--trust-existing-media`。

若门禁报 `incomplete_media_pair`，说明 Colab 在 MP4/WAV 成对替换中途断线。不要手动删其中一个文件，直接重跑对应 full 预处理格，脚本会整对重建并清除事务标记。

## 5. 特征抽取

### emotion2vec 下载失败

- 确认 Colab 能访问 ModelScope/Hugging Face。
- 默认使用 `hub=ms`；海外环境可把命令改为 `--hub hf`。
- 已成功生成且 shape 正确的 NPY 会自动跳过，重跑不会从头覆盖。

### emotion2vec shape 不是 `(1024,)`

确认模型是：

```text
iic/emotion2vec_plus_large
```

不要换成基础版 emotion2vec 或帧级 `granularity="frame"`。

### EDTalk 特征有 NaN/Inf 或 shape 错误

- 删除对应损坏的 `*_ED_exp.npy`、`*_ED_pose.npy`、`*_ED_lip.npy`。
- 对该视频使用 `--overwrite` 重跑。
- 先确认输入 MP4 能被 ffprobe 解码且时长大于 0。

若门禁报 `incomplete_edtalk_features`，直接重跑 `RUN_EDTALK_FULL`。这表示上次在 exp/pose/lip 三件套替换中途断线，脚本会强制整组重做，避免混用新旧特征。

## 6. 数据门禁失败

### `missing_mp4` 但训练缓存中的 MP4 是零字节

`model` 阶段只要求路径存在，训练不会解码 MP4；零字节标记是预期设计。不要对本地训练缓存运行 `--check-media`。

### `training` 通过但 `model` 失败

官方 DataLoader 的特征采样编号与索引编号不同。`model` 门禁按真实读取合同检查：

- 中性索引：1-3、31-40。
- 情绪索引：1-3、21-30。
- 中性表达特征：1-10、31-40。
- 情绪表达特征：1-10、21-30。
- emotion2vec：中性 1-40、情绪 1-30。

缺哪个编号就补哪个，不要跳过直接训练，否则 DataLoader 可能长时间重复找文件。

## 7. 训练与续训

### 先跑 smoke

完整训练前必须看到：

- 2 个训练 step。
- 1 个验证 batch。
- 非空 checkpoint。

只看到进程启动不算通过。

### 实验目录已有 checkpoint

`run_training.py` 会阻止无意覆盖。选择：

```bash
--resume auto
```

或换一个 seed。不要删除已有 checkpoint 后假装重新开始。

### 最新 checkpoint 是零字节或半文件

新补丁先写 `.tmp`，成功后原子替换正式 `.pth`；自动续训会忽略零字节文件。若你使用旧版本留下损坏文件，将它移动到备份目录后再续训。

### checkpoint 占满 Drive

完整实验默认每 1000 step 保存，但只保留最近 3 份和每 5 万 step 里程碑。先确认 dry-run manifest 的命令中包含：

```text
--checkpoint_keep_recent 3
--checkpoint_milestone_interval 50000
```

如果目录中已有旧版本留下的大量 checkpoint，先更新仓库并正常续训到下一次成功保存；新补丁会在新文件原子落盘后清理非里程碑旧文件。不要在只有一份可用 checkpoint 时手工批量删除。

### Colab 断线后怎么继续

重新执行：

1. GPU 和 Drive。
2. 路径配置。
3. 克隆、依赖、补丁、权重连接、环境自检。
4. 解压训练缓存并运行本地缓存门禁。
5. 打开对应 `RESUME_*` 开关。

不要重新运行完整数据预处理和特征抽取。

训练缓存使用临时 tar 原子替换；若构建中断，旧缓存仍保留。重新打开 `BUILD_TRAIN_CACHE` 即可，不要使用隐藏的 `.tmp` 文件。

### 续训是否逐 bit 一致

补丁恢复模型、优化器、step、epoch 和主进程 Python/NumPy/Torch/CUDA 随机状态。但官方 DataLoader 使用多 worker，worker 内部随机状态没有写入 checkpoint，因此不能保证断点前后逐 bit 完全相同。

## 8. 推理与 benchmark

### `neutral` 特征池找不到

neutral 必须是：

```text
audios/MEAD/neutral/emotion2vec+large_features
```

不是 `audios/gemini/neutral`。请使用最新 `run_qualitative_inference.py`。

### `desirous` 路径找不到

官方目录名是：

```text
audios/gemini/desirous
```

不是 `desire`。

### 自训练 checkpoint 未到 20 万 step

正式推理默认会停止。继续训练到 20 万 step；只做中间诊断时才显式设置：

```python
ALLOW_PARTIAL_SELF_TRAINED_CHECKPOINT = True
```

### JSONL 最后一行损坏

断线可能截断最后一条记录。脚本会原子裁掉没有换行结尾的最后一条残片，再从前一条完整记录继续；如果中间行损坏，仍会报错，避免静默丢失进度。

### 输出被错误跳过

新版笔记本按 checkpoint 文件名隔离输出。确认你没有手动把不同 checkpoint 指向同一输出目录。

### benchmark 在模型加载后才发现数据缺失

新版 runner 会先检查所选样本、checkpoint 和情绪池，再加载模型。若仍出现输入缺失，确认 Colab 使用的是 GitHub 最新 commit。

### `dataset` 协议提示 10-shot 候选不足

先确认已同时运行：

```python
RUN_MEAD_E2V_FULL = True
RUN_CREMAD_E2V_FULL = True
RUN_FEATURE_GATE = True
```

MEAD 需要 test 身份中对应情绪和 `level_1/2/3` 的 emotion2vec；CREMA-D 需要对应情绪和 `HI/LO/MD/XX` 的 emotion2vec。不要用 `official-static` 临时绕过缺失后仍把结果写成强度感知 benchmark。

### 切换情绪协议后结果被跳过或指标 ID 不匹配

确认 `BENCHMARK_EMOTION_PROTOCOL` 只取 `dataset` 或 `official-static`。新版路径为：

```text
videos/<emotion_protocol>/<dataset>/
manifests/<emotion_protocol>/<dataset>.csv
progress/<emotion_protocol>/<dataset>.jsonl
```

两套协议的 sample ID 也包含协议名。外部评估 CSV 必须来自同一份 manifest，不能复用另一协议的 `sample_id`。

指标汇总会读取 manifest 中的 `emotion_protocol`，并与命令和 progress 逐项核验；协议不同会直接报错，不再生成混合报告。

## 9. 评估

### 技术检查通过但指标很差

技术检查只说明视频可播放。情绪、同步、身份和 FID/FVD 需要独立评估器。

### `--strict` 报缺少指标

严格模式要求五项指标和全部 benchmark ID。检查：

- `sample_metrics.csv` 是否覆盖所有 sample ID。
- 是否存在重复或未知 ID。
- `global_metrics.json` 是否含 `aitv`、`fid`、`fvd`。
- benchmark JSONL 是否全部完成。

### 能否直接使用网上任意 SyncNet/Emotion-FAN

可以做自定义协议实验，但不能称为严格论文数值复现。作者没有公开其精确预处理和微调 checkpoint，应在报告中明确标注实现差异。
