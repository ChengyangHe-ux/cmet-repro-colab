# 排错手册

## `torch.cuda.is_available()` 是 false

说明当前运行时没有使用 GPU。在 Colab 里进入：

```text
Runtime -> Change runtime type -> GPU
如果 Colab 界面是中文：运行时 -> 更改运行时类型 -> GPU
```

优先选 `A100`，其次 `L4`，再其次 `T4`。

## `CUDA out of memory`

先这样处理：

- 不要使用 `--sr`。
- 优先使用 A100 或 L4。
- 重启运行时，只运行必要单元。
- 不要同时跑多个演示。

## 权重检查点下载失败

官方 `inference.py` 会尝试从 Hugging Face 下载这些权重：

- `pretrained_weights/Audio2Lip.pt`
- `pretrained_weights/EDTalk.pt`
- `pretrained_weights/EDTalk-V.pt`
- `checkpoints/_epoch_2105_checkpoint_step000200000.pth`

如果自动下载失败，就用官方 README 里的链接或 Hugging Face 模型页手动处理。

## `ModuleNotFoundError`

在当前运行时安装缺失的包：

```bash
pip install package_name
```

如果报错涉及编译包，安装后重启运行时。

## `ModuleNotFoundError: No module named 'funasr'`

演示路径其实不需要 `funasr`，因为 Colab 笔记本使用的是预先抽取好的 `emotion2vec+large` 特征文件夹。运行这个单元：

```text
4.1 修补可选导入和兼容问题
```

然后重新运行推理单元。

## `pip install -r requirements.txt` 失败

第一次 Colab 演示不要安装完整官方 requirements。Colab 可能使用 Python 3.12，而官方项目更接近 Python 3.8/3.9 环境。使用 Colab 笔记本里的最小依赖安装单元：

```bash
python -m pip install --no-cache-dir \
  'setuptools<82' 'jedi>=0.16' \
  huggingface_hub==1.20.1 omegaconf==2.3.0 tqdm==4.67.3 \
  pandas==2.2.2 scipy==1.16.3 \
  librosa==0.10.2.post1 'soundfile>=0.12.1' \
  imageio==2.37.3 imageio-ffmpeg==0.6.0 \
  opencv-python-headless==4.13.0.92 pillow==11.3.0
```

完整 requirements 更适合在可控的 conda Python 3.9 环境里使用。

## `pandas` 或 `setuptools` 依赖冲突警告

第一次 Colab 演示的依赖单元不要用 `--upgrade`。它可能把 Colab 核心包升级得太新，比如 `pandas==3.x` 或 `setuptools>=82`，从而和 Colab/Torch 冲突。运行 Colab 笔记本里固定版本的最小依赖单元，就能恢复到兼容版本。

## `AttributeError: module 'numpy' has no attribute 'complex'`

这是旧 `librosa` 和新 NumPy 的兼容问题。运行更新后的 Colab 笔记本单元：

```text
4. 安装最小 Python 依赖
4.1 修补可选导入和兼容问题
4.2 验证补丁是否生效
```

安装单元已经固定：

```bash
librosa==0.10.2.post1
```

补丁单元还会修改 `src/audio.py`，在导入 `librosa` 前恢复旧 NumPy 别名。

## `AttributeError: module 'pkgutil' has no attribute 'ImpImporter'`

这是 Python 3.12 下 `moviepy -> pygame -> pkg_resources` 导入链导致的兼容问题。第一次演示不使用 `--sr`，所以不需要 `moviepy.editor`。

运行更新后的 Colab 笔记本单元：

```text
4.1 修补可选导入和兼容问题
```

它会从 `inference.py` 和 `src/util.py` 里移除顶层 `moviepy.editor` 导入，适用于不跑超分的演示路径。

## `_pickle.UnpicklingError: Weights only load failed`

新版 PyTorch 默认用更安全的 `weights_only=True` 读取权重检查点。官方 C-MET 演示权重检查点是可信文件，但 `inference.py` 需要显式写成：

```python
torch.load(..., weights_only=False)
```

运行更新后的 Colab 笔记本单元：

```text
4.1 修补可选导入和兼容问题
```

然后重新运行 happy 演示单元。补丁会修改 `inference.py` 里的三处 `torch.load`。

## `AttributeError: module 'torchvision.io' has no attribute 'read_video'`

一些新版 Colab torchvision 不再暴露官方 C-MET `src/util.py` 使用的旧视频 API：

```python
torchvision.io.read_video
torchvision.io.write_video
```

运行更新后的 Colab 笔记本单元：

```text
4.1 修补可选导入和兼容问题
```

然后运行：

```text
4.2 验证补丁是否生效
```

补丁会在 `src/util.py` 里追加新的 `vid_preprocessing()` 和 `save_video()` 函数，改用 `imageio + ffmpeg`，这样推理就不再依赖 torchvision 的视频 I/O。

## `gfpgan` 或 `basicsr` 报错

第一次演示不要用超分。基础推理命令里不要加 `--sr`。

## Mac 不能干净地跑官方推理

官方 `inference.py` 里有直接 `.cuda()` 调用。先用 CUDA 运行时复现。32GB Mac 仍然适合读论文、记笔记、做预处理和写报告。
