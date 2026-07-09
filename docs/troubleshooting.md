# Troubleshooting

## `torch.cuda.is_available()` is false

The runtime is not using a GPU. In Colab, go to:

```text
Runtime -> Change runtime type -> GPU
```

Prefer `A100`, then `L4`, then `T4`.

## `CUDA out of memory`

First attempts:

- Do not use `--sr`.
- Prefer A100 or L4.
- Restart runtime and run only the required cells.
- Avoid running multiple demo jobs in parallel.

## Checkpoint download fails

The official `inference.py` tries to download weights from Hugging Face:

- `pretrained_weights/Audio2Lip.pt`
- `pretrained_weights/EDTalk.pt`
- `pretrained_weights/EDTalk-V.pt`
- `checkpoints/_epoch_2105_checkpoint_step000200000.pth`

If automatic download fails, use the official README links or Hugging Face model page.

## `ModuleNotFoundError`

Install the missing package in the active runtime:

```bash
pip install package_name
```

Then restart the runtime if the error involves compiled packages.

## `ModuleNotFoundError: No module named 'funasr'`

For the demo path, `funasr` is not actually needed because the notebook uses pre-extracted `emotion2vec+large` feature folders. Run the notebook cell named:

```text
4.1 Patch Optional funasr Import
```

Then rerun the inference cell.

## `pip install -r requirements.txt` fails

Do not use the full official requirements for the first Colab demo. Colab may use Python 3.12, while the official project was tested with Python 3.8/3.9. Use the minimal install cell in the notebook:

```bash
python -m pip install \
  huggingface_hub omegaconf tqdm pandas numpy scipy librosa soundfile \
  moviepy==1.0.3 imageio-ffmpeg av opencv-python-headless pillow
```

The full requirements are more appropriate for a controlled conda environment with Python 3.9.

## `AttributeError: module 'numpy' has no attribute 'complex'`

This is an old `librosa`/new NumPy compatibility issue. Run the updated notebook cells:

```text
4. Install Minimal Python Dependencies
4.1 Patch Optional Imports And Compatibility
4.2 Verify Audio Dependencies
```

The install cell now pins:

```bash
librosa==0.10.2.post1
```

The patch cell also edits `src/audio.py` so older NumPy aliases are restored before `librosa` is imported.

## `AttributeError: module 'pkgutil' has no attribute 'ImpImporter'`

This is a Python 3.12 compatibility issue from the `moviepy -> pygame -> pkg_resources` import chain. The first demo does not need `moviepy.editor` because we are not using `--sr`.

Run the updated notebook cell:

```text
4.1 Patch Optional Imports And Compatibility
```

It removes the top-level `moviepy.editor` import from `inference.py` and `src/util.py` for the no-super-resolution demo path.

## `_pickle.UnpicklingError: Weights only load failed`

Newer PyTorch versions load checkpoints with a safer `weights_only=True` default. The official C-MET demo checkpoints are trusted files, but `inference.py` needs to explicitly call:

```python
torch.load(..., weights_only=False)
```

Run the updated notebook cell:

```text
4.1 Patch Optional Imports And Compatibility
```

Then rerun the happy demo cell. The patch edits the three `torch.load` calls in `inference.py`.

## `gfpgan` or `basicsr` errors

Do not use super-resolution for the first demo. The baseline inference command should not include `--sr`.

## Mac Cannot Run Official Inference Cleanly

The official `inference.py` contains direct `.cuda()` calls. Use a CUDA runtime first. The 32GB Mac is still useful for reading, note-taking, preprocessing, and report writing.
