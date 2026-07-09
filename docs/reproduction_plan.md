# C-MET Reproduction Plan

## Scope

The first reproduction target is official checkpoint inference, not full training.

## Phase 1: Official Demo

Tasks:

- [ ] Open `notebooks/C-MET_Colab_Demo.ipynb` in Colab.
- [ ] Select `A100 GPU` if available, otherwise `L4 GPU`, otherwise `T4 GPU`.
- [ ] Run the GPU check cell.
- [ ] Clone official C-MET.
- [ ] Install dependencies.
- [ ] Run happy demo.
- [ ] Run sarcastic demo.
- [ ] Save output videos.

Exit condition:

- `ChatGPT_man3_happy.mp4` plays.
- `ChatGPT_man3_sarcastic.mp4` plays.

## Phase 2: Report Evidence

Tasks:

- [ ] Record GPU type and VRAM.
- [ ] Record dependency issues.
- [ ] Screenshot or export generated videos.
- [ ] Compare basic emotion and extended emotion outputs qualitatively.

Exit condition:

- The report can show one successful generated output or a precise blocker log.

## Phase 3: Small-Sample Training

Tasks:

- [ ] Prepare a small MEAD subset.
- [ ] Use official preprocessing.
- [ ] Pre-extract emotion2vec+large features if possible.
- [ ] Train C-MET connector only first.
- [ ] Keep batch size small.

Exit condition:

- Training loop starts and loss decreases for a small number of iterations.

## Out Of Scope Before The Presentation

- Full MEAD training.
- Full CREMA-D evaluation.
- FID/FVD reproduction.
- Complete ablation table reproduction.

