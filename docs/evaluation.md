# C-MET Result Evaluation Guide

This reproduction should be evaluated in three layers: demo success, visual quality, then optional quantitative metrics.

## 1. Minimum Success Criteria

You can claim a successful first demo reproduction if all of these pass:

- `ChatGPT_man3_happy.mp4` and `ChatGPT_man3_sarcastic.mp4` are generated.
- Both videos play correctly and contain an audio stream.
- Output resolution is `256x256`, and duration is greater than zero.
- Identity is stable, with no obvious face collapse or identity drift.
- Mouth motion is roughly synchronized with the driving audio.
- The happy and sarcastic outputs show visibly different expression styles.

## 2. Quick Evaluation In Colab

Run these notebook sections after generating both videos:

```text
10. Evaluate Outputs: Technical QA
10.1 Evaluate Outputs: Frame Contact Sheet
10.2 Evaluation Rubric For Report
```

The technical QA cell should print:

```text
technical QA: ok
```

The contact sheet can be used directly in the report to show temporal consistency and visual quality.

## 3. How To Explain It In The Report

Use this structure:

- Reproduction target: run the official pretrained checkpoint and generate basic emotion `happy` plus extended emotion `sarcastic`.
- Runtime: record GPU type, Python version, PyTorch/CUDA version, and note that `--sr` is not used.
- Outputs: show both generated videos and the contact sheet.
- Evaluation dimensions: technical validity, emotion transfer, identity preservation, lip synchronization, pose stability, and visual quality.
- Limitation: this is official-checkpoint inference reproduction, not full training reproduction yet.

## 4. Stronger Quantitative Metrics

If there is time after the demo works, add metrics in this order:

- SyncNet / LSE-C / LSE-D for lip-audio synchronization.
- ArcFace cosine similarity for identity preservation.
- Facial emotion classifier score for target emotion transfer.
- FVD / LPIPS / FID only when you have many generated samples and comparable baselines.
