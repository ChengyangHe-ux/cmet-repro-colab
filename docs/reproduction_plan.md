# C-MET 历史 Demo 计划

> [!WARNING]
> 这是仓库早期仅验证官方权重推理的历史计划，不是当前全流程复现入口。完整数据、训练、消融、benchmark 和评估请使用 `notebooks/C-MET_Full_Reproduction_Colab.ipynb`，并阅读 `docs/full_reproduction_plan.md`。

## 复现范围

以下内容仅保留用于理解早期 demo 范围：第一阶段只复现官方权重检查点推理，不做完整训练。

## 阶段 1：官方演示

任务清单：

- [ ] 在 Colab 打开 `notebooks/C-MET_Colab_Demo.ipynb`。
- [ ] 优先选择 `A100 GPU`，没有就选 `L4 GPU`，再不行选 `T4 GPU`。
- [ ] 运行 GPU 检查单元。
- [ ] 克隆官方 C-MET。
- [ ] 安装依赖。
- [ ] 运行 happy 演示。
- [ ] 运行 sarcastic 演示。
- [ ] 保存输出视频。

完成标准：

- `ChatGPT_man3_happy.mp4` 能播放。
- `ChatGPT_man3_sarcastic.mp4` 能播放。

## 阶段 2：整理汇报证据

任务清单：

- [ ] 记录 GPU 型号和显存。
- [ ] 记录依赖问题和解决办法。
- [ ] 截图或导出生成视频。
- [ ] 定性比较基础情绪和扩展情绪输出。

完成标准：

- 汇报里能展示至少一个成功生成结果，或者给出清楚的阻塞日志。

## 阶段 3：小样本训练

任务清单：

- [ ] 准备一个小的 MEAD 子集。
- [ ] 使用官方预处理流程。
- [ ] 尽量提前抽取 `emotion2vec+large` 特征。
- [ ] 先只训练 C-MET connector。
- [ ] 保持较小 batch size。

完成标准：

- 训练循环能启动，并且 loss 在少量迭代内下降。

## 汇报前先不做

- 完整 MEAD 训练。
- 完整 CREMA-D 评估。
- FID/FVD 复现。
- 完整消融实验表复现。
