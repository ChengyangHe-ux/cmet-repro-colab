# 把这个仓库推到 GitHub

本地仓库路径：

```text
/Users/hechengyang/科研/CV/跨模态情感迁移在对话人脸视频情感编辑中的应用/cmet-repro-colab
```

本地仓库已经初始化并完成过提交。

## 方案 1：使用 GitHub CLI

如果 `gh auth status` 正常：

```bash
cd "/Users/hechengyang/科研/CV/跨模态情感迁移在对话人脸视频情感编辑中的应用/cmet-repro-colab"
gh repo create cmet-repro-colab --public --source=. --remote=origin --push
```

如果 `gh auth status` 失败，重新登录：

```bash
gh auth login --hostname github.com --git-protocol https --web
```

然后重新运行：

```bash
gh repo create cmet-repro-colab --public --source=. --remote=origin --push
```

## 方案 2：在浏览器里创建仓库

1. 打开 https://github.com/new
2. 创建一个空的公开仓库，仓库名：

```text
cmet-repro-colab
```

3. 不要在浏览器里添加 README、.gitignore 或 license。

4. 从本地推送：

```bash
cd "/Users/hechengyang/科研/CV/跨模态情感迁移在对话人脸视频情感编辑中的应用/cmet-repro-colab"
git remote add origin https://github.com/ChengyangHe-ux/cmet-repro-colab.git
git branch -M main
git push -u origin main
```

## 推送后的 Colab 链接

GitHub 上有仓库之后，打开：

```text
https://colab.research.google.com/github/ChengyangHe-ux/cmet-repro-colab/blob/main/notebooks/C-MET_Colab_Demo.ipynb
```
