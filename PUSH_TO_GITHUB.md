# Push This Repo To GitHub

Local repository path:

```text
/Users/hechengyang/科研/CV/跨模态情感迁移在对话人脸视频情感编辑中的应用/cmet-repro-colab
```

The local repo is already initialized and committed.

## Option 1: Use GitHub CLI

If `gh auth status` works:

```bash
cd "/Users/hechengyang/科研/CV/跨模态情感迁移在对话人脸视频情感编辑中的应用/cmet-repro-colab"
gh repo create cmet-repro-colab --public --source=. --remote=origin --push
```

If `gh auth status` fails, log in again:

```bash
gh auth login --hostname github.com --git-protocol https --web
```

Then rerun:

```bash
gh repo create cmet-repro-colab --public --source=. --remote=origin --push
```

## Option 2: Create Repo In Browser

1. Go to https://github.com/new
2. Create an empty public repository named:

```text
cmet-repro-colab
```

3. Do not add README, .gitignore, or license in the browser.

4. Push from local:

```bash
cd "/Users/hechengyang/科研/CV/跨模态情感迁移在对话人脸视频情感编辑中的应用/cmet-repro-colab"
git remote add origin https://github.com/ChengyangHe-ux/cmet-repro-colab.git
git branch -M main
git push -u origin main
```

## Colab Link After Push

After the repository exists on GitHub, open:

```text
https://colab.research.google.com/github/ChengyangHe-ux/cmet-repro-colab/blob/main/notebooks/C-MET_Colab_Demo.ipynb
```

