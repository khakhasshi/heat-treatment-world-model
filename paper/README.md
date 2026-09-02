# 论文 LaTeX 稿

本文档是冻结实验版本 `experiment-freeze-v1` 对应的学术论文初稿。

实验图由冻结数据和已有模型统一生成：

```bash
uv run --no-sync python scripts/generate_paper_trajectory_figures.py
```

```bash
cd paper
make
```

编译结果位于 `paper/build/main.pdf`。正文使用 `ctexrep`、XeLaTeX 和 GB/T 7714 数字制参考文献。正式提交前可按学校模板调整封面、页眉、字号和学位声明，正文结构与交叉引用可以直接保留。
