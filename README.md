# BridgeAPT

基于 GVP（Geometric Vector Perceptron）+ Transformer 架构的核酸序列生成与筛选工具。输入核酸结构 PDB 文件，生成核酸序列，并通过内嵌的 DeepDNAShape 模块提取 DNA 形状特征，利用聚类算法筛选出与原始结构形状最相似的序列。

## 安装

```bash
cd /path/to/bridgeapt-package
pip install -e .
```

或使用 conda 环境：

```bash
conda env create -f environment.yml
conda activate bridgeapt
pip install -e .
```

> DeepDNAShape 已内嵌为 `bridgeapt.deepdnashape` 子模块，无需单独安装。

## 快速开始

### 仅生成序列

```bash
bridgeapt-run -i structure.pdb -o ./output/
```

### 生成 + 形状聚类筛选（推荐）

```bash
bridgeapt-analyze -i structure.pdb -o ./output/
```

流程：生成 1000 条序列 → DeepDNAShape 提取形状特征 → K-Means / 层次聚类 / GMM 三种方法聚类 → 输出三种方法均与原始序列同簇的序列（交集）。

### 批量处理

```bash
bridgeapt-analyze -i ./pdb_files/ -o ./output/
```

## 参数说明

### `bridgeapt-run`

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | `-i` | 必填 | PDB 文件路径或目录 |
| `--output` | `-o` | 必填 | 输出目录 |
| `--num-sequences` | `-n` | 1000 | 每个 PDB 生成的序列数量 |
| `--temperature` | `-t` | 1.0 | 采样温度（越高越随机） |
| `--length` | `-l` | 自动 | 生成序列长度（默认自动匹配 PDB 残基数） |
| `--model-path` | `-m` | 内置权重 | 自定义权重文件路径 |
| `--analyze` | — | 关闭 | 开启形状聚类筛选 |
| `--clusters` | `-k` | 5 | 聚类数 |
| `--layer` | — | 7 | DeepDNAShape flanking 层数（0–7） |

### `bridgeapt-analyze`（完整分析流程）

在 `bridgeapt-run --analyze` 基础上额外支持：

| 参数 | 简写 | 默认值 | 说明 |
|------|------|--------|------|
| `--features` | `-f` | 8 种 | 要提取的 DNA 形状特征 |
| `--methods` | — | 全部 | 聚类方法（kmeans hierarchical gmm） |

## 输出文件

运行 `bridgeapt-analyze` 后，输出目录结构如下：

```
output/
└── {stem}/
    ├── {stem}_final.txt              # 最终输出：三种方法交集序列（主要结果）
    ├── {stem}_kmeans_selected.txt    # K-Means 筛选序列
    ├── {stem}_hierarchical_selected.txt
    ├── {stem}_gmm_selected.txt
    ├── {stem}_kmeans_pca.png         # K-Means 聚类 PCA 图
    ├── {stem}_hierarchical_pca.png
    ├── {stem}_gmm_pca.png
    ├── {stem}_kmeans_heatmap.png     # 聚类中心热图
    ├── {stem}_hierarchical_heatmap.png
    ├── {stem}_gmm_heatmap.png
    └── {stem}_consensus_pca.png      # 最终交集序列综合 PCA 图
```

`{stem}_final.txt` 为主要结果，每行一条序列，均与原始 PDB 序列在三种聚类方法下同属一簇。

## Python API

```python
from bridgeapt.runner import Runner

runner = Runner()

# 仅生成序列
sequences = runner.run("structure.pdb", num_sequences=1000)

# 生成 + 聚类筛选
results = runner.run_with_analysis(
    pdb_path="structure.pdb",
    output_dir="./output",
    num_sequences=1000,
    n_clusters=5,
)
consensus = results["consensus"]  # 三种方法交集序列
```

## 模型架构

BridgeAPT 使用 GVP 处理三维坐标和二面角特征，结合 Transformer 编码器生成核酸序列。

- 输入：PDB 文件中的原子坐标（C4', C1', N1, C2, C5', O5', P）
- 特征：坐标特征 [B, L, 21] + 二面角 sin/cos 特征 [B, L, 6]
- 输出：核酸序列（A/T/C/G）

形状筛选模块（DeepDNAShape）默认提取 8 种特征：MGW、Roll、HelT、ProT、Shift、Slide、Rise、Tilt。

## 依赖项

- Python >= 3.8
- torch >= 1.10
- biopython >= 1.79
- tensorflow（DeepDNAShape 依赖）
- scikit-learn >= 1.0
- matplotlib >= 3.5
- seaborn >= 0.11
