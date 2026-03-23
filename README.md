# BridgeAPT

A nucleic acid sequence generation and screening tool based on the GVP (Geometric Vector Perceptron) + Transformer architecture. It takes nucleic acid structure PDB files as input to generate sequences. It features an embedded AptShape module to extract DNA shape features and utilizes clustering algorithms to screen for sequences that most closely resemble the shape of the original structure.

## Installation

```bash
cd /path/to/BridgeAPT
pip install -e .
```

Alternatively, use a conda environment:

```bash
conda env create -f environment.yml
conda activate bridgeapt
pip install -e .
```

## Quick Start

### Generate Sequences Only

```bash
bridgeapt-run -i structure.pdb -o ./output/
```

### Generate + Shape Clustering Screening (Recommended)

```bash
bridgeapt-analyze -i structure.pdb -o ./output/
```

Workflow: Generates 1,000 sequences → Extracts shape features via DeepDNAShape → Clusters using three methods (K-Means / Hierarchical / GMM) → Outputs sequences that fall into the same cluster as the original sequence across all three methods (Intersection).

### Batch Processing

```bash
bridgeapt-analyze -i ./pdb_files/ -o ./output/
```

## Parameter Descriptions
### `bridgeapt-run`

| Parameter | Short | Default | Description |
|------|------|--------|------|
| `--input` | `-i` | Required | Path to PDB file or directory |
| `--output` | `-o` | Required | Output directory |
| `--num-sequences` | `-n` | 1000 | Number of sequences generated per PDB |
| `--temperature` | `-t` | 1.0 | Sampling temperature (higher is more random) |
| `--length` | `-l` | Auto | Sequence length (matches PDB residue count by default) |
| `--model-path` | `-m` | Built-in | Path to custom weight file |
| `--analyze` | — | Off | Enable shape clustering screening |
| `--clusters` | `-k` | 5 | Number of clusters |
| `--layer` | — | 7 | AptShape flanking layers (0–7) |

### `bridgeapt-analyze`(Full Analysis Workflow)

In addition to `bridgeapt-run --analyze`it supports:

| Parameter | Short | Default | Description |
|------|------|--------|------|
| `--features` | `-f` | 8 types | DNA shape features to extract |
| `--methods` | — | All | Clustering methods (kmeans hierarchical gmm) |

## Output Files

After running `bridgeapt-analyze`, the output directory structure is as follows:

```
output/
└── {stem}/
    ├── {stem}_final.txt              # ain Result: Intersection of the three methods
    ├── {stem}_kmeans_selected.txt    # K-Means screened sequences
    ├── {stem}_hierarchical_selected.txt
    ├── {stem}_gmm_selected.txt
    ├── {stem}_kmeans_pca.png         # K-Means clustering PCA plot
    ├── {stem}_hierarchical_pca.png
    ├── {stem}_gmm_pca.png
    ├── {stem}_kmeans_heatmap.png     # Cluster center heatmap
    ├── {stem}_hierarchical_heatmap.png
    ├── {stem}_gmm_heatmap.png
    └── {stem}_consensus_pca.png      # Final intersection consensus PCA plot
```

`{stem}_final.txt` is the primary output. Each line contains a sequence that belongs to the same cluster as the original PDB sequence under all three clustering methods.

## Python API

```python
from bridgeapt.runner import Runner

runner = Runner()

# Generate sequences only
sequences = runner.run("structure.pdb", num_sequences=1000)

# Generate + Clustering screening
results = runner.run_with_analysis(
    pdb_path="structure.pdb",
    output_dir="./output",
    num_sequences=1000,
    n_clusters=5,
)
consensus = results["consensus"]  # Intersection of the three methods
```

## Model Architecture

BridgeAPT uses GVP to process 3D coordinates and dihedral angle features, combined with a Transformer encoder to generate nucleic acid sequences.

- Input：Atomic coordinates from PDB files (C4', C1', N1, C2, C5', O5', P)
- Features: Coordinate features [B, L, 21] + Dihedral sin/cos features [B, L, 6]
- Output: Nucleic acid sequences (A/T/C/G)

The shape screening module (DeepDNAShape) extracts 8 features by default: MGW, Roll, HelT, ProT, Shift, Slide, Rise, and Tilt.

## Dependencies

- Python >= 3.8
- torch >= 1.10
- biopython >= 1.79
- tensorflow（DeepDNAShape 依赖）
- scikit-learn >= 1.0
- matplotlib >= 3.5
- seaborn >= 0.11
