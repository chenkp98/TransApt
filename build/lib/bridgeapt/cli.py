import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="bridgeapt-run",
        description="BridgeAPT: 从核酸结构 PDB 文件生成核酸序列",
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="输入 PDB 文件路径或包含 .pdb 文件的目录",
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="输出目录，生成的序列 .txt 文件将保存在此",
    )
    parser.add_argument(
        "--num-sequences", "-n", type=int, default=1000,
        help="每个 PDB 文件生成的序列数量（默认: 1000）",
    )
    parser.add_argument(
        "--temperature", "-t", type=float, default=1.0,
        help="采样温度（默认: 1.0）",
    )
    parser.add_argument(
        "--length", "-l", type=int, default=None,
        help="生成序列长度（默认: 自动使用 PDB 实际残基数）",
    )
    parser.add_argument(
        "--model-path", "-m", default=None,
        help="自定义权重文件路径（默认使用包内预训练权重）",
    )
    parser.add_argument(
        "--analyze", action="store_true",
        help="生成序列后自动进行 DNA 形状特征提取和聚类筛选",
    )
    parser.add_argument(
        "--clusters", "-k", type=int, default=5,
        help="聚类数（--analyze 模式下有效，默认: 5）",
    )
    parser.add_argument(
        "--layer", type=int, default=7,
        help="deepDNAshape flanking 层数 0-7（--analyze 模式下有效，默认: 7）",
    )

    args = parser.parse_args()

    # 验证输入路径
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] 输入路径不存在: {input_path}", file=sys.stderr)
        sys.exit(2)

    # 收集 PDB 文件列表
    if input_path.is_file():
        if not input_path.suffix.lower() == ".pdb":
            print(f"[ERROR] 输入文件不是 .pdb 格式: {input_path}", file=sys.stderr)
            sys.exit(2)
        pdb_files = [input_path]
    else:
        pdb_files = sorted(input_path.glob("*.pdb"))
        if not pdb_files:
            print(f"[WARNING] 目录中未找到任何 .pdb 文件: {input_path}", file=sys.stderr)
            sys.exit(1)

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 延迟导入，避免在参数解析阶段加载 torch
    from .runner import Runner

    try:
        runner = Runner(model_path=args.model_path)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    success_count = 0
    fail_count = 0

    for pdb_file in pdb_files:
        out_file = output_dir / (pdb_file.stem + ".txt")
        print(f"处理: {pdb_file.name} → {out_file.name}")
        try:
            if args.analyze:
                # 端到端：生成 + 形状分析 + 聚类筛选
                analysis_dir = output_dir / (pdb_file.stem + "_analysis")
                results = runner.run_with_analysis(
                    pdb_path=pdb_file,
                    output_dir=analysis_dir,
                    num_sequences=args.num_sequences,
                    temperature=args.temperature,
                    length=args.length,
                    n_clusters=args.clusters,
                    layer=args.layer,
                )
                consensus_path = analysis_dir / f"{pdb_file.stem}_final.txt"
                n_consensus = len(results.get("consensus", []))
                print(f"  ✓ 分析完成，共识序列 {n_consensus} 条 → {consensus_path}")
            else:
                sequences = runner.run(
                    pdb_path=pdb_file,
                    num_sequences=args.num_sequences,
                    temperature=args.temperature,
                    length=args.length,
                )
                with open(out_file, "w", encoding="utf-8") as f:
                    for seq in sequences:
                        f.write(seq + "\n")
                print(f"  ✓ 生成 {len(sequences)} 条序列 → {out_file}")
            success_count += 1
        except Exception as e:
            print(f"[ERROR] 处理 {pdb_file.name} 失败: {e}", file=sys.stderr)
            fail_count += 1

    print(f"\n完成: {success_count} 个文件成功，{fail_count} 个文件失败")
    if fail_count > 0 and success_count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()


def analyze():
    """bridgeapt-analyze 命令：对 PDB 文件生成序列，提取 DNA 形状特征并聚类筛选。"""
    parser = argparse.ArgumentParser(
        prog="bridgeapt-analyze",
        description="BridgeAPT: 生成序列 → DNA 形状特征提取 → 聚类筛选",
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="输入 PDB 文件路径或包含 .pdb 文件的目录",
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="输出目录，聚类图和筛选序列将保存在此",
    )
    parser.add_argument(
        "--num-sequences", "-n", type=int, default=1000,
        help="每个 PDB 生成的序列数量（默认: 1000）",
    )
    parser.add_argument(
        "--temperature", "-t", type=float, default=1.0,
        help="采样温度（默认: 1.0）",
    )
    parser.add_argument(
        "--length", "-l", type=int, default=None,
        help="生成序列长度（默认: 自动使用 PDB 实际残基数）",
    )
    parser.add_argument(
        "--clusters", "-k", type=int, default=5,
        help="K-Means/层次/GMM 聚类数（默认: 5）",
    )
    parser.add_argument(
        "--features", "-f", nargs="+",
        default=["MGW", "Roll", "HelT", "ProT", "Shift", "Slide", "Rise", "Tilt"],
        help="要提取的 DNA 形状特征（默认: MGW Roll HelT ProT Shift Slide Rise Tilt）",
    )
    parser.add_argument(
        "--methods", nargs="+", default=["kmeans", "hierarchical", "gmm"],
        help="聚类方法（默认: kmeans hierarchical gmm）",
    )
    parser.add_argument(
        "--layer", type=int, default=7,
        help="deepDNAshape flanking 层数 0-7（默认: 7）",
    )
    parser.add_argument(
        "--model-path", "-m", default=None,
        help="自定义权重文件路径",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] 输入路径不存在: {input_path}", file=sys.stderr)
        sys.exit(2)

    if input_path.is_file():
        pdb_files = [input_path]
    else:
        pdb_files = sorted(input_path.glob("*.pdb"))
        if not pdb_files:
            print(f"[WARNING] 目录中未找到任何 .pdb 文件: {input_path}", file=sys.stderr)
            sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    from .runner import Runner

    try:
        runner = Runner(model_path=args.model_path)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    for pdb_file in pdb_files:
        print(f"\n{'='*50}")
        print(f"处理: {pdb_file.name}")
        pdb_out_dir = output_dir / pdb_file.stem
        try:
            runner.run_with_analysis(
                pdb_path=pdb_file,
                output_dir=pdb_out_dir,
                num_sequences=args.num_sequences,
                temperature=args.temperature,
                length=args.length,
                features=args.features,
                n_clusters=args.clusters,
                methods=args.methods,
                layer=args.layer,
            )
            consensus_path = pdb_out_dir / f"{pdb_file.stem}_final.txt"
            print(f"\n  ✓ 最终共识序列 → {consensus_path}")
        except Exception as e:
            print(f"[ERROR] 处理 {pdb_file.name} 失败: {e}", file=sys.stderr)

    print("\n分析完成。")
