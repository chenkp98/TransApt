import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        prog="bridgeapt-run",
        description="BridgeAPT: Generate nucleic acid sequences from a PDB structure file",
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Input PDB file or directory containing .pdb files")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory for generated sequence .txt files")
    parser.add_argument("--num-sequences", "-n", type=int, default=1000,
                        help="Number of sequences to generate per PDB (default: 1000)")
    parser.add_argument("--temperature", "-t", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0)")
    parser.add_argument("--length", "-l", type=int, default=None,
                        help="Sequence length (default: auto-detect from PDB residue count)")
    parser.add_argument("--model-path", "-m", default=None,
                        help="Custom weights file path (default: bundled pretrained weights)")
    parser.add_argument("--analyze", action="store_true",
                        help="Run shape feature extraction and clustering after generation")
    parser.add_argument("--clusters", "-k", type=int, default=5,
                        help="Number of clusters for --analyze mode (default: 5)")
    parser.add_argument("--layer", type=int, default=7,
                        help="AptShape flanking layer 0-7 for --analyze mode (default: 7)")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input path does not exist: {input_path}", file=sys.stderr)
        sys.exit(2)

    if input_path.is_file():
        if input_path.suffix.lower() != ".pdb":
            print(f"[ERROR] Input file is not a .pdb file: {input_path}", file=sys.stderr)
            sys.exit(2)
        pdb_files = [input_path]
    else:
        pdb_files = sorted(input_path.glob("*.pdb"))
        if not pdb_files:
            print(f"[WARNING] No .pdb files found in: {input_path}", file=sys.stderr)
            sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    from .runner import Runner

    try:
        runner = Runner(model_path=args.model_path)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    success_count = 0
    fail_count = 0

    for pdb_file in pdb_files:
        out_file = output_dir / (pdb_file.stem + ".txt")
        print(f"Processing: {pdb_file.name}")
        try:
            if args.analyze:
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
                print(f"  ✓ Done. Consensus sequences: {n_consensus} → {consensus_path}")
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
                print(f"  ✓ Generated {len(sequences)} sequences → {out_file}")
            success_count += 1
        except Exception as e:
            print(f"[ERROR] Failed to process {pdb_file.name}: {e}", file=sys.stderr)
            fail_count += 1

    print(f"\nDone: {success_count} succeeded, {fail_count} failed")
    if fail_count > 0 and success_count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()


def analyze():
    """bridgeapt-analyze: Generate sequences, extract shape features, and cluster."""
    parser = argparse.ArgumentParser(
        prog="bridgeapt-analyze",
        description="BridgeAPT: Generate → Shape features → Clustering → Filter",
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Input PDB file or directory containing .pdb files")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory for plots and filtered sequences")
    parser.add_argument("--num-sequences", "-n", type=int, default=1000,
                        help="Number of sequences to generate per PDB (default: 1000)")
    parser.add_argument("--temperature", "-t", type=float, default=1.0,
                        help="Sampling temperature (default: 1.0)")
    parser.add_argument("--length", "-l", type=int, default=None,
                        help="Sequence length (default: auto-detect from PDB residue count)")
    parser.add_argument("--clusters", "-k", type=int, default=5,
                        help="Number of clusters (default: 5)")
    parser.add_argument("--features", "-f", nargs="+",
                        default=["MGW", "Roll", "HelT", "ProT", "Shift", "Slide", "Rise", "Tilt"],
                        help="Shape features to extract (default: MGW Roll HelT ProT Shift Slide Rise Tilt)")
    parser.add_argument("--methods", nargs="+", default=["kmeans", "hierarchical", "gmm"],
                        help="Clustering methods (default: kmeans hierarchical gmm)")
    parser.add_argument("--layer", type=int, default=7,
                        help="AptShape flanking layer 0-7 (default: 7)")
    parser.add_argument("--model-path", "-m", default=None,
                        help="Custom weights file path")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input path does not exist: {input_path}", file=sys.stderr)
        sys.exit(2)

    pdb_files = [input_path] if input_path.is_file() else sorted(input_path.glob("*.pdb"))
    if not pdb_files:
        print(f"[WARNING] No .pdb files found in: {input_path}", file=sys.stderr)
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
        print(f"Processing: {pdb_file.name}")
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
            print(f"\n  ✓ Final consensus sequences → {consensus_path}")
        except Exception as e:
            print(f"[ERROR] Failed to process {pdb_file.name}: {e}", file=sys.stderr)

    print("\nAnalysis complete.")
