import sys
import torch
from pathlib import Path

from .model import BridgeAPT
from .generate import generate_sequence
from .compute.pdb2pt import read_pdb


def _get_default_weights_path() -> Path:
    try:
        if sys.version_info >= (3, 9):
            from importlib.resources import files
            return Path(str(files("bridgeapt").joinpath("weights/model_zhilian.pth")))
        else:
            import importlib.resources as pkg_resources
            ref = pkg_resources.files("bridgeapt") / "weights" / "model_zhilian.pth"
            return Path(str(ref))
    except Exception:
        return Path(__file__).parent / "weights" / "model_zhilian.pth"


class Runner:
    """End-to-end inference: PDB file → nucleic acid sequence list."""

    def __init__(self, model_path: str = None, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        weights_path = Path(model_path) if model_path else _get_default_weights_path()

        if not weights_path.exists():
            raise FileNotFoundError(
                f"Weights file not found: {weights_path}\n"
                "Reinstall the package or specify a path with --model-path."
            )

        try:
            checkpoint = torch.load(str(weights_path), map_location=self.device)
            self.model = BridgeAPT()
            self.model.load_state_dict(checkpoint)
            self.model.to(self.device)
            self.model.eval()
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to load weights: {e}\nEnsure the weights match the model version."
            ) from e

    def preprocess(self, pdb_path, max_len: int = 100):
        from .compute.extract_coord import extra_six_coord
        _, seq = extra_six_coord(str(pdb_path))
        actual_len = min(len(seq), max_len)

        coords, angles = read_pdb(str(pdb_path), max_len=max_len)
        coords = coords.reshape(coords.shape[0], max_len, -1).to(dtype=torch.float32)
        angles = torch.nan_to_num(angles, nan=0.0).to(dtype=torch.float32)
        return coords, angles, actual_len

    def run(
        self,
        pdb_path,
        num_sequences: int = 1000,
        temperature: float = 1.0,
        length: int = None,
    ):
        coords, scalar_features, actual_len = self.preprocess(pdb_path)

        use_len = length if length is not None else actual_len
        coords = coords[:, :use_len, :].to(self.device)
        scalar_features = scalar_features[:, :use_len, :].to(self.device)

        sequences = []
        try:
            for _ in range(num_sequences):
                seq = generate_sequence(
                    self.model, coords, scalar_features, temperature=temperature
                )[0]
                sequences.append(seq)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and self.device.type == "cuda":
                print("[WARNING] CUDA out of memory, falling back to CPU", file=sys.stderr)
                torch.cuda.empty_cache()
                self.model.to("cpu")
                self.device = torch.device("cpu")
                coords = coords.cpu()
                scalar_features = scalar_features.cpu()
                for _ in range(num_sequences - len(sequences)):
                    seq = generate_sequence(
                        self.model, coords, scalar_features, temperature=temperature
                    )[0]
                    sequences.append(seq)
            else:
                raise

        return sequences

    def run_with_analysis(
        self,
        pdb_path,
        output_dir,
        num_sequences: int = 1000,
        temperature: float = 1.0,
        length: int = None,
        features: list = None,
        n_clusters: int = 5,
        methods: list = None,
        layer: int = 7,
    ) -> dict:
        from .analysis import extract_shape_features, cluster_and_filter, DEFAULT_FEATURES
        from .compute.extract_coord import extra_six_coord

        if features is None:
            features = DEFAULT_FEATURES

        pdb_path = Path(pdb_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Generate sequences
        print(f"[1/3] Generating sequences ({num_sequences}) ...")
        gen_sequences = self.run(
            pdb_path, num_sequences=num_sequences,
            temperature=temperature, length=length,
        )

        # Step 2: Extract reference sequence from PDB
        _, ref_seq_residues = extra_six_coord(str(pdb_path))
        _mapping = {"DA": "A", "A": "A", "DG": "G", "G": "G",
                    "DC": "C", "C": "C", "DT": "T", "T": "T", "U": "T"}
        ref_seq = "".join(_mapping.get(r, "N") for r in ref_seq_residues)
        if length is not None:
            ref_seq = ref_seq[:length]
        ref_sequences = [ref_seq]

        # Step 3: Extract shape features
        print(f"[2/3] Extracting shape features ({len(features)} features) ...")
        ref_features = extract_shape_features(ref_sequences, features=features, layer=layer)
        gen_features = extract_shape_features(gen_sequences, features=features, layer=layer)

        # Step 4: Cluster and filter
        _methods = methods or ["kmeans", "hierarchical", "gmm"]
        print(f"[3/3] Clustering and filtering (k={n_clusters}, methods: {_methods}) ...")
        prefix = pdb_path.stem
        results = cluster_and_filter(
            ref_features=ref_features,
            gen_features=gen_features,
            gen_sequences=gen_sequences,
            feature_names=features,
            output_dir=output_dir,
            n_clusters=n_clusters,
            prefix=prefix,
            methods=methods,
        )

        return results
