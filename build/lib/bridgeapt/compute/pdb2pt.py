import torch
import numpy as np
import os
from .extract_coord import extra_six_coord, residue_to_num, pad_sequence, pad_coords
from .compute_dihedrals import compute_dihedrals


def read_pdb(pdb_path, max_len=100):
    """
    处理单个 PDB 文件或目录，返回 (coords, scalar_features)。

    pdb_path: 单个 .pdb 文件路径或包含 .pdb 文件的目录
    返回: (coords [1, max_len, 21], scalar_features [1, max_len, 6])
    """
    pdb_path = str(pdb_path)

    if os.path.isfile(pdb_path):
        pdb_files = [pdb_path]
    else:
        pdb_files = [
            os.path.join(pdb_path, f)
            for f in os.listdir(pdb_path)
            if f.endswith('.pdb')
        ]

    list_coords = []
    list_angles = []

    for fpath in pdb_files:
        coord_six, seq = extra_six_coord(fpath)
        if len(seq) == 0:
            continue

        angles = compute_dihedrals(coord_six)

        angles_padded = np.zeros((max_len, 6))
        length = min(angles.shape[0], max_len)
        angles_padded[:length] = angles[:length]

        list_angles.append(angles_padded)
        list_coords.append(coord_six)

    if not list_coords:
        raise ValueError(f"No valid nucleic acid residues found in: {pdb_path}")

    coords = pad_coords(max_len, list_coords)
    angles_all = torch.tensor(np.array(list_angles), dtype=torch.float32)

    return coords, angles_all
