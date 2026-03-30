import torch
import numpy as np
from Bio.PDB import PDBParser


def extra_six_coord(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)
    coord_six = []
    sequence = []
    target_atoms = ["C4'", "C1'", "N1", "C2", "C5'", "O5'", "P"]

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname().strip() in ['DA', 'DC', 'DG', 'DT', 'A', 'G', 'C', 'T', 'U']:
                    sequence.append(residue.get_resname())
                    coords = [[0.0, 0.0, 0.0]] * 7
                    atom_coords = {atom.get_name(): atom.get_coord().tolist() for atom in residue}
                    for i, atom_name in enumerate(target_atoms):
                        if atom_name in atom_coords:
                            coords[i] = atom_coords[atom_name]
                    coord_six.append(coords)
    return np.array(coord_six), sequence


def residue_to_num(sequence):
    mapping = {'DA': 1, 'A': 1, 'DG': 2, 'G': 2, 'DC': 3, 'C': 3, 'DT': 4, 'T': 4, 'U': 4, 'X': 0}
    seq = [mapping.get(i, 0) for i in sequence]
    return torch.tensor(seq, dtype=torch.int64)


def pad_sequence(max_len, list_seqs):
    list_seqs_new = []
    for seq in list_seqs:
        padded_seq = torch.zeros(max_len, dtype=torch.int64)
        length = min(len(seq), max_len)
        padded_seq[:length] = seq[:length]
        list_seqs_new.append(padded_seq)
    return list_seqs_new


def pad_coords(max_len, coords):
    list_coords_new = []
    for coord in coords:
        padded = np.zeros((max_len, 7, 3))
        length = min(len(coord), max_len)
        padded[:length] = coord[:length]
        list_coords_new.append(padded)
    return torch.tensor(np.array(list_coords_new), dtype=torch.float32)
