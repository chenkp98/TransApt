import os
import glob
import numpy as np
from Bio.PDB import MMCIFParser, MMCIFIO, PDBParser, PDBIO, Select
from Bio.PDB import is_aa, is_nucleic
from scipy.spatial import cKDTree

def chain_type(chain):
    residue_cnt = 0
    protein_cnt = 0
    nucleic_cnt = 0
    for residue in chain:
        residue_cnt += 1
        if is_aa(residue, standard=True):
            protein_cnt += 1
        elif is_nucleic(residue):
            nucleic_cnt += 1
    if residue_cnt == 0:
        return "other"
    if protein_cnt / residue_cnt > 0.8:
        return "protein"
    if nucleic_cnt / residue_cnt > 0.8:
        return "nucleic"
    return "other"


class ChainSelect(Select):
    def __init__(self, chain_ids):
        self.chain_ids = set(chain_ids)
    def accept_chain(self, chain):
        return chain.get_id() in self.chain_ids

def have_contact_kdtree(chain1, chain2, cutoff=4.0):
    coords1 = [atom.coord for residue in chain1 for atom in residue if atom.element != 'H']
    coords2 = [atom.coord for residue in chain2 for atom in residue if atom.element != 'H']
    if not coords1 or not coords2:
        return False
    tree = cKDTree(coords2)
    neighbors = tree.query_ball_point(coords1, cutoff, return_sorted=False)
    return any(len(lst) > 0 for lst in neighbors)


def process_structure(file_path, output_dir, cutoff=4.0):
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".cif":
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure('complex', file_path)
        elif ext == ".pdb":
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('complex', file_path)
        else:
            print(f"Skipping unsupported format: {file_path}")
            return
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    model = structure[0]
    chains_info = {}
    for chain in model:
        cid = chain.get_id()
        chains_info[cid] = chain_type(chain)
    protein_chains = [cid for cid, typ in chains_info.items() if typ == "protein"]
    nucleic_chains = [cid for cid, typ in chains_info.items() if typ == "nucleic"]
    if not protein_chains or not nucleic_chains:
        return

    chain_map = {chain.get_id(): chain for chain in model}
    for prot_id in protein_chains:
        prot_chain = chain_map[prot_id]
        contact_nuc_ids = []
        for nuc_id in nucleic_chains:
            nuc_chain = chain_map[nuc_id]
            if have_contact_kdtree(prot_chain, nuc_chain, cutoff):
                contact_nuc_ids.append(nuc_id)
        if contact_nuc_ids:
            keep_ids = [prot_id] + contact_nuc_ids
            base = os.path.splitext(os.path.basename(file_path))[0]
            out_ext = ext
            out_name = f"{base}_{prot_id}_complex{out_ext}"
            out_path = os.path.join(output_dir, out_name)

            if out_ext == ".cif":
                io = MMCIFIO()
            else:
                io = PDBIO()
            io.set_structure(structure)
            io.save(out_path, ChainSelect(keep_ids))
            print(f"Created {out_path} (protein {prot_id} + nucleic {contact_nuc_ids})")

if __name__ == "__main__":
    INPUT_DIR = "./pdb"
    OUTPUT_DIR = "./pdb_cleaned"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    file_list = []
    file_list.extend(glob.glob(os.path.join(INPUT_DIR, "*.cif")))
    file_list.extend(glob.glob(os.path.join(INPUT_DIR, "*.pdb")))
    print(f"Found {len(file_list)} CIF/PDB files to process.")

    CONTACT_CUTOFF = 4.0
    for idx, struct_file in enumerate(file_list):
        if idx % 100 == 0:
            print(f"Processing: {idx}/{len(file_list)}")
        process_structure(struct_file, OUTPUT_DIR, CONTACT_CUTOFF)
    print("✅ All files processed!")
