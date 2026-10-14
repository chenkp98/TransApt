#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automatic classification script for nucleic acid complex PDB files.
Classifies PDB files in a directory into four categories:
  1) nucleic_acid_protein   Nucleic acid-protein complex
  2) nucleic_acid_ligand    Nucleic acid-ligand complex
  3) nucleic_acid_na        Nucleic acid-nucleic acid complex (e.g. DNA-RNA, DNA-DNA hybrid)
  4) nucleic_acid_complex   Nucleic acid multi-component complex (contains both protein and ligand, etc.)

Usage: python classify_na_complex.py <pdb_directory> [--move] [--outdir OUTPUT_DIR]
By default, files are copied to four subdirectories. Use --move to move them instead.
Files without nucleic acids are skipped.
"""
import os
import shutil
import argparse

DNA_RES = {"DA", "DG", "DC", "DT", "DI", "DU", "DN",
           "ADE", "GUA", "CYT", "THY", "URA"}
RNA_RES = {"A", "G", "C", "U", "I"}
MOD_NA = {"5BU", "PSU", "H2U", "H2T", "OMG", "1MA", "2MG", "7MG", "M2G",
          "5MC", "OMC", "A23", "U33", "G7M", "4SU", "S4U", "CBR", "C38",
          "6OG", "8OG", "FHU", "N6G", "N6A", "MA6", "6MA", "MIA", "MNU",
          "NMS", "U8U", "QUO", "YYG", "GDP", "GTP", "ATP", "ADP", "CTP",
          "UTP", "TTP", "AMP", "CMP", "GMP", "UMP", "TMP", "NAD", "FAD"}
AMINO_ACIDS = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
               "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
               "THR", "TRP", "TYR", "VAL", "MSE", "SEC", "PYL", "HYP", "PCA"}
EXCLUDE_HET = {"HOH", "WAT", "DOD", "H2O",
               "NA", "CL", "K", "MG", "CA", "ZN", "MN", "FE", "FE2", "CU",
               "CU1", "CO", "NI", "CD", "BR", "IOD", "F", "CA2", "HG", "PB",
               "SR", "BA", "CS", "LI", "RB", "AL", "CR", "V", "MO", "W",
               "PO4", "SO4", "NO3", "ACT", "ACY", "FMT"}


def classify_pdb(filepath):
    """Returns category name; returns None if no nucleic acid is present."""
    has_na = has_protein = has_small = False
    with open(filepath, "r", errors="ignore") as f:
        for line in f:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            resname = line[17:20].strip()          # PDB columns 18-20
            if resname in EXCLUDE_HET:
                continue
            if resname in DNA_RES or resname in RNA_RES or resname in MOD_NA:
                has_na = True
            elif resname in AMINO_ACIDS:
                has_protein = True
            elif line.startswith("HETATM"):
                has_small = True
    if not has_na:
        return None
    if has_protein and has_small:
        return "nucleic_acid_complex"
    if has_protein:
        return "nucleic_acid_protein"
    if has_small:
        return "nucleic_acid_ligand"
    return "nucleic_acid_na"


FOLDER_DESC = {
    "nucleic_acid_protein": "Nucleic Acid-Protein Complex",
    "nucleic_acid_ligand": "Nucleic Acid-Ligand Complex",
    "nucleic_acid_na": "Nucleic Acid-Nucleic Acid Complex",
    "nucleic_acid_complex": "Nucleic Acid Multi-Component Complex",
}


def main():
    ap = argparse.ArgumentParser(description="Nucleic Acid Complex PDB Classifier")
    ap.add_argument("folder", help="Directory containing PDB files")
    ap.add_argument("--outdir", default=None, help="Output directory (default: subdirectories inside input folder)")
    ap.add_argument("--move", action="store_true", help="Move files instead of copying")
    args = ap.parse_args()
    src = os.path.abspath(args.folder)
    dst_root = os.path.abspath(args.outdir) if args.outdir else src

    for cat in FOLDER_DESC.keys():
        os.makedirs(os.path.join(dst_root, cat), exist_ok=True)

    stats = {c: 0 for c in FOLDER_DESC}
    stats["_no_na"] = 0
    for fn in sorted(os.listdir(src)):
        if not fn.lower().endswith((".pdb", ".ent")):
            continue
        fp = os.path.join(src, fn)
        if not os.path.isfile(fp):
            continue
        cat = classify_pdb(fp)
        if cat is None:
            stats["_no_na"] += 1
            print(f"[SKIP - No NA] {fn}")
            continue
        dst = os.path.join(dst_root, cat, fn)
        (shutil.move if args.move else shutil.copy2)(fp, dst)
        stats[cat] += 1
        print(f"[{FOLDER_DESC[cat]}] {fn}")

    print("\n===== Statistics =====")
    for cat, desc in FOLDER_DESC.items():
        print(f"{desc} ({cat}): {stats[cat]}")
    print(f"No Nucleic Acid (Unprocessed): {stats['_no_na']}")


if __name__ == "__main__":
    main()