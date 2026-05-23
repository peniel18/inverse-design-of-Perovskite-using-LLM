"""
Step 2 — Preprocessing Pipeline for Custom CIF Files
=======================================================
Converts your cleaned CIF files + label CSV into batched PyTorch tensors
ready for CTGNN training.

Key differences vs. a standard dataset pipeline
-------------------------------------------------
  - Structure cleaner handles disordered sites, supercells, and
    structures with partial occupancy that pymatgen can auto-resolve
  - Flexible label CSV reader: tolerates extra columns, different
    column name variants (e_form / Ef / formation_energy / etc.)
  - Per-crystal graph cache: expensive pymatgen parsing is done once;
    subsequent runs load from .pt files in milliseconds
  - Detailed warnings for unusual bond lengths and isolated atoms

Usage
-----
    from preprocessing_custom import CIFGraphDataset, split_dataset, collate_fn
    from torch.utils.data import DataLoader

    dataset = CIFGraphDataset(
        cif_dir   = "my_cifs/",
        label_csv = "my_labels.csv",
        cache_dir = "graph_cache/",
    )

    train_ds, val_ds, test_ds = split_dataset(dataset)
    loader = DataLoader(train_ds, batch_size=32,
                        shuffle=True, collate_fn=collate_fn)
"""

import os
import json
import math
import warnings
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset, DataLoader
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

from pymatgen.core import Structure, Element
from pymatgen.core.periodic_table import Element as PMGElement



ATOM_FEA_DIM = 92   # CGCNN-compatible, expected by CTGNN

# Column name variants accepted in the label CSV
EF_ALIASES = ["Ef", "ef", "E_f", "e_f", "formation_energy",
               "e_form", "formation_energy_per_atom", "delta_e"]
EG_ALIASES = ["Eg", "eg", "E_g", "e_g", "bandgap", "band_gap",
               "band_gap_energy", "gap", "Egap"]
FN_ALIASES = ["filename", "file", "name", "cif", "id", "structure_id",
               "material_id", "cif_file"]



def _onehot(value, allowed: list) -> List[int]:
    vec = [0] * (len(allowed) + 1)
    try:
        vec[allowed.index(value)] = 1
    except ValueError:
        vec[-1] = 1
    return vec


def build_atom_features(symbol: str) -> Tensor:
    """
    92-dimensional atom feature vector for a given element symbol.
    Encodes: group, period, electronegativity bucket, valence electrons.
    Unknown elements fall back to a zero vector with the 'unknown' slot set.
    """
    try:
        el     = PMGElement(symbol)
        group  = int(el.group)
        period = int(el.row)
        en_raw = el.X

        # Discretise electronegativity into 10 equal bins over [0.7, 4.0]
        en_bins = np.linspace(0.7, 4.0, 11).tolist()
        if en_raw is None:
            en_bucket = None
        else:
            en_bucket = float(en_bins[
                min(int((en_raw - 0.7) / (4.0 - 0.7) * 10), 9)
            ])

        try:
            val_e = int(el.common_oxidation_states[0]) \
                    if el.common_oxidation_states else 0
            val_e = max(0, min(12, val_e))
        except Exception:
            val_e = 0

    except Exception:
        group, period, en_bucket, val_e = 1, 1, None, 0

    feat = (
        _onehot(group,     list(range(1, 19)))          # 19 dims
        + _onehot(period,  list(range(1, 10)))           # 10 dims
        + _onehot(en_bucket, np.linspace(0.7, 4.0, 11).tolist())  # 12 dims
        + _onehot(val_e,   list(range(0, 13)))           # 14 dims
        + [0] * (ATOM_FEA_DIM - 55)                      # pad to 92
    )
    feat = feat[:ATOM_FEA_DIM]
    return torch.tensor(feat, dtype=torch.float32)


def clean_structure(structure: Structure,
                    primitive: bool = True) -> Optional[Structure]:
    """
    Apply standard cleaning steps to a custom CIF structure.

    Steps applied (in order):
      1. Convert to primitive cell (reduces redundant atoms in supercells)
      2. Resolve disorder — if the structure has partial occupancy, attempt
         automatic ordering via pymatgen's transformation
      3. Remove sites with occupancy < 0.5  (very rare edge case)
      4. Validate: at least 2 atoms and no site within 0.5 Å of another

    Args:
        structure:  raw pymatgen Structure from CIF
        primitive:  if True, reduce to primitive cell (recommended)

    Returns:
        Cleaned Structure, or None if the structure cannot be salvaged
    """
    try:
        # Step 1: primitive cell
        if primitive:
            prim = structure.get_primitive_structure(tolerance=0.25)
            if len(prim) >= 2:
                structure = prim

        # Step 2: handle disorder
        if not structure.is_ordered:
            try:
                from pymatgen.transformations.standard_transformations import (
                    OrderDisorderedStructureTransformation,
                )
                t = OrderDisorderedStructureTransformation(algo=0)
                structure = t.apply_transformation(structure)
            except Exception:
                # Remove low-occupancy sites as fallback
                to_remove = [i for i, site in enumerate(structure)
                             if site.species.total_occupancy < 0.5]
                if to_remove:
                    structure.remove_sites(to_remove)

        # Step 3: minimum atom check
        if len(structure) < 2:
            return None

        # Step 4: check for unphysically close atoms (< 0.5 Å)
        dist_matrix = structure.distance_matrix
        np.fill_diagonal(dist_matrix, np.inf)
        if dist_matrix.min() < 0.5:
            return None

        return structure

    except Exception as e:
        return None


def build_graph(structure: Structure,
                cutoff:        float = 8.0,
                max_neighbors: int   = 12,
                min_neighbors: int   = 1,
                ) -> Optional[Dict[str, Tensor]]:
    """
    Build a crystal graph from a pymatgen Structure.

      Nodes  = atoms          (features: 92-dim one-hot)
      Edges  = bonds          (features: RBF distance + angular direction)

    Args:
        structure:     cleaned pymatgen Structure
        cutoff:        max bond distance in Å (8 Å matches paper)
        max_neighbors: keep only the N closest neighbours per atom
        min_neighbors: warn if any atom has fewer than this many neighbours

    Returns:
        dict with keys: atom_fea [N,92], edge_index [2,E],
                        dist [E], bond_vec [E,3]
        or None if graph cannot be built
    """
    N = len(structure)

    # ── Atom features ────────────────────────────────────────────────
    atom_fea = torch.stack([
        build_atom_features(str(site.specie.symbol))
        for site in structure
    ])   # [N, 92]

    # ── Neighbour search ─────────────────────────────────────────────
    try:
        all_nbrs = structure.get_all_neighbors(cutoff, include_index=True)
    except Exception:
        return None

    src_list, dst_list, dist_list, vec_list = [], [], [], []
    isolated_atoms = []

    for i, nbrs in enumerate(all_nbrs):
        # Sort by distance; keep max_neighbors closest
        nbrs_sorted = sorted(nbrs, key=lambda x: x.nn_distance)[:max_neighbors]

        if len(nbrs_sorted) < min_neighbors:
            isolated_atoms.append(i)

        for nbr in nbrs_sorted:
            j   = nbr.index
            d   = float(nbr.nn_distance)
            vec = np.array(nbr.coords) - np.array(structure[i].coords)
            unit = vec / (np.linalg.norm(vec) + 1e-8)

            src_list.append(i);  dst_list.append(j)
            dist_list.append(d); vec_list.append(unit.tolist())

    if isolated_atoms:
        print(f"  [WARN] {len(isolated_atoms)} atom(s) have fewer than "
              f"{min_neighbors} neighbour(s) within {cutoff} Å — "
              f"consider increasing --cutoff")

    if len(src_list) == 0:
        return None

    return {
        "atom_fea":   atom_fea,
        "edge_index": torch.tensor([src_list, dst_list], dtype=torch.long),
        "dist":       torch.tensor(dist_list,            dtype=torch.float32),
        "bond_vec":   torch.tensor(vec_list,             dtype=torch.float32),
    }



def load_label_csv(csv_path: str) -> pd.DataFrame:
    """
    Load a label CSV and normalise column names to [filename, Ef, Eg].

    Accepts many common column name variants — see EF_ALIASES / EG_ALIASES
    at the top of this file. Prints which columns were found.

    Raises:
        ValueError if required columns cannot be identified.
    """
    df = pd.read_csv(csv_path)
    cols = {c.lower().strip(): c for c in df.columns}

    def _find(aliases, role):
        for alias in aliases:
            if alias.lower() in cols:
                return cols[alias.lower()]
        raise ValueError(
            f"Could not find '{role}' column in {csv_path}.\n"
            f"  Columns found: {list(df.columns)}\n"
            f"  Accepted names: {aliases}"
        )

    fn_col = _find(FN_ALIASES, "filename")
    ef_col = _find(EF_ALIASES, "formation energy (Ef)")
    eg_col = _find(EG_ALIASES, "bandgap (Eg)")

    print(f"  Label CSV columns detected:")
    print(f"    filename → '{fn_col}'")
    print(f"    Ef       → '{ef_col}'")
    print(f"    Eg       → '{eg_col}'")

    df = df.rename(columns={fn_col: "filename", ef_col: "Ef", eg_col: "Eg"})
    df["filename"] = df["filename"].astype(str).str.strip()

    # Drop rows with missing labels
    before = len(df)
    df = df.dropna(subset=["Ef", "Eg"])
    if len(df) < before:
        print(f"  [WARN] Dropped {before - len(df)} rows with missing Ef/Eg values.")

    return df[["filename", "Ef", "Eg"]]




class CIFGraphDataset(Dataset):
    """
    Dataset for custom CIF files with formation energy + bandgap labels.

    Each __getitem__ call:
      1. Checks the graph cache (.pt file)  →  returns immediately if found
      2. Parses the CIF with pymatgen
      3. Cleans the structure (primitive cell, disorder resolution)
      4. Builds the crystal graph
      5. Normalises labels (z-score)
      6. Caches result to disk

    Args:
        cif_dir:          directory containing your .cif files
        label_csv:        CSV with [filename, Ef, Eg] columns
                          (flexible naming — see load_label_csv)
        cutoff:           neighbour search cutoff in Å
        max_neighbors:    max neighbours per atom
        cache_dir:        graph cache directory  (strongly recommended —
                          saves hours on large datasets)
        normalize_labels: z-score normalise Ef and Eg before training
        primitive:        convert structures to primitive cell
        verbose:          print per-crystal warnings
    """

    def __init__(self,
                 cif_dir:          str,
                 label_csv:        str,
                 cutoff:           float = 8.0,
                 max_neighbors:    int   = 12,
                 cache_dir:        Optional[str] = "graph_cache",
                 normalize_labels: bool  = True,
                 primitive:        bool  = True,
                 verbose:          bool  = False):

        self.cif_dir       = cif_dir
        self.cutoff        = cutoff
        self.max_neighbors = max_neighbors
        self.cache_dir     = cache_dir
        self.normalize     = normalize_labels
        self.primitive     = primitive
        self.verbose       = verbose

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        # ── Load labels ───────────────────────────────────────────────
        print(f"\nLoading labels from '{label_csv}' ...")
        df = load_label_csv(label_csv)

        # Build filename → (Ef, Eg) lookup
        # Normalise key: strip path, strip .cif extension
        def _key(s: str) -> str:
            return os.path.splitext(os.path.basename(s.strip()))[0]

        self.label_map: Dict[str, Tuple[float, float]] = {
            _key(row["filename"]): (float(row["Ef"]), float(row["Eg"]))
            for _, row in df.iterrows()
        }

        # ── Match CIF files to labels ─────────────────────────────────
        all_cifs = sorted([f for f in os.listdir(cif_dir) if f.endswith(".cif")])
        self.samples: List[Tuple[str, str]] = []   # (full_path, key)
        skipped = []

        for fname in all_cifs:
            key = os.path.splitext(fname)[0]
            if key in self.label_map:
                self.samples.append((os.path.join(cif_dir, fname), key))
            else:
                skipped.append(fname)

        print(f"\n  Matched   : {len(self.samples)} CIFs with labels")
        if skipped:
            print(f"  Skipped   : {len(skipped)} CIFs (no matching label row)")
            if verbose and len(skipped) <= 10:
                for f in skipped:
                    print(f"    {f}")

        if len(self.samples) == 0:
            raise RuntimeError(
                "No CIF files matched any label in the CSV.\n"
                "Check that your CSV filename column matches your .cif filenames."
            )

        # ── Label statistics for normalisation ────────────────────────
        ef_vals = np.array([self.label_map[k][0] for _, k in self.samples])
        eg_vals = np.array([self.label_map[k][1] for _, k in self.samples])

        self.ef_mean = float(ef_vals.mean());  self.ef_std = float(ef_vals.std()) + 1e-8
        self.eg_mean = float(eg_vals.mean());  self.eg_std = float(eg_vals.std()) + 1e-8

        self.label_stats = {
            "Ef_mean": self.ef_mean, "Ef_std": self.ef_std,
            "Eg_mean": self.eg_mean, "Eg_std": self.eg_std,
        }

        print(f"\n  Label statistics (raw):")
        print(f"    Ef  mean={self.ef_mean:.4f}  std={self.ef_std:.4f} eV/atom")
        print(f"    Eg  mean={self.eg_mean:.4f}  std={self.eg_std:.4f} eV")
        print(f"    Ef  range [{ef_vals.min():.3f},  {ef_vals.max():.3f}]")
        print(f"    Eg  range [{eg_vals.min():.3f},  {eg_vals.max():.3f}]\n")

        # Track failed samples at runtime (not pre-computed to avoid slow init)
        self._failed: List[str] = []

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        cif_path, key = self.samples[idx]

        # ── Cache hit ────────────────────────────────────────────────
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"{key}.pt")
            if os.path.exists(cache_path):
                try:
                    return torch.load(cache_path, weights_only=True)
                except Exception:
                    pass   # corrupted cache — rebuild

        # ── Parse + clean ────────────────────────────────────────────
        try:
            raw_structure = Structure.from_file(cif_path)
        except Exception as e:
            if self.verbose:
                print(f"  [FAIL] Parse error for {key}: {e}")
            self._failed.append(key)
            return self._dummy_sample()

        structure = clean_structure(raw_structure, primitive=self.primitive)
        if structure is None:
            if self.verbose:
                print(f"  [FAIL] Cleaning failed for {key}")
            self._failed.append(key)
            return self._dummy_sample()

        # ── Build graph ───────────────────────────────────────────────
        graph = build_graph(structure, self.cutoff, self.max_neighbors)
        if graph is None:
            if self.verbose:
                print(f"  [FAIL] Graph build failed for {key}")
            self._failed.append(key)
            return self._dummy_sample()

        # ── Labels ───────────────────────────────────────────────────
        ef_raw, eg_raw = self.label_map[key]
        ef = (ef_raw - self.ef_mean) / self.ef_std if self.normalize else ef_raw
        eg = (eg_raw - self.eg_mean) / self.eg_std if self.normalize else eg_raw

        graph["y"]     = torch.tensor([[ef, eg]],         dtype=torch.float32)
        graph["y_raw"] = torch.tensor([[ef_raw, eg_raw]], dtype=torch.float32)

        # ── Cache write ───────────────────────────────────────────────
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"{key}.pt")
            torch.save(graph, cache_path)

        return graph

    # ------------------------------------------------------------------
    def _dummy_sample(self) -> Dict[str, Tensor]:
        return {
            "atom_fea":   torch.zeros(2, ATOM_FEA_DIM),
            "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            "dist":       torch.ones(2),
            "bond_vec":   torch.zeros(2, 3),
            "y":          torch.zeros(1, 2),
            "y_raw":      torch.zeros(1, 2),
            "_dummy":     torch.tensor(True),
        }

    # ------------------------------------------------------------------
    def denormalize(self, y_norm: Tensor) -> Tensor:
        """Convert normalised model outputs back to eV/atom and eV."""
        ef = y_norm[:, 0] * self.ef_std + self.ef_mean
        eg = y_norm[:, 1] * self.eg_std + self.eg_mean
        return torch.stack([ef, eg], dim=-1)

    # ------------------------------------------------------------------
    def save_label_stats(self, path: str = "label_stats.json"):
        with open(path, "w") as f:
            json.dump(self.label_stats, f, indent=2)
        print(f"Label normalisation stats saved → {path}")

    # ------------------------------------------------------------------
    def print_summary(self, n_inspect: int = 5):
        """Print a diagnostic summary of the dataset."""
        print("\n" + "="*55)
        print("  DATASET SUMMARY")
        print("="*55)
        print(f"  Total matched samples : {len(self.samples)}")
        print(f"  Cutoff / max_neighbors: {self.cutoff} Å  /  {self.max_neighbors}")

        n_atoms_list, n_edges_list = [], []
        print(f"\n  {'#':<5} {'Key':<28} {'N':<6} {'E':<7} Ef_raw   Eg_raw")
        print("  " + "-" * 58)

        for i, (_, key) in enumerate(self.samples[:n_inspect]):
            g  = self[i]
            N  = g["atom_fea"].size(0)
            E  = g["dist"].size(0)
            ef = g["y_raw"][0, 0].item()
            eg = g["y_raw"][0, 1].item()
            n_atoms_list.append(N);  n_edges_list.append(E)
            print(f"  {i:<5} {key[:27]:<28} {N:<6} {E:<7} {ef:.4f}   {eg:.4f}")

        # Quick stats over up to 200 samples
        for i in range(n_inspect, min(200, len(self.samples))):
            g = self[i]
            n_atoms_list.append(g["atom_fea"].size(0))
            n_edges_list.append(g["dist"].size(0))

        na = np.array(n_atoms_list); ne = np.array(n_edges_list)
        print(f"\n  Atoms/crystal  min={na.min()}  max={na.max()}  mean={na.mean():.1f}")
        print(f"  Edges/crystal  min={ne.min()}  max={ne.max()}  mean={ne.mean():.1f}")
        if self._failed:
            print(f"\n  [WARN] {len(self._failed)} samples failed during loading.")
        print("="*55 + "\n")




def split_dataset(
    dataset,
    train_frac: float = 0.60,
    val_frac:   float = 0.20,
    seed:       int   = 42,
) -> Tuple[Subset, Subset, Subset]:
    """
    Reproducible 60 / 20 / 20 stratified-by-Ef split.

    Stratification ensures train/val/test have similar Ef distributions,
    which is especially important for custom datasets that may have
    compositional clustering.

    Returns:
        (train_subset, val_subset, test_subset)
    """
    n   = len(dataset)
    rng = np.random.default_rng(seed)

    # For small datasets (< 15 samples) use simple random split
    if n < 15:
        idx     = rng.permutation(n).tolist()
        n_train = max(1, int(n * train_frac))
        n_val   = max(1, int(n * val_frac))
        train_idx = idx[:n_train]
        val_idx   = idx[n_train: n_train + n_val]
        test_idx  = idx[n_train + n_val:] or idx[-1:]  # ensure at least 1
        print(f"Split (random, small dataset) →  "
              f"train: {len(train_idx)}  |  val: {len(val_idx)}  |  test: {len(test_idx)}")
    else:
        # Stratify by Ef quantile bins
        ef_vals = np.array([dataset.label_map[k][0] for _, k in dataset.samples])
        n_bins  = min(10, n // 5)
        bins    = np.quantile(ef_vals, np.linspace(0, 1, n_bins + 1))
        bin_ids = np.digitize(ef_vals, bins[1:-1])

        train_idx, val_idx, test_idx = [], [], []
        for b in range(n_bins):
            b_idx = np.where(bin_ids == b)[0].tolist()
            rng.shuffle(b_idx)
            n_b     = len(b_idx)
            n_train = max(1, int(n_b * train_frac))
            n_val   = max(0, int(n_b * val_frac))
            train_idx += b_idx[:n_train]
            val_idx   += b_idx[n_train: n_train + n_val]
            test_idx  += b_idx[n_train + n_val:]

        print(f"Split (stratified by Ef) →  "
              f"train: {len(train_idx)}  |  val: {len(val_idx)}  |  test: {len(test_idx)}")

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — COLLATE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def collate_fn(samples: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    """
    Batch variable-size crystal graphs into a single dict.

    Offsets edge_index by cumulative atom count so all indices are globally
    consistent, and builds a `batch` tensor mapping each atom to its crystal.

    Skips dummy samples (failed CIF parses) silently.
    """
    # Filter out dummy samples
    samples = [s for s in samples if "_dummy" not in s]
    if len(samples) == 0:
        raise RuntimeError("All samples in this batch failed to load.")

    atom_feas, edge_indices, dists, bond_vecs, ys, y_raws, batch_ids = \
        [], [], [], [], [], [], []
    atom_offset = 0

    for ci, s in enumerate(samples):
        N = s["atom_fea"].size(0)
        atom_feas.append(s["atom_fea"])
        edge_indices.append(s["edge_index"] + atom_offset)
        dists.append(s["dist"])
        bond_vecs.append(s["bond_vec"])
        ys.append(s["y"])
        y_raws.append(s["y_raw"])
        batch_ids.append(torch.full((N,), ci, dtype=torch.long))
        atom_offset += N

    return {
        "atom_fea":   torch.cat(atom_feas,    dim=0),
        "edge_index": torch.cat(edge_indices, dim=1),
        "dist":       torch.cat(dists,        dim=0),
        "bond_vec":   torch.cat(bond_vecs,    dim=0),
        "batch":      torch.cat(batch_ids,    dim=0),
        "y":          torch.cat(ys,           dim=0),
        "y_raw":      torch.cat(y_raws,       dim=0),
    }



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess custom CIF files for CTGNN")
    parser.add_argument("--cif_dir",    required=True)
    parser.add_argument("--label_csv",  required=True)
    parser.add_argument("--cache_dir",  default="graph_cache")
    parser.add_argument("--cutoff",     type=float, default=8.0)
    parser.add_argument("--max_nbrs",   type=int,   default=12)
    parser.add_argument("--no_prim",    action="store_true",
                        help="Skip primitive cell conversion")
    parser.add_argument("--no_norm",    action="store_true",
                        help="Disable label normalisation")
    parser.add_argument("--stats_out",  default="label_stats.json")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    dataset = CIFGraphDataset(
        cif_dir          = args.cif_dir,
        label_csv        = args.label_csv,
        cutoff           = args.cutoff,
        max_neighbors    = args.max_nbrs,
        cache_dir        = args.cache_dir,
        normalize_labels = not args.no_norm,
        primitive        = not args.no_prim,
        verbose          = True,
    )

    dataset.print_summary(n_inspect=5)
    dataset.save_label_stats(args.stats_out)

    train_ds, val_ds, test_ds = split_dataset(dataset)

    loader = DataLoader(train_ds, batch_size=args.batch_size,
                        shuffle=True, collate_fn=collate_fn)
    batch  = next(iter(loader))

    print("Batch tensor shapes (verify these before training):")
    for k, v in batch.items():
        print(f"  {k:<15} {tuple(v.shape)}")
    print("\nPreprocessing complete. You are ready to train.")
