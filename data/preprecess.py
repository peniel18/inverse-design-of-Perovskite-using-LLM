import os
import gzip
import pickle
from typing import List, Tuple
from utils.logger import logger
from pymatgen.core import Structure
from pymatgen.io.cif import CifParser, CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import numpy as np


def preprocess_cif(cif_path: str, decimal_places: int = 4):
    try:
        parser = CifParser(cif_path)
        structure = parser.parse_structures(primitive=True)[0]

        if structure.composition.num_atoms == 0:
            return None

        # Standardize symmetry
        sga = SpacegroupAnalyzer(structure, symprec=0.01)
        structure = sga.get_conventional_standard_structure()

        # Round lattice matrix
        rounded_lattice = np.round(structure.lattice.matrix, decimal_places)

        # Round fractional coordinates
        rounded_coords = np.round(structure.frac_coords, decimal_places)

        # Create NEW structure (correct way)
        structure = Structure(
            lattice=rounded_lattice,
            species=structure.species,
            coords=rounded_coords,
            coords_are_cartesian=False,
        )

        writer = CifWriter(structure)
        return writer.__str__()

    except Exception as e:
        logger.error(f"Error processing {cif_path}: {e}")
        print(f"Error processing {cif_path}: {e}")
        return None


def preprocess_folder(input_dir: str, output_file: str):
    processed_data: List[Tuple[str, str]] = []

    for fname in os.listdir(input_dir):
        if fname.endswith(".cif"):
            path = os.path.join(input_dir, fname)
            cif_id = os.path.splitext(fname)[0]

            try:
                cif_string = preprocess_cif(path)
                if cif_string:
                    processed_data.append((cif_id, cif_string))
            except Exception as e:
                logger.error(f"Error processing {fname}: {e}")
                print(f"Error processing {fname}: {e}")

    # save file in gzip compressed pickle format
    with gzip.open(output_file, "wb") as f:
        pickle.dump(processed_data, f)

    print(f"Saved {len(processed_data)} processed CIFs")


if __name__ == "__main__":
    input_directory = "./data/cif_files/"
    output_pickle = "./data/processed_data/processed_cifs.pkl.gz"
    preprocess_folder(input_directory, output_pickle)