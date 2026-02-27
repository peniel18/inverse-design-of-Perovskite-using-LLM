from mp_api.client import MPRester
from pymatgen.io.cif import CifWriter
import json 
import pandas as pd
from typing import List, Dict
from dotenv import load_dotenv
import os 
load_dotenv()


API_KEY = os.getenv("API_KEY")


def save_data_as_cif(summaries: List[object], mpr: MPRester, folder_path: str) -> None:
    """
    Saves CIF files with embedded material properties in CIF format

    Args:
        summaries (List[object]): List of material summaries
        mpr (MPRester): MPRester object to access materials project API
        folder_path (str): Path to folder where CIF files will be saved
    """
    
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for doc in summaries:
        material_id = doc.material_id

        if hasattr(doc, "structure") and doc.structure: 
            file_path = os.path.join(folder_path, f"{material_id}.cif")
            
            # First save the basic CIF
            doc.structure.to(filename=file_path, fmt="cif")
            
            # Now append properties in CIF format
            with open(file_path, 'a') as f:
                #f.write("\n# Materials Project Properties\n")
                f.write(f"_prop_material_id '{material_id}'\n")
                f.write(f"_prop_formula '{getattr(doc, 'formula_pretty', 'N/A')}'\n")
                
                # Band gap properties
                band_gap = getattr(doc, 'band_gap', None)
                if band_gap is not None:
                    f.write(f"_prop_band_gap {band_gap}\n")
                    f.write("_prop_band_gap_units 'eV'\n")
                    f.write("_prop_band_gap_method 'DFT'\n")
                    is_direct = getattr(doc, 'is_gap_direct', None)
                    if is_direct is not None:
                        f.write(f"_prop_band_gap_direct '{is_direct}'\n")
                
                # Electronic properties
                is_metal = getattr(doc, 'is_metal', None)
                if is_metal is not None:
                    f.write(f"_prop_is_metal '{is_metal}'\n")
                
                efermi = getattr(doc, 'efermi', None)
                if efermi is not None:
                    f.write(f"_prop_fermi_energy {efermi}\n")
                    f.write("_prop_fermi_energy_units 'eV'\n")
                
                cbm = getattr(doc, 'cbm', None)
                if cbm is not None:
                    f.write(f"_prop_cbm {cbm}\n")
                    f.write("_prop_cbm_units 'eV'\n")
                
                vbm = getattr(doc, 'vbm', None)
                if vbm is not None:
                    f.write(f"_prop_vbm {vbm}\n")
                    f.write("_prop_vbm_units 'eV'\n")
                
                # Magnetic properties
                is_magnetic = getattr(doc, 'is_magnetic', None)
                if is_magnetic is not None:
                    f.write(f"_prop_is_magnetic '{is_magnetic}'\n")
                
                total_mag = getattr(doc, 'total_magnetization', None)
                if total_mag is not None:
                    f.write(f"_prop_total_magnetization {total_mag}\n")
                    f.write("_prop_total_magnetization_units 'muB'\n")
                
                # Thermodynamic properties
                energy_above_hull = getattr(doc, 'energy_above_hull', None)
                if energy_above_hull is not None:
                    f.write(f"_prop_energy_above_hull {energy_above_hull}\n")
                    f.write("_prop_energy_above_hull_units 'eV/atom'\n")
                
                formation_energy = getattr(doc, 'formation_energy_per_atom', None)
                if formation_energy is not None:
                    f.write(f"_prop_formation_energy_per_atom {formation_energy}\n")
                    f.write("_prop_formation_energy_units 'eV/atom'\n")
                
                is_stable = getattr(doc, 'is_stable', None)
                if is_stable is not None:
                    f.write(f"_prop_is_stable '{is_stable}'\n")
                
                # Structural properties
                density = getattr(doc, 'density', None)
                if density is not None:
                    f.write(f"_prop_density {density}\n")
                    f.write("_prop_density_units 'g/cm3'\n")
                
                volume = getattr(doc, 'volume', None)
                if volume is not None:
                    f.write(f"_prop_volume {volume}\n")
                    f.write("_prop_volume_units 'angstrom3'\n")
                
                nsites = getattr(doc, 'nsites', None)
                if nsites is not None:
                    f.write(f"_prop_nsites {nsites}\n")
                
                # Symmetry properties
                if hasattr(doc, 'symmetry') and doc.symmetry:
                    crystal_system = getattr(doc.symmetry, 'crystal_system', None)
                    if crystal_system:
                        f.write(f"_prop_crystal_system '{crystal_system}'\n")
                    
                    space_group = getattr(doc.symmetry, 'symbol', None)
                    if space_group:
                        f.write(f"_prop_space_group '{space_group}'\n")
                    
                    space_group_num = getattr(doc.symmetry, 'number', None)
                    if space_group_num:
                        f.write(f"_prop_space_group_number {space_group_num}\n")
                
                # Dielectric properties (if available)
                if hasattr(doc, 'diel') and doc.diel:
                    total_diel = getattr(doc.diel, 'total', None)
                    if total_diel is not None:
                        f.write(f"_prop_dielectric_constant_total {total_diel}\n")
                    
                    electronic_diel = getattr(doc.diel, 'electronic', None)
                    if electronic_diel is not None:
                        f.write(f"_prop_dielectric_constant_electronic {electronic_diel}\n")
                    
                    ionic_diel = getattr(doc.diel, 'ionic', None)
                    if ionic_diel is not None:
                        f.write(f"_prop_dielectric_constant_ionic {ionic_diel}\n")
                
                # Elastic properties (if available)
                if hasattr(doc, 'elasticity') and doc.elasticity:
                    bulk_mod = getattr(doc.elasticity, 'k_vrh', None)
                    if bulk_mod is not None:
                        f.write(f"_prop_bulk_modulus {bulk_mod}\n")
                        f.write("_prop_bulk_modulus_units 'GPa'\n")
                    
                    shear_mod = getattr(doc.elasticity, 'g_vrh', None)
                    if shear_mod is not None:
                        f.write(f"_prop_shear_modulus {shear_mod}\n")
                        f.write("_prop_shear_modulus_units 'GPa'\n")
        else: 
            print(f"No structure found for {material_id}, skipping CIF save.")


def get_data_from_mp(mpr: MPRester)-> object:
    """
    Gets perovskites data from materials project platform 

    Args:
        mpr (MPRester): MPRester object to access materials project API

    Returns:
        List of perovskite material summaries
    """


    robocrys_docs = mpr.materials.robocrys.search(keywords=["perovskite"])
    robo_perov_mpids = [doc.material_id for doc in robocrys_docs]
    

    prov_docs = mpr.materials.provenance.search(
        fields=["material_id", "remarks", "tags"]
    )
    possible_perov = [
        doc.get("material_id") for doc in prov_docs
        if any("perovskite" in tag.lower() 
               for tag in (doc.get("tags", []) + doc.get("remarks", [])))
    ]
    

    all_perovskite_mpids = list(set(robo_perov_mpids).union(possible_perov))
    

    summaries = mpr.materials.summary.search(material_ids=all_perovskite_mpids)
    
    print(f"Total unique perovskites found: {len(all_perovskite_mpids)}")

    return summaries



def get_data_from_database(): 
    pass 



if __name__ == "__main__": 
    with MPRester(API_KEY) as mpr:
        perovskies = get_data_from_mp(mpr=mpr)

    save_data_as_cif(
        summaries=perovskies, 
        mpr=mpr, 
        folder_path="data/cif_files"
    )
