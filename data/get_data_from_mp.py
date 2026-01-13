from mp_api.client import MPRester
import json 
import pandas as pd
from typing import List, Dict
from dotenv import load_dotenv
import os 
load_dotenv()


API_KEY = os.getenv("API_KEY")


def get_perovskite_by_space_group(mpr: MPRester) -> List[Dict]: 
    """
    Search Perovskite using their space group 
    """
    perovskite_space_groups = [221, 99 , 160,167, 123,71, 62, 140, 204, 59, 11, 2, 63, 127, 140, ]

    perovskites = []

    for space_group in perovskite_space_groups:
        docs = mpr.materials.summary.search(
            spacegroup_number=space_group,
             fields=[
                "material_id",
                "formula_pretty",
                "structure",
                "symmetry",
                "band_gap",
                "formation_energy_per_atom",
                "energy_above_hull",
                "is_stable",
                "theoretical",
                "nelements",
                "elements",
                "volume",
                "density"
            ]
        )

    for doc in docs: 
        if doc.nelements == 3:  
            perovskites.append({
                 "material_id": doc.material_id,
                    "formula": doc.formula_pretty,
                    "space_group": doc.symmetry.number,
                    "crystal_system": doc.symmetry.crystal_system,
                    "band_gap": doc.band_gap,
                    "formation_energy": doc.formation_energy_per_atom,
                    "energy_above_hull": doc.energy_above_hull,
                    "is_stable": doc.is_stable,
                    "volume": doc.volume,
                    "density": doc.density,
                    "structure": doc.structure
            })

        print(f"  Found {len([d for d in perovskites if d['space_group'] == perovskite_space_groups ])} candidates")

    return perovskites



def get_perovskites_by_formula(mpr: MPRester) -> List[Dict]:
    """  
    Search Perovskites using their chemical systems 
    """   
    perovskites = []
    common_systems = [
        "Ba-Ti-O", "Sr-Ti-O", "Ca-Ti-O", "Pb-Ti-O",
        "Ba-Zr-O", "Sr-Zr-O", "La-Fe-O", "La-Mn-O",
        "K-Nb-O", "K-Ta-O", "Na-Nb-O"
    ]

    for system in common_systems:
        # Use .search() instead of ._search()
        # Ensure fields match the current SummaryDoc schema
        docs = mpr.materials.summary.search(
            chemsys=system, 
            fields=[
                "material_id",
                "formula_pretty",
                "structure",
                "symmetry",
                "band_gap",
                "formation_energy_per_atom",
                "energy_above_hull",
                "is_stable",
                "theoretical",
                "nelements",
                "elements",
                "volume",
                "density"
            ]
        )

        for doc in docs:
            perovskites.append({
                "material_id": str(doc.material_id),
                "formula": doc.formula_pretty,
                "space_group": doc.symmetry.number,
                "crystal_system": str(doc.symmetry.crystal_system),
                "band_gap": doc.band_gap,
                "formation_energy": doc.formation_energy_per_atom,
                "energy_above_hull": doc.energy_above_hull,
                "is_stable": doc.is_stable,
                "volume": doc.volume,
                "density": doc.density,
                "structure": doc.structure.to(fmt="cif") 
            })

    return perovskites


def get_data_from_mp(mpr: MPRester):
    """
    

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


if __name__ == "__main__": 
    with MPRester(API_KEY) as mpr:
        perovskies = get_data_from_mp(mpr=mpr)

    # save as json
    #print(perovskies)
    print(len(perovskies))
    #with open("data/perovskites_by_space_group.json", "w") as f:
    #   json.dump(perovskites, f, indent=4)

    with open("data/perovskites_from_mp.json", "w") as f:
        json.dump([doc.dict() for doc in perovskies], f, indent=4)

    #print(perovskies)
    