from mp_api.client import MPRester
import json 
import pandas as pd
from typing import List, Dict
from dotenv import load_dotenv
import os 
load_dotenv()


API_KEY = os.getenv("API_KEY")


def save_data_as_cif(summaries: List[object], mpr: MPRester, folder_path: str)-> None:
    """
    Saves CIF files for given material summaries

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

            doc.structure.to(filename=file_path, fmt="cif")

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


if __name__ == "__main__": 
    with MPRester(API_KEY) as mpr:
        perovskies = get_data_from_mp(mpr=mpr)

    save_data_as_cif(
        summaries=perovskies, 
        mpr=mpr, 
        folder_path="data/cif_files"
    )
