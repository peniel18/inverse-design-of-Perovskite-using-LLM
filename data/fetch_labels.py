# fetch_labels.py
import pandas as pd
from mp_api.client import MPRester
import os 
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("API_KEY")  # get free at materialsproject.org/api

print(API_KEY)
df = pd.read_csv("label_template.csv")

# Extract mp-ids (strip .cif extension)
df["mp_id"] = df["filename"].str.replace(".cif", "", regex=False)

print(f"Fetching labels for {len(df)} structures...")

with MPRester(API_KEY) as mpr:
    results = mpr.summary.search(
        material_ids=df["mp_id"].tolist(),
        fields=["material_id", "formation_energy_per_atom", "band_gap"]
    )

# Build lookup
lookup = {
    r.material_id: (r.formation_energy_per_atom, r.band_gap)
    for r in results
}

df["Ef"] = df["mp_id"].map(lambda x: lookup.get(x, (None, None))[0])
df["Eg"] = df["mp_id"].map(lambda x: lookup.get(x, (None, None))[1])

# Report
filled = df["Ef"].notna().sum()
print(f"Filled labels for {filled}/{len(df)} structures")
print(f"Still missing: {len(df) - filled}")

df.drop(columns=["mp_id"]).to_csv("label_template.csv", index=False)
print("Saved → label_template.csv")