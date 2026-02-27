#!/usr/bin/env python
"""
Script to retrieve perovskite materials from the MPDS database.
Requires: pip install mpds_client
"""

import json
from mpds_client import MPDSDataRetrieval
import os
from dotenv import load_dotenv


api_key = os.getenv("API_KEY")
  

# Initialize the client
client = MPDSDataRetrieval(api_key=api_key) if api_key else MPDSDataRetrieval()

# Define search criteria for perovskites
search_query = {
    "classes": "perovskite",
    "props": "atomic structure"  # Request crystalline structures
}

# Specify which fields to retrieve
fields = {
    'S': [
        'entry',
        'phase_id',
        'chemical_formula',
        'cell_abc',
        'sg_n',
        'space_group',
        'basis_noneq',
        'els_noneq'
    ]
}

try:
    print("Retrieving perovskite structures from MPDS...")
    print(f"Search query: {search_query}\n")
    
    # Get data as a list
    data = client.get_data(
        search_query,
        fields=fields,
      #  pagesize=100  # Adjust as needed (10, 100, 500, or 1000)
    )
    
    print(f"Retrieved {len(data)} perovskite structures\n")
    print("=" * 80)
    
    # Display first few results
    for i, item in enumerate(data[:10], 1):  # Show first 10 results
        entry_id = item[0] if len(item) > 0 else "N/A"
        phase_id = item[1] if len(item) > 1 else "N/A"
        formula = item[2] if len(item) > 2 else "N/A"
        cell_params = item[3] if len(item) > 3 else "N/A"
        sg_number = item[4] if len(item) > 4 else "N/A"
        sg_symbol = item[5] if len(item) > 5 else "N/A"
        
        print(f"\n{i}. Entry ID: {entry_id}")
        print(f"   Phase ID: {phase_id}")
        print(f"   Formula: {formula}")
        print(f"   Cell parameters: {cell_params}")
        print(f"   Space group: {sg_number} ({sg_symbol})")
        print(f"   URL: https://mpds.io/entry/{entry_id}")
    
    if len(data) > 10:
        print(f"\n... and {len(data) - 10} more results")
    
    # Optional: Get data as a pandas DataFrame for easier analysis
    print("\n" + "=" * 80)
    print("\nRetrieving as DataFrame for analysis...")
    
    df = client.get_dataframe(
        search_query,
        fields=fields,
        pagesize=100
    )
    
    if not df.empty:
        print(f"\nDataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst few rows:")
        print(df.head())
        
        # Save to CSV
        output_file = "perovskites_mpds.csv"
        df.to_csv(output_file, index=False)
        print(f"\nData saved to {output_file}")
    
    # Optional: Export structures to ASE format
    print("\n" + "=" * 80)
    print("\nExample: Converting a structure to ASE format...")
    
    if data:
        crystal = MPDSDataRetrieval.compile_crystal(data[0], 'ase')
        if crystal:
            print(f"Successfully converted entry {data[0][0]} to ASE Atoms object")
            print(f"Formula: {crystal.get_chemical_formula()}")
            print(f"Number of atoms: {len(crystal)}")
        else:
            print("Could not convert structure to ASE format")
    
except Exception as e:
    print(f"Error retrieving data: {e}")
    print("\nMake sure you have:")
    print("1. Installed mpds_client: pip install mpds_client")
    print("2. Set your API key either in the script or as MPDS_KEY environment variable")
    print("3. Have an active MPDS subscription")

# Additional examples for different searches

def search_specific_perovskites():
    """Search for specific types of perovskites"""
    
    # Example 1: Oxide perovskites
    print("\n\nExample: Searching for oxide perovskites...")
    client = MPDSDataRetrieval()
    
    oxide_perovskites = client.get_dataframe({
        "classes": "perovskite, oxide",
        "props": "atomic structure"
    }, pagesize=50)
    
    print(f"Found {len(oxide_perovskites)} oxide perovskites")
    
    # Example 2: Perovskites with specific elements (e.g., Sr-Ti-O)
    print("\n\nExample: Searching for SrTiO3-type perovskites...")
    
    sto_perovskites = client.get_dataframe({
        "elements": "Sr-Ti-O",
        "classes": "perovskite",
        "props": "atomic structure"
    }, pagesize=50)
    
    print(f"Found {len(sto_perovskites)} Sr-Ti-O perovskites")
    
    # Example 3: Perovskites with specific space group (e.g., Pm-3m)
    print("\n\nExample: Searching for cubic perovskites (Pm-3m)...")
    
    cubic_perovskites = client.get_dataframe({
        "classes": "perovskite",
        "sgs": "Pm-3m",
        "props": "atomic structure"
    }, pagesize=50)
    
    print(f"Found {len(cubic_perovskites)} cubic Pm-3m perovskites")

# Uncomment to run additional examples
# search_specific_perovskites()