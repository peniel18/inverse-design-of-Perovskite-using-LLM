import qmpy_rester as qr
import os


def get_all_perovskites_as_cif(output_dir: str = 'perovskites_cif') -> None:
    """
    Retrieve all perovskite structures (ABO3) from OQMD database and save as CIF files
    
    Parameters:
    -----------
    output_dir : str
        Directory where CIF files will be saved
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the API wrapper
    with qr.QMPYRester() as q:
        
        print("Fetching perovskites with generic formula ABO3...")
        
        # Define the query parameters - request structure data
        query_params = {
            'filter': 'generic=ABO3',  # Perovskite stoichiometry
            'fields': 'name,entry_id,composition_generic,spacegroup,sites,unit_cell',
        }
        
        try:
            # Get data using qmpy_rester
            data = q.get_oqmd_phases(verbose=True, **query_params)
            
            print(f"Retrieved {len(data)} perovskite structures")
            print(f"\nSaving CIF files to '{output_dir}/' directory...")
            
            # Counter for successfully saved files
            saved_count = 0
            
            # Process each entry
            for i, entry in enumerate(data):
                try:
                    # Extract information
                    if isinstance(entry, dict):
                        entry_id = entry.get('entry_id', i)
                        name = entry.get('name', f'structure_{i}')
                        composition = entry.get('composition_generic', 'Unknown')
                        spacegroup = entry.get('spacegroup', 'P1')
                        sites = entry.get('sites', [])
                        unit_cell = entry.get('unit_cell', [])
                    else:
                        print(f"  Skipping entry {i}: unexpected format")
                        continue
                    
                    # Create CIF content
                    cif_content = create_cif_from_data(
                        name, entry_id, composition, spacegroup, sites, unit_cell
                    )
                    
                    # Create filename
                    filename = f"{name.replace('/', '_').replace(' ', '_')}_{entry_id}.cif"
                    filepath = os.path.join(output_dir, filename)
                    
                    # Save CIF file
                    with open(filepath, 'w') as f:
                        f.write(cif_content)
                    
                    saved_count += 1
                    print(f"  Saved {saved_count}/{len(data)}: {filename}")
                    
                except Exception as e:
                    print(f"  Error saving entry {i}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            print(f"\n{'='*50}")
            print(f"Successfully saved {saved_count}/{len(data)} CIF files")
            print(f"Files saved in: {os.path.abspath(output_dir)}")
            print(f"{'='*50}")
            
        except Exception as e:
            print(f"Error retrieving data: {e}")
            import traceback
            traceback.print_exc()


def create_cif_from_data(name, entry_id, composition, spacegroup, sites, unit_cell):
    """
    Create CIF format string from OQMD data
    
    Parameters:
    -----------
    name : str
        Structure name
    entry_id : int
        OQMD entry ID
    composition : str
        Chemical composition
    spacegroup : str
        Space group symbol
    sites : list
        List of atomic sites
    unit_cell : list
        Unit cell parameters [a, b, c, alpha, beta, gamma]
    
    Returns:
    --------
    str : CIF formatted string
    """
    
    cif = []
    
    # Header
    cif.append("data_" + str(entry_id))
    cif.append("")
    cif.append(f"_chemical_name_common '{name}'")
    cif.append(f"_chemical_formula_sum '{composition}'")
    cif.append(f"_oqmd_entry_id {entry_id}")
    cif.append("")
    
    # Space group
    cif.append(f"_symmetry_space_group_name_H-M '{spacegroup}'")
    cif.append("")
    
    # Cell parameters
    if unit_cell and len(unit_cell) >= 6:
        cif.append("_cell_length_a    {:.6f}".format(unit_cell[0]))
        cif.append("_cell_length_b    {:.6f}".format(unit_cell[1]))
        cif.append("_cell_length_c    {:.6f}".format(unit_cell[2]))
        cif.append("_cell_angle_alpha {:.6f}".format(unit_cell[3]))
        cif.append("_cell_angle_beta  {:.6f}".format(unit_cell[4]))
        cif.append("_cell_angle_gamma {:.6f}".format(unit_cell[5]))
    else:
        cif.append("_cell_length_a    1.0")
        cif.append("_cell_length_b    1.0")
        cif.append("_cell_length_c    1.0")
        cif.append("_cell_angle_alpha 90.0")
        cif.append("_cell_angle_beta  90.0")
        cif.append("_cell_angle_gamma 90.0")
    
    cif.append("")
    
    # Atomic positions
    if sites:
        cif.append("loop_")
        cif.append("_atom_site_label")
        cif.append("_atom_site_type_symbol")
        cif.append("_atom_site_fract_x")
        cif.append("_atom_site_fract_y")
        cif.append("_atom_site_fract_z")
        
        for idx, site in enumerate(sites, 1):
            if isinstance(site, dict):
                element = site.get('species', 'X')
                x = site.get('x', 0.0)
                y = site.get('y', 0.0)
                z = site.get('z', 0.0)
            elif isinstance(site, (list, tuple)) and len(site) >= 4:
                element = site[0]
                x, y, z = site[1:4]
            else:
                continue
            
            label = f"{element}{idx}"
            cif.append(f"{label:6s} {element:4s} {x:10.6f} {y:10.6f} {z:10.6f}")
    
    return "\n".join(cif)


if __name__ == "__main__":
    get_all_perovskites_as_cif(output_dir='perovskites_cif')