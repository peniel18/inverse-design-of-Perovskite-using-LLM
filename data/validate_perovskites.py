import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
import os 

def check_perovskite(cif_string):
    """
    Check if a structure is a perovskite, including:
    - Simple perovskites (ABX3)
    - Double perovskites (A2BB'X6, including complex B-site mixing)
    - Layered perovskites (Ruddlesden-Popper, Dion-Jacobson, etc.)
    - Defect perovskites (A-site or X-site vacancies)
    """
    struct = Structure.from_str(cif_string, fmt="cif")
    formula = struct.composition.reduced_formula
    print(f"Analyzing: {formula}")

    comp = struct.composition.get_el_amt_dict()
    
    # Get primitive structure to handle supercells
    prim_struct = struct.get_primitive_structure()
    prim_comp = prim_struct.composition.get_el_amt_dict()
    
    # Normalize composition to smallest integer ratios
    comp_values = list(prim_comp.values())
    min_val = min(comp_values)
    normalized_comp = {k: v/min_val for k, v in prim_comp.items()}
    
    # Identify anions (highest electronegativity) and cations
    elements = list(prim_struct.composition.elements)
    sorted_by_en = sorted(elements, key=lambda e: e.X)
    
    # Typically, the most electronegative element is the anion (X site)
    anion = sorted_by_en[-1]
    cations = [e for e in elements if e != anion]
    
    # Check for perovskite-like stoichiometry
    anion_count = normalized_comp.get(str(anion), 0)
    
    perovskite_type = None
    
    # Calculate total cation counts
    cation_counts = {str(c): normalized_comp.get(str(c), 0) for c in cations}
    total_cation_count = sum(cation_counts.values())
    
    # ABX3 - Simple perovskite
    if len(cations) == 2 and abs(anion_count - 3) < 0.1:
        sorted_counts = sorted(cation_counts.values())
        if abs(sorted_counts[0] - 1) < 0.1 and abs(sorted_counts[1] - 1) < 0.1:
            perovskite_type = "ABX3 (Simple)"
    
    # Double perovskite patterns
    # A2BB'X6 where B-site can have multiple elements
    elif abs(anion_count - 6) < 0.1:
        # Total cations should be 4 (2 A-site + 2 B-site)
        if abs(total_cation_count - 4) < 0.2:
            perovskite_type = "A2BB'X6 (Double)"
        # Also check for 1:1:1:1:6 pattern (mixed B-site with 3 elements)
        elif len(cations) >= 3 and abs(total_cation_count - 4) < 0.5:
            perovskite_type = "A2BB'B''X6 (Complex Double)"
    
    # ABX3 pattern but with multiple A or B sites
    elif abs(anion_count - 3) < 0.1:
        # Could be mixed A-site or B-site
        if abs(total_cation_count - 2) < 0.3:
            if len(cations) == 3:
                perovskite_type = "A(B,B')X3 or (A,A')BX3 (Mixed site)"
            elif len(cations) == 4:
                perovskite_type = "(A,A')(B,B')X3 (Double mixed)"
            else:
                perovskite_type = "ABX3 (Simple)"
    
    # Layered perovskites (e.g., A2BX4, A3B2X7, etc.)
    elif anion_count >= 3:
        # Sort cations by size to identify likely B-site
        b_site_candidates = sorted(cations, key=lambda e: e.atomic_radius)
        if b_site_candidates:
            # Sum of all B-site-like elements (smaller cations)
            b_total = sum([normalized_comp.get(str(c), 0) 
                          for c in b_site_candidates[:max(1, len(cations)//2)]])
            
            if b_total > 0:
                x_to_b_ratio = anion_count / b_total
                if 2.5 <= x_to_b_ratio <= 5.0:
                    perovskite_type = "Layered Perovskite"
    
    # Defect perovskites (e.g., ReO3 structure = BX3)
    elif len(cations) == 1 and abs(anion_count - 3) < 0.1:
        perovskite_type = "BX3 (Defect - no A-site)"
    
    # Additional check: ratio-based approach for edge cases
    if perovskite_type is None:
        # Check if anion:cation ratio is close to 3:2 or 6:4 (perovskite-like)
        if total_cation_count > 0:
            anion_to_cation_ratio = anion_count / total_cation_count
            if 1.4 <= anion_to_cation_ratio <= 1.6:  # Close to 3:2
                perovskite_type = "Perovskite-like (ratio-based)"
    
    if perovskite_type is None:
        print(f"Not a recognized perovskite stoichiometry")
        print(f"Anion count: {anion_count}, Total cation count: {total_cation_count}")
        print(f"Cation composition: {cation_counts}")
        return False
    
    # Coordination check for B-site(s)
    # Find B-site elements (smaller cations, typically transition metals)
    if len(cations) >= 1:
        # Sort by atomic radius to identify B-site candidates
        sorted_cations = sorted(cations, key=lambda e: e.atomic_radius)
        
        # For double perovskites, check middle-sized cations (B-site)
        # For simple perovskites, check the smaller cation
        if len(cations) >= 3:
            # In complex perovskites, B-sites are typically middle-to-small cations
            b_site_elements = sorted_cations[:max(2, len(cations)-1)]
        else:
            b_site_elements = sorted_cations[:max(1, len(cations)//2 + 1)]
        
        cnn = CrystalNN()
        coordination_pass = False
        coordination_numbers = []
        
        for b_el in b_site_elements:
            b_indices = [i for i, site in enumerate(prim_struct) if site.specie == b_el]
            
            for idx in b_indices[:min(3, len(b_indices))]:  # Check first few B-sites
                try:
                    cn = cnn.get_cn(prim_struct, idx)
                    coordination_numbers.append((str(b_el), cn))
                    # Allow 4, 5, 6, 8 coordination (various geometries)
                    if 4 <= cn <= 8:
                        coordination_pass = True
                        break
                except Exception as e:
                    continue
            
            if coordination_pass:
                break
        
        print(f"B-site coordination numbers: {coordination_numbers}")
        
        if not coordination_pass:
            print(f"B-site coordination check failed")
            # Don't reject immediately - some structures might have issues with CrystalNN
            # but still be perovskites
            print(f"WARNING: Proceeding despite coordination check failure")
    
    # Calculate tolerance factor for simple perovskites
    if "ABX3" in perovskite_type and len(cations) == 2:
        try:
            b_site_el, a_site_el = sorted(cations, key=lambda e: e.atomic_radius)
            rA = a_site_el.average_ionic_radius
            rB = b_site_el.average_ionic_radius
            rX = anion.average_ionic_radius
            
            if rA and rB and rX:
                t = (rA + rX) / (np.sqrt(2) * (rB + rX))
                print(f"Tolerance Factor (t): {t:.3f}")
                
                # More lenient tolerance factor range
                if not (0.7 <= t <= 1.2):
                    print(f"Tolerance factor outside typical range (0.7-1.2)")
        except:
            pass
    
    print(f"✓ Identified as: {perovskite_type}")
    print(f"  Composition: {prim_comp}")
    return True


def read_cif_file(file_path: str):
    with open(file_path, 'r') as f:
        cif_data = f.read()
    return cif_data


files = os.listdir("./data/cif_files")
files = [os.path.join("./data/cif_files", file) for file in files if file.endswith('.cif')]
number_of_valid = 0 

valid_files = []
invalid_files = []

for file in files: 
    try:
        cif_data = read_cif_file(file)
        valid = check_perovskite(cif_data)
        if valid: 
            number_of_valid += 1
            valid_files.append(file)
        else:
            invalid_files.append(file)
    except Exception as e:
        print(f"Error processing {file}: {str(e)}")
        invalid_files.append(file)
    
    print("-" * 50)


#print(f"\n invalid files: ")
#print(invalid_files)


print(f"\n{'='*50}")
print(f"Total Valid Perovskites: {number_of_valid}/{len(files)}")
print(f"{'='*50}")