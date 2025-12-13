import numpy as np
import os

def load_data(filename):
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return None
    data = np.load(filename, allow_pickle=True)
    return data['energies'], data['q_vectors']

def main():
    # Determine project root (parent of 'scripts')
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)
    
    manual_file = os.path.join(project_root, "cache", "data", "KFe3J_disp_data.npz")
    # For decl_file, assuming it lives in examples/KFe3J or same place?
    # If it was KFe3J/..., let's assume examples/KFe3J for now, or maybe it should be moved to cache too?
    # User only mentioned KFe3J_disp_data.npz. I keep decl_file pointing to expected relative location or maybe cache?
    # Let's verify decl_file location first. But assuming it's legacy comparison script.
    # I will point decl_file to examples/KFe3J for now if that's where it resides.
    decl_file = os.path.join(project_root, "examples", "KFe3J", "KFe3J_decl_prim_disp_data.npz")
    
    # Load manual
    en_man, q_man = load_data(manual_file)
    # Load declarative
    en_decl, q_decl = load_data(decl_file)
    
    if en_man is None or en_decl is None:
        return

    print(f"Manual Data Shape: {en_man.shape} (N_q, N_bands)")
    print(f"Decl Data Shape: {en_decl.shape} (N_q, N_bands)")
    
    # Check bands
    # Flatten and get stats
    # Filter nan
    
    flat_man = np.array([e for sub in en_man for e in (sub if sub is not None else [])])
    flat_decl = np.array([e for sub in en_decl for e in (sub if sub is not None else [])])
    
    print("\n--- Statistics ---")
    print(f"Manual Energy: Min={np.min(flat_man):.4f}, Max={np.max(flat_man):.4f}, Mean={np.mean(flat_man):.4f}")
    print(f"Decl Energy:   Min={np.min(flat_decl):.4f}, Max={np.max(flat_decl):.4f}, Mean={np.mean(flat_decl):.4f}")
    
    # Check specific Q-point (e.g. index 0 - Gamma)
    print("\n--- Gamma Point (Index 0) ---")
    print(f"Manual E[0]: {en_man[0]}")
    print(f"Decl E[0]:   {en_decl[0]}")
    
    # Check Index 50 (M point?)
    if len(en_man) > 50:
         print(f"\n--- Index 50 ---")
         print(f"Manual E[50]: {en_man[50]}")
         print(f"Decl E[50]:   {en_decl[50]}")

if __name__ == "__main__":
    main()
