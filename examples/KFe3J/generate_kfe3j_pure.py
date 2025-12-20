from magcalc.config_builder import MagCalcConfigBuilder
import numpy as np
import yaml
import os

def generate_kfe3j_pure():
    builder = MagCalcConfigBuilder()
    
    # 1. Lattice (Hexagonal)
    # Jarosite a=7.33, c=17.1374
    builder.set_lattice(a=7.33, b=7.33, c=17.1374, alpha=90, beta=90, gamma=120, space_group=191) # P6/mmm
    
    # 2. Atoms (Fe at 3f: 0.5, 0, 0)
    builder.add_wyckoff_atom(label="Fe", pos=[0.5, 0.0, 0.0], spin=2.5)
    
    # 3. No alignment needed, we will use our generated labels:
    # Fe0: [0.5, 0, 0]
    # Fe1: [0.5, 0.5, 0]
    # Fe2: [0, 0.5, 0]
    ref_config_path = 'examples/KFe3J/config_modern.yaml'
    with open(ref_config_path, 'r') as f:
        ref_cfg = yaml.safe_load(f)
    
    # 4. Interactions
    # Heisenberg J1 (NN)
    builder.add_symmetry_interaction(type="heisenberg", ref_pair=("Fe0", "Fe1"), distance=3.665, value="J1")
    # Heisenberg J2 (NNN)
    builder.add_symmetry_interaction(type="heisenberg", ref_pair=("Fe0", "Fe2"), distance=6.348, value="J2")
    
    # DM Interactions
    # Fe0(0.5, 0) -> Fe1(0.5, 0.5) [0.5, 0.5, 0] bond vector
    # This bond is along the 'b' axis direction.
    # In config_modern.yaml, they had [0, -Dy, -Dz] for a bond along 'a' (Fe1->Fe2).
    # Since my symmetry-based model will rotate vectors, I can pick a bond.
    # Standard: if bond is r_j - r_i = [0, 0.5, 0], then D is perp to bond.
    # Let's align with the modern config's Dy, Dz.
    # Modern: Fe1(0,0) -> Fe2(0.5,0) is [0, -Dy, -Dz].
    # In my model: Fe2(0, 0.5) -> Fe0(0.5, 0) is roughly the same direction (diagonal).
    # Actually, Fe2(0, 0.5) to Fe1(0.5, 0.5) is along 'a' axis.
    builder.add_symmetry_interaction(
        type="dm",
        ref_pair=("Fe2", "Fe1"),
        distance=3.665,
        value=["0", "1.0*Dy", "1.0*Dz"]
    )
    
    # 5. Parameters
    builder.config["parameters"] = ref_cfg.get("parameters", {})
    builder.config["parameter_order"] = ["J1", "J2", "Dy", "Dz", "H_dir", "H_mag"]
    
    # 6. Minimization & Magnetic Structure
    # All-in-all-out guess for Fe0, Fe1, Fe2
    # In config_modern alignment:
    # Fe2(0.5,0) -> 300 deg: [0.5, -0.866, 0]
    # Fe3(0,0.5) -> 60 deg: [0.5, 0.866, 0]
    # Fe1(0,0)   -> 180 deg: [-1.0, 0, 0]
    # In my basis:
    # Fe0(0.5, 0) is like reference Fe2 -> 300 deg
    # Fe2(0, 0.5) is like reference Fe3 -> 60 deg
    # Fe1(0.5, 0.5) is like reference Fe1 (shifted) -> 180 deg
    builder.set_minimization(
        enabled=True,
        method="L-BFGS-B",
        maxiter=5000,
        initial_configuration=[
            {"atom_index": 0, "phi": float(np.deg2rad(300))},
            {"atom_index": 1, "phi": float(np.deg2rad(180))},
            {"atom_index": 2, "phi": float(np.deg2rad(60))}
        ]
    )
    
    # 7. Field
    builder.set_field(magnitude="H_mag", direction="H_dir")
    
    # 8. Calculation & Output
    builder.config["crystal_structure"]["dimensionality"] = 2
    builder.config["calculation"] = {
        "cache_mode": "w",
        "cache_file_base": "KFe3J_pure_cache"
    }
    
    # Q-path from reference
    builder.config["q_path"] = ref_cfg.get("q_path", {})
    
    builder.set_tasks(
        run_minimization=True,
        run_dispersion=True,
        plot_dispersion=True,
        run_sqw_map=True,
        plot_sqw_map=True
    )
    
    builder.set_output(
        disp_data_filename="cache/data/KFe3J_pure_disp.npz",
        sqw_data_filename="cache/data/KFe3J_pure_sqw.npz"
    )
    
    builder.set_plotting(
        save_plot=True,
        disp_plot_filename="examples/plots/KFe3J_pure_disp.png",
        sqw_plot_filename="examples/plots/KFe3J_pure_sqw.png",
        show_plot=False,
        energy_limits_disp=[0, 20],
        energy_limits_sqw=[0, 20],
        plot_structure=True
    )
    
    # Expand interactions
    builder._expand_heisenberg_rules()
    
    final_inters = []
    final_inters.extend(builder.config["interactions"].get("heisenberg", []))
    final_inters.extend(builder.config["interactions"].get("dm_interaction", []))
    builder.config["interactions"] = final_inters
    
    # Ensure magnetic_elements
    builder.config["crystal_structure"]["magnetic_elements"] = ["Fe"]
    builder.config["crystal_structure"]["atoms_uc"] = [
        {"label": a["label"], "pos": [float(x) for x in a["pos"]], "spin_S": a["spin_S"]}
        for a in builder.atoms_uc
    ]

    with open("config_kfe3j_pure.yaml", "w") as f:
        yaml.dump(builder.config, f, sort_keys=False)

    print("Generated config_kfe3j_pure.yaml")

if __name__ == "__main__":
    generate_kfe3j_pure()
