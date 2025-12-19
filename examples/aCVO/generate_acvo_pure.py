from magcalc.config_builder import MagCalcConfigBuilder
import numpy as np
import yaml

def generate_acvo_pure():
    builder = MagCalcConfigBuilder()
    
    # 1. Lattice (aCVO) - Space Group 43 (Fdd2)
    builder.set_lattice(a=20.645, b=8.383, c=6.442, space_group=43)
    
    # 2. Atoms (Wyckoff 16b site)
    x, y, z = 0.165720, 0.364600, 0.754500
    builder.add_wyckoff_atom(label="Cu", pos=[x, y, z], spin=0.5)
    
    # 2. Alignment (to match labeling for parity check)
    with open('examples/aCVO/config_modern.yaml', 'r') as f:
        ref_cfg = yaml.safe_load(f)
    ref_pos = [a['pos'] for a in ref_cfg['crystal_structure']['atoms_uc']]
    builder.align_atoms(ref_pos)
    
    # Heisenberg
    builder.add_symmetry_interaction(type="heisenberg", ref_pair=("Cu0", "Cu9"), distance=3.1325, value="J1")
    builder.add_symmetry_interaction(type="heisenberg", ref_pair=("Cu0", "Cu7"), distance=3.9751, value="J2")
    builder.add_symmetry_interaction(type="heisenberg", ref_pair=("Cu0", "Cu5"), distance=5.2572, value="J3")

    # Anisotropic Exchange (G1)
    builder.add_symmetry_interaction(
        type="anisotropic_exchange",
        ref_pair=("Cu0", "Cu9"),
        distance=3.1325,
        value=["G1", "-G1", "-G1"]
    )

    # DM (on J1 bonds - staggered)
    builder.add_symmetry_interaction(
        type="dm",
        ref_pair=("Cu0", "Cu9"),
        distance=3.1325,
        value=["-1.0*Dx", "0", "0"]
    )
    
    # 4. Parameters
    builder.config["parameters"] = ref_cfg.get("parameters", {})
    builder.config["parameter_order"] = ref_cfg.get("parameter_order", ["J1", "J2", "J3", "G1", "Dx", "H_dir", "H_mag"])

    # 4. Magnetic Structure (Exactly same as reference for parity check)
    ref_init = ref_cfg.get("minimization", {}).get("initial_configuration", [])
    
    # 5. Minimization
    builder.set_minimization(
        enabled=True, 
        method="L-BFGS-B", 
        maxiter=3000,
        initial_configuration=ref_init
    )
    
    # 6. Field
    builder.set_field(magnitude="H_mag", direction="H_dir")
    
    # 7. Calculation & Output
    builder.config["calculation"] = {
        "cache_mode": "w",
        "cache_file_base": "acvo_pure_cache"
    }
    builder.set_q_path(start=[0, 1, 0], end=[0, 3, 0], steps=200)
    builder.set_tasks(
        run_minimization=True,
        run_dispersion=True,
        plot_dispersion=True
    )
    builder.set_output(
        disp_data_filename="acvo_pure_disp.npz"
    )
    builder.set_plotting(
        save_plot=True,
        disp_plot_filename="acvo_pure_disp.png",
        show_plot=False
    )

    # 8. Post-process: Flatten interactions for the runner
    # Expand generic rules first
    builder._expand_heisenberg_rules()
    builder._expand_anisotropic_exchange_rules()
    
    final_inters = []
    final_inters.extend(builder.config["interactions"].get("heisenberg", []))
    final_inters.extend(builder.config["interactions"].get("dm_interaction", []))
    final_inters.extend(builder.config["interactions"].get("anisotropic_exchange", []))
    builder.config["interactions"] = final_inters
    
    # Ensure magnetic_elements
    builder.config["crystal_structure"]["magnetic_elements"] = ["Cu"]
    builder.config["crystal_structure"]["atoms_uc"] = [
        {"label": a["label"], "pos": [float(x) for x in a["pos"]], "spin_S": a["spin_S"]}
        for a in builder.atoms_uc
    ]

    with open("config_acvo_pure.yaml", "w") as f:
        yaml.dump(builder.config, f, sort_keys=False)

    print("Generated config_acvo_pure.yaml")

if __name__ == "__main__":
    generate_acvo_pure()
