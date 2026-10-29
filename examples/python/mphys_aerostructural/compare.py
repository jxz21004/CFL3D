"""Compare completed runs; never treat partial coupling history as a solution."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native", type=Path)
    parser.add_argument("idwarp", type=Path)
    args = parser.parse_args()
    configs, summaries = [], []
    for directory in (args.native, args.idwarp):
        path = directory / "summary.json"
        if not path.is_file():
            parser.error(f"{directory}: no completed summary; inspect the run log. "
                         "Final force differences cannot be computed from partial histories.")
        summaries.append(json.loads(path.read_text()))
        configs.append(json.loads((directory / "configuration.json").read_text()))
    if [c["warper"] for c in configs] != ["cfl3d", "idwarp"]:
        parser.error("Expected native CFL3D directory followed by IDWarp directory")
    for key in ("grid", "struct_mesh", "levels", "coarse_iterations", "cfd_iterations",
                "coupling_iterations", "fixed_budget", "smoke", "thickness"):
        default = 0.002 if key == "thickness" else None
        if configs[0].get(key, default) != configs[1].get(key, default):
            parser.error(f"Comparison settings differ: {key}")
    print("scenario,quantity,native,idwarp,native_minus_idwarp")
    for name in ("cruise", "maneuver"):
        left, right = (s[name] for s in summaries)
        for key in ("cfd_cycles", "startup_cycles", "total_cycles", "coupling_evaluations"):
            if left[key] != right[key]:
                parser.error(f"{name}: budgets differ ({key})")
        for key in ("cl", "cd", "lift", "drag", "max_struct_displacement_dof"):
            print(f"{name},{key},{left[key]:.12g},{right[key]:.12g},"
                  f"{left[key]-right[key]:.12g}")
    print("Budget completion does not establish convergence; inspect both solver logs.")


if __name__ == "__main__":
    main()
