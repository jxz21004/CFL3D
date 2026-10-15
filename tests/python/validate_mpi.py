"""Compare actual MPI solutions against serial references, without h5py.

Prepare once with --prepare, then run under mpirun with 2 or 3 processes.
The MPI path repeats solves, checks that MPI remains usable, and exercises a
root-side configuration error before another successful native call. With
--adf, native CGNS files are written during preparation and read during solves.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from cfl3d import Boundary, CFL3D, CGNSMesh, Connection, MultigridStage, SolverOptions, Zone
from fixtures import box_mesh, flatplate_mesh


def split_plate():
    original = flatplate_mesh().zones[0]
    zones = []
    for number, (lo, hi) in enumerate(((1, 33), (33, 65))):
        zone = Zone(f"plate{number}", original.coordinates[:, lo-1:hi].copy())
        for bc in original.boundaries:
            p = bc.point_range.copy()
            if p[1, 0] == p[1, 1]:
                if not lo <= p[1, 0] <= hi:
                    continue
            else:
                p[1] = max(lo, p[1, 0]), min(hi, p[1, 1])
                if p[1, 1] <= p[1, 0]:
                    continue
            p[1] -= lo-1
            zone.boundaries.append(Boundary(bc.name, bc.bc_type, p, bc.data))
        p = np.array([[1, 2], [33 if number == 0 else 1]*2, [1, 97]])
        donor = p.copy()
        donor[1] = 1 if number == 0 else 33
        zone.connections.append(Connection(f"plate{1-number}", p, donor, np.array([1, 2, 3])))
        zones.append(zone)
    return CGNSMesh(zones)


def _cases(controls=False):
    yield "box", box_mesh(True), SolverOptions(mach=.3, turbulence="euler", iterations=5,
                                               multigrid_levels=2)
    options = SolverOptions(mach=.2, reynolds=6e6, temperature=460/1.8, cfl=3,
                            iterations=5, multigrid_levels=3, mesh_sequence=True,
                            coarse_iterations=3, first_order_iterations=2,
                            coarse_first_order_iterations=1, cfl_ramp_steps=3,
                            cfl_ramp_factor=2, restart_interval=2)
    for model in ("sa", "sst"):
        yield model, split_plate(), replace(options, turbulence=model)
    if controls:
        yield "lift", box_mesh(True), SolverOptions(turbulence="euler", alpha=1., iterations=6,
            target_cl=.1, alpha_relaxation=3., alpha_step_limit=.02, alpha_update_interval=2)
        yield "controls", split_plate(), SolverOptions(turbulence="sst", cfl=.5,
            mach=.2, reynolds=6e6, temperature=460/1.8, multigrid_levels=3,
            limiter=(0, 2, 4), flux=(1, 0, 1), inversion=(1, 0, 1), kappa=(0., 1/3, 1/3),
            multigrid_up_iterations=1,
            multigrid_stages=(MultigridStage(3, 2, 1, (1, 2)), MultigridStage(5, 2, 2, (2, 1))))
        yield "lift_plate", split_plate(), SolverOptions(turbulence="sst", cfl=3.,
            mach=.2, reynolds=6e6, temperature=460/1.8, multigrid_levels=2,
            iterations=200, limiter=2, target_cl=.01, alpha_relaxation=.5,
            alpha_step_limit=.1, alpha_update_interval=5)


def cases(controls=False, forces=False):
    for name, mesh, options in _cases(controls):
        if forces:
            for index, zone in enumerate(mesh.zones, 1):
                for boundary in zone.boundaries:
                    boundary.family = f"Part{index}"
                    boundary.additional_families = ("Whole",)
            options = replace(options, freestream_pressure=101325.)
        yield name, mesh, options


def force_groups(enabled):
    return {"Assembly": ["Part1", "Part2"], "Overlap": ["Whole", "Part1", "Whole"]} if enabled else None


def check_forces(solver, result):
    total = solver.getSolution()
    assert total == solver.getSolution("Assembly") == solver.getSolution("Overlap") == solver.getSolution("Whole")
    for key in ("CL", "CD", "CY", "Cmx", "Cmy", "Cmz"):
        native_key = key.upper() if key.startswith("Cm") else key
        np.testing.assert_allclose(total[key], result.coefficients[native_key], rtol=2e-11, atol=2e-13)
    np.testing.assert_allclose(total["CD"], total["CDp"]+total["CDv"], rtol=2e-11, atol=2e-13)
    parts = solver.getSolution("Part1"), solver.getSolution("Part2")
    for key in total:
        if key != "units":
            np.testing.assert_allclose(total[key], sum(part[key] for part in parts), rtol=2e-11, atol=2e-11)
    return total


def compare_reference(result, filename, name):
    differences = {}
    with np.load(filename) as reference:
        for zone, values in result.fields.items():
            expected = reference[f"zone_{zone}"]
            np.testing.assert_allclose(values, expected, rtol=2e-11, atol=2e-13)
            differences[zone] = float(np.max(np.abs(values-expected)))
        # Uniform-flow residual normalization is sensitive to roundoff.
        uniform = name in ("box", "lift")
        columns = slice(1, None) if uniform else slice(None)
        np.testing.assert_allclose(result.history[:, columns], reference["history"][:, columns],
                                   rtol=2e-10, atol=2e-13)
        history_difference = float(np.max(np.abs(
            result.history[:, columns]-reference["history"][:, columns])))
        np.testing.assert_allclose(result.final_alpha, reference["final_alpha"], rtol=0, atol=2e-12)
        keys = sorted(result.surface_coefficients)
        np.testing.assert_array_equal(np.asarray(keys), reference["surface_names"])
        values = np.asarray([result.surface_coefficients[key] for key in keys])
        np.testing.assert_allclose(values, reference["surface_coefficients"], rtol=2e-10, atol=2e-13)
        surface_difference = float(np.max(np.abs(values-reference["surface_coefficients"])))
    assert np.isfinite(result.history).all()
    return {"max_field_difference": max(differences.values()),
            "max_history_difference": history_difference,
            "history_columns_compared": "coefficients" if uniform else "all",
            "history_rows": len(result.history), "final_alpha": result.final_alpha,
            "max_surface_difference": surface_difference}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--split", action="store_true", help="Run independent two-rank subcommunicators")
    parser.add_argument("--native-error", action="store_true", help="Trigger a native negative-volume MPI abort")
    parser.add_argument("--adf", action="store_true", help="Read actual CGNS/ADF files instead of memory meshes")
    parser.add_argument("--controls", action="store_true", help="Also check CL matching and custom numerical controls")
    parser.add_argument("--forces", action="store_true", help="Check CGNS family groups, force split and SI loads")
    parser.add_argument("--only", choices=("box", "sa", "sst", "lift", "controls", "lift_plate"))
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    root = args.directory.resolve()
    if args.prepare:
        adf_report = {}
        for name, mesh, options in cases(args.controls, args.forces):
            solver = CFL3D(mesh, options, groups=force_groups(args.forces))
            result = solver.solve(output_directory=root / "serial" / name)
            if args.forces:
                check_forces(solver, result)
            expected_rows = sum(stage.iterations for stage in options.stage_schedule())
            assert result.history.shape == (expected_rows, 8)
            if name not in ("box", "lift"):
                assert result.coefficients["CD"] > 0
                assert max(np.ptp(q[..., 1]) for q in result.fields.values()) > (1e-3 if name == "controls" else .01)
            if name == "lift":
                np.testing.assert_allclose(result.final_alpha, 1.06, rtol=0, atol=2e-12)
            if name == "lift_plate":
                assert abs(result.coefficients["CL"]-options.target_cl) < 1e-3
                assert result.final_alpha > 0
            if args.adf:
                from cgns_fixtures import write_adf
                (root / "meshes").mkdir(parents=True, exist_ok=True)
                path = write_adf(mesh, root / "meshes" / f"{name}.cgns")
                file_solver = CFL3D(path, options, groups=force_groups(args.forces))
                from_file = file_solver.solve(output_directory=root / "serial-adf" / name)
                if args.forces:
                    check_forces(file_solver, from_file)
                adf_report[name] = compare_reference(from_file, root / "serial" / name / "solution.npz", name)
                adf_report[name]["mesh_input"] = "CGNS/ADF"
        if args.adf:
            (root / "serial-adf/report.json").write_text(json.dumps(adf_report, indent=2))
            print(json.dumps(adf_report, indent=2), flush=True)
        print("Serial references prepared", flush=True)
        return

    from mpi4py import MPI
    world = MPI.COMM_WORLD
    if args.split and (world.size < 4 or world.size % 2):
        parser.error("--split requires an even number of processes, at least four")
    group = world.rank // 2 if args.split else None
    comm = world.Split(group, world.rank) if args.split else world
    run_name = f"mpi-{comm.size}" + (f"-group-{group}" if args.split else "")
    if args.only:
        run_name += f"-{args.only}"
    if args.native_error:
        mesh = box_mesh(True)
        for zone in mesh.zones:
            zone.coordinates[..., 1] *= -1
        solver = CFL3D(mesh if comm.rank == 0 else None,
                       SolverOptions(turbulence="euler", iterations=1), comm=comm)
        solver.solve(output_directory=root / "native-error")
        raise AssertionError("Expected MPI abort for negative cell volumes")
    report = {}
    # Invalid root options must raise on every rank and leave MPI usable.
    try:
        CFL3D(box_mesh() if comm.rank == 0 else None, SolverOptions(iterations=0), comm=comm)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected collective configuration failure")
    comm.Barrier()
    if args.adf:
        try:
            CFL3D(root / "missing.cgns" if comm.rank == 0 else None, comm=comm)
        except RuntimeError as exc:
            assert "FileNotFoundError" in str(exc)
        else:
            raise AssertionError("Expected collective CGNS file-read failure")
        comm.Barrier()
    for name, mesh, options in cases(args.controls, args.forces):
        if args.only and name != args.only:
            continue
        source = root / "meshes" / f"{name}.cgns" if args.adf else mesh
        solver = CFL3D(source if comm.rank == 0 else None, options, comm=comm,
                       groups=force_groups(args.forces) if comm.rank == 0 else None)
        result = solver.solve(output_directory=root / run_name / name)

        report[name] = solver._checked(lambda: compare_reference(
            result, root / "serial" / name / "solution.npz", name))
        report[name]["mesh_input"] = "CGNS/ADF" if args.adf else "memory"
        if args.forces:
            report[name]["loads"] = solver._checked(lambda: check_forces(solver, result))
            if comm.rank == 0:
                # Only root calls these methods: they must never enter a collective.
                solver.addGroups({"RootOnly": ["Part1"]})
                assert solver.getSolution("RootOnly") == solver.getSolution("Part1")
        comm.Barrier()  # Native solve must not finalize MPI.
    if comm.rank == 0:
        report_path = root / run_name / "report.json"
        report_path.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    print(f"MPI_ALIVE world_rank={world.rank} local_rank={comm.rank}", flush=True)
    if args.split:
        comm.Free()
    world.Barrier()


if __name__ == "__main__":
    main()
