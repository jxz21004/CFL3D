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
from cfl3d import Boundary, CFL3D, CGNSMesh, Connection, SolverOptions, Zone
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


def cases():
    yield "box", box_mesh(True), SolverOptions(mach=.3, turbulence="euler", iterations=5,
                                               multigrid_levels=2)
    options = SolverOptions(mach=.2, reynolds=6e6, temperature=460/1.8, cfl=3,
                            iterations=5, multigrid_levels=3, mesh_sequence=True,
                            coarse_iterations=3, first_order_iterations=2,
                            coarse_first_order_iterations=1, cfl_ramp_steps=3,
                            cfl_ramp_factor=2, restart_interval=2)
    for model in ("sa", "sst"):
        yield model, split_plate(), replace(options, turbulence=model)


def compare_reference(result, filename, name):
    differences = {}
    with np.load(filename) as reference:
        for zone, values in result.fields.items():
            expected = reference[f"zone_{zone}"]
            np.testing.assert_allclose(values, expected, rtol=2e-11, atol=2e-13)
            differences[zone] = float(np.max(np.abs(values-expected)))
        # Uniform-flow residual normalization is sensitive to roundoff.
        columns = slice(1, None) if name == "box" else slice(None)
        np.testing.assert_allclose(result.history[:, columns], reference["history"][:, columns],
                                   rtol=2e-10, atol=2e-13)
        history_difference = float(np.max(np.abs(
            result.history[:, columns]-reference["history"][:, columns])))
    assert np.isfinite(result.history).all()
    return {"max_field_difference": max(differences.values()),
            "max_history_difference": history_difference,
            "history_columns_compared": "coefficients" if name == "box" else "all",
            "history_rows": len(result.history)}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--split", action="store_true", help="Run independent two-rank subcommunicators")
    parser.add_argument("--native-error", action="store_true", help="Trigger a native negative-volume MPI abort")
    parser.add_argument("--adf", action="store_true", help="Read actual CGNS/ADF files instead of memory meshes")
    parser.add_argument("--only", choices=("box", "sa", "sst"))
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    root = args.directory.resolve()
    if args.prepare:
        adf_report = {}
        for name, mesh, options in cases():
            result = CFL3D(mesh, options).solve(output_directory=root / "serial" / name)
            expected_rows = options.iterations + (options.multigrid_levels-1) * options.coarse_iterations \
                if options.mesh_sequence else options.iterations
            assert result.history.shape == (expected_rows, 8)
            if name != "box":
                assert result.coefficients["CD"] > 0
                assert max(np.ptp(q[..., 1]) for q in result.fields.values()) > .01
            if args.adf:
                from cgns_fixtures import write_adf
                (root / "meshes").mkdir(parents=True, exist_ok=True)
                path = write_adf(mesh, root / "meshes" / f"{name}.cgns")
                from_file = CFL3D(path, options).solve(output_directory=root / "serial-adf" / name)
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
    for name, mesh, options in cases():
        if args.only and name != args.only:
            continue
        source = root / "meshes" / f"{name}.cgns" if args.adf else mesh
        solver = CFL3D(source if comm.rank == 0 else None, options, comm=comm)
        result = solver.solve(output_directory=root / run_name / name)

        report[name] = solver._checked(lambda: compare_reference(
            result, root / "serial" / name / "solution.npz", name))
        report[name]["mesh_input"] = "CGNS/ADF" if args.adf else "memory"
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
