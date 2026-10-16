#!/usr/bin/env bash
# Load the desired Python/Intel MPI environment before invoking this script.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:$PWD/tests/python${PYTHONPATH:+:$PYTHONPATH}"
export I_MPI_FABRICS="${I_MPI_FABRICS:-shm}"
mpi_python=$(python -c 'import sys; print(sys.executable)')
validation_root="${CFL3D_VALIDATION_DIRECTORY:-build/mpi-validation}"
mkdir -p "$validation_root/logs"
python tests/python/validate_mpi.py --prepare --directory "$validation_root" "$@"
for processes in 2 3; do
    timeout -k 5s 60s mpirun -np "$processes" "$mpi_python" \
        tests/python/validate_mpi.py --directory "$validation_root" "$@" \
        > "$validation_root/logs/mpi-$processes.log" 2>&1
    cat "$validation_root/logs/mpi-$processes.log"
done
timeout -k 5s 60s mpirun -np 4 "$mpi_python" \
    tests/python/validate_mpi.py --split --directory "$validation_root" "$@" \
    > "$validation_root/logs/subcommunicators.log" 2>&1
cat "$validation_root/logs/subcommunicators.log"

if command -v valgrind >/dev/null; then
    # MPI maps a large shared region; scanning it for leaks at exit can stall.
    # Keep invalid-access/undefined-value checks; this is not a leak test.
    memory_case=box
    if [[ " $* " == *" --controls "* ]]; then
        memory_case=controls  # Exercise MIT > 1 and the block matrix inversion.
    fi
    timeout -k 5s 90s mpirun -np 3 valgrind --leak-check=no --error-exitcode=97 \
        --log-file="$PWD/$validation_root/logs/valgrind-%p.log" \
        "$mpi_python" tests/python/validate_mpi.py --only "$memory_case" --directory "$validation_root" "$@" \
        > "$validation_root/logs/memory-check.log" 2>&1
    cat "$validation_root/logs/memory-check.log"
fi

set +e
timeout -k 5s 30s mpirun -np 3 "$mpi_python" \
    tests/python/validate_mpi.py --native-error --directory "$validation_root" "$@" \
    > "$validation_root/logs/native-error.log" 2>&1
abort_status=$?
set -e
if [ "$abort_status" -eq 0 ] || [ "$abort_status" -eq 124 ] || [ "$abort_status" -eq 137 ]; then
    cat "$validation_root/logs/native-error.log"
    exit 1
fi
if ! rg -q 'MPI_Abort|MPI_ABORT' "$validation_root/logs/native-error.log"; then
    cat "$validation_root/logs/native-error.log"
    exit 1
fi
printf 'Expected native MPI abort returned status %s without timeout\n' "$abort_status"
