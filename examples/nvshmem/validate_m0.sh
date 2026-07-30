#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

usage() {
    cat <<'EOF'
Run the oneCCL NVSHMEM M0 interoperability checks.

Usage:
  validate_m0.sh [options]

Options:
  --binary FILE         M0 executable (default: <script-dir>/oneccl-nvshmem-m0-build/oneccl_nvshmem_m0)
  --launcher NAME       mpirun or srun (default: mpirun)
  --launcher-path FILE  Explicit launcher executable
  --nodes N             Number of nodes (default: 1)
  --pes-per-node N      PEs per node (default: 2 for one node, 1 for multiple nodes)
  --counts LIST         Comma-separated element counts (default: 1,1024,1048576)
  --mpirun-flavor NAME  intel or openmpi (default: openmpi)
  --srun-mpi NAME       Explicit Slurm MPI plugin (default: site default)
  --dry-run             Print commands without checking or executing them
  -h, --help            Show this help

Examples:
  ./validate_m0.sh --launcher mpirun --nodes 1 --pes-per-node 2
  ./validate_m0.sh --launcher srun --nodes 2 --pes-per-node 1
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

require_value() {
    [[ $# -ge 2 ]] || die "option $1 requires a value"
}

print_command() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
}

default_build_dir=${NVSHMEM_M0_BUILD_DIR:-${SCRATCH:-$script_dir}/oneccl-nvshmem-m0-build}
binary=${NVSHMEM_M0_BINARY:-$default_build_dir/oneccl_nvshmem_m0}
launcher=${NVSHMEM_M0_LAUNCHER:-mpirun}
launcher_path=${NVSHMEM_M0_LAUNCHER_PATH:-}
nodes=${NVSHMEM_M0_NODES:-1}
pes_per_node=${NVSHMEM_M0_PES_PER_NODE:-}
counts=${NVSHMEM_M0_COUNTS:-1,1024,1048576}
mpirun_flavor=${NVSHMEM_M0_MPIRUN_FLAVOR:-openmpi}
srun_mpi=${NVSHMEM_M0_SRUN_MPI:-}
dry_run=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --binary)
            require_value "$@"
            binary=$2
            shift 2
            ;;
        --launcher)
            require_value "$@"
            launcher=$2
            shift 2
            ;;
        --launcher-path)
            require_value "$@"
            launcher_path=$2
            shift 2
            ;;
        --nodes)
            require_value "$@"
            nodes=$2
            shift 2
            ;;
        --pes-per-node)
            require_value "$@"
            pes_per_node=$2
            shift 2
            ;;
        --counts)
            require_value "$@"
            counts=$2
            shift 2
            ;;
        --mpirun-flavor)
            require_value "$@"
            mpirun_flavor=$2
            shift 2
            ;;
        --srun-mpi)
            require_value "$@"
            srun_mpi=$2
            shift 2
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

[[ "$nodes" =~ ^[1-9][0-9]*$ ]] || die "nodes must be a positive integer, got: $nodes"
if [[ -z "$pes_per_node" ]]; then
    if [[ "$nodes" == 1 ]]; then
        pes_per_node=2
    else
        pes_per_node=1
    fi
fi
[[ "$pes_per_node" =~ ^[1-9][0-9]*$ ]] ||
    die "PEs per node must be a positive integer, got: $pes_per_node"

case "$launcher" in
    mpirun|srun)
        ;;
    *)
        die "unsupported launcher: $launcher"
        ;;
esac

case "$mpirun_flavor" in
    intel|openmpi) ;;
    *) die "unsupported mpirun flavor: $mpirun_flavor" ;;
esac

if [[ -z "$launcher_path" ]]; then
    if [[ "$dry_run" == 1 ]]; then
        launcher_path=$launcher
    else
        launcher_path=$(command -v "$launcher") || die "$launcher is not available"
    fi
fi

if [[ "$dry_run" == 0 ]]; then
    [[ -x "$binary" ]] || die "M0 binary is not executable: $binary"
    [[ -x "$launcher_path" ]] || die "launcher is not executable: $launcher_path"
fi

IFS=',' read -r -a count_values <<< "$counts"
[[ ${#count_values[@]} -gt 0 ]] || die "at least one element count is required"
for count in "${count_values[@]}"; do
    [[ "$count" =~ ^[1-9][0-9]*$ ]] || die "invalid element count: $count"
done

total_pes=$((nodes * pes_per_node))

printf '%s\n' \
    "M0 binary:       $binary" \
    "Launcher:        $launcher_path" \
    "Bootstrap:       MPI" \
    "Slurm MPI:       ${srun_mpi:-<site default>}" \
    "Nodes:           $nodes" \
    "PEs per node:    $pes_per_node" \
    "Total PEs:       $total_pes" \
    "Element counts:  $counts"

for count in "${count_values[@]}"; do
    command=(env NVSHMEM_BOOTSTRAP=MPI "$launcher_path")
    case "$launcher" in
        mpirun)
            command+=(-np "$total_pes")
            if [[ "$mpirun_flavor" == openmpi ]]; then
                command+=(--map-by "ppr:${pes_per_node}:node")
            else
                command+=(-ppn "$pes_per_node")
            fi
            ;;
        srun)
            if [[ -n "$srun_mpi" ]]; then
                command+=(--mpi="$srun_mpi")
            fi
            command+=(--nodes="$nodes"
                      --ntasks="$total_pes"
                      --ntasks-per-node="$pes_per_node"
                      --gpus-per-task=1)
            ;;
    esac
    command+=("$binary" "$count")

    print_command "${command[@]}"
    if [[ "$dry_run" == 0 ]]; then
        printf 'Running M0 with %s float elements\n' "$count"
        "${command[@]}"
    fi
done
