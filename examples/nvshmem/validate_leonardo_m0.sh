#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
default_source_dir=$(cd -- "$script_dir/../.." && pwd)

usage() {
    cat <<'EOF'
Validate the NVSHMEM M0 spike in Leonardo's hybrid DPC++/NVHPC environment.

Usage:
  validate_leonardo_m0.sh [environment options] [-- <validate_m0.sh options>]

Environment options:
  --source-dir DIR      oneCCL source tree (default: repository containing this script)
  --env-script FILE     comm-playground Leonardo environment loader used by
                        leonardo_env.sh (default:
                        $COMM_PLAYGROUND_ROOT/cluster/leonardo/environment.sh)
  --skip-env            Use an already composed DPC++/CUDA/NVSHMEM environment
  --cuda-root DIR       Override the NVHPC CUDA installation captured from the CUDA stack
  --nvshmem-root DIR    Override the NVHPC NVSHMEM installation
  --dry-run             Pass dry-run mode to the generic validator
  -h, --help            Show this help

Arguments after `--` are passed to validate_m0.sh. Leonardo defaults to `srun`
with explicit MPI bootstrap and the site-default Slurm MPI integration.

Examples:
  ./examples/nvshmem/validate_leonardo_m0.sh -- \
    --nodes 1 --pes-per-node 2

  ./examples/nvshmem/validate_leonardo_m0.sh -- \
    --nodes 2 --pes-per-node 1
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

require_value() {
    [[ $# -ge 2 ]] || die "option $1 requires a value"
}

source_dir=${ONECCL_SOURCE_DIR:-$default_source_dir}
env_script=${ONECCL_ENV_SCRIPT:-}
cuda_root=
nvshmem_root=
load_environment=1
dry_run=0
validation_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source-dir)
            require_value "$@"
            source_dir=$2
            shift 2
            ;;
        --env-script)
            require_value "$@"
            env_script=$2
            shift 2
            ;;
        --skip-env)
            load_environment=0
            shift
            ;;
        --cuda-root)
            require_value "$@"
            cuda_root=$2
            shift 2
            ;;
        --nvshmem-root)
            require_value "$@"
            nvshmem_root=$2
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
        --)
            shift
            validation_args=("$@")
            break
            ;;
        *)
            die "unknown environment option: $1 (put validator options after --)"
            ;;
    esac
done

source_dir=$(cd -- "$source_dir" && pwd -P)
generic_validator=$source_dir/examples/nvshmem/validate_m0.sh
[[ -x "$generic_validator" ]] || die "M0 validator is not executable: $generic_validator"

if [[ "$load_environment" == 1 ]]; then
    env_script=${env_script:-${COMM_PLAYGROUND_ROOT:-$HOME/comm-playground}/cluster/leonardo/environment.sh}
    hybrid_env=$source_dir/examples/nvshmem/leonardo_env.sh
    [[ -f "$hybrid_env" ]] || die "Leonardo environment composer not found: $hybrid_env"
    source "$hybrid_env" "$env_script"
    cuda_root=${cuda_root:-${CUDA_ROOT:-}}
    nvshmem_root=${nvshmem_root:-${NVSHMEM_HOME:-}}
else
    cuda_root=${cuda_root:-${CUDA_ROOT:-${CUDA_HOME:-${CUDA_PATH:-}}}}
    nvshmem_root=${nvshmem_root:-${NVSHMEM_HOME:-${NVSHMEM_ROOT:-}}}
fi

if [[ "$dry_run" == 0 ]]; then
    [[ -n "$cuda_root" && -d "$cuda_root/lib64" ]] ||
        die "NVHPC CUDA runtime was not found; use --cuda-root"
    [[ -n "$nvshmem_root" && -f "$nvshmem_root/include/nvshmem.h" ]] ||
        die "NVHPC NVSHMEM installation was not found; use --nvshmem-root"

    mpi_plugin=
    for library_dir in "$nvshmem_root/lib" "$nvshmem_root/lib64"; do
        for candidate in "$library_dir"/libnvshmem_bootstrap_mpi.so* \
                         "$library_dir"/nvshmem_bootstrap_mpi.so*; do
            if [[ -e "$candidate" ]]; then
                mpi_plugin=$candidate
                break 2
            fi
        done
    done
    command -v mpicxx >/dev/null 2>&1 ||
        die "mpicxx is unavailable; source leonardo_env.sh to load HPC-X"
fi

export CUDA_HOME="$cuda_root"
export CUDA_ROOT="$cuda_root"
export CUDA_PATH="$cuda_root"
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export NVSHMEM_HOME="$nvshmem_root"
export NVSHMEM_ROOT="$nvshmem_root"
validation_nodes=${NVSHMEM_M0_NODES:-1}
for ((index = 0; index < ${#validation_args[@]}; index++)); do
    if [[ "${validation_args[$index]}" == --nodes &&
          $((index + 1)) -lt ${#validation_args[@]} ]]; then
        validation_nodes=${validation_args[$((index + 1))]}
    fi
done

export COMM_PLAYGROUND_JOB_NODES="$validation_nodes"
base_env=${env_script:-${ONECCL_LEONARDO_BASE_ENV:-}}
runtime_script=
if [[ -n "$base_env" ]]; then
    runtime_script=$(cd -- "$(dirname -- "$base_env")" && pwd)/runtime/nvshmem.sh
fi
if [[ -n "$runtime_script" && -f "$runtime_script" ]]; then
    source "$runtime_script"
else
    export NVSHMEM_REMOTE_TRANSPORT=${NVSHMEM_REMOTE_TRANSPORT:-ibrc}
    export NVSHMEM_IB_ENABLE_IBGDA=${NVSHMEM_IB_ENABLE_IBGDA:-0}
    export NVSHMEM_DISABLE_NCCL=${NVSHMEM_DISABLE_NCCL:-1}
    export NVSHMEM_IB_SL=${NVSHMEM_IB_SL:-1}
    export SHMEM_SYMMETRIC_SIZE=${SHMEM_SYMMETRIC_SIZE:-1G}
fi
export NVSHMEM_BOOTSTRAP=MPI

runtime_paths=("$nvshmem_root/lib" "$nvshmem_root/lib64" "$cuda_root/lib64")
if [[ -n "${MATH_LIBS:-}" ]]; then
    runtime_paths+=("$MATH_LIBS/lib")
fi
for runtime_path in "${runtime_paths[@]}"; do
    if [[ -d "$runtime_path" ]]; then
        export LD_LIBRARY_PATH="$runtime_path:${LD_LIBRARY_PATH:-}"
    fi
done
export PATH="$cuda_root/bin:$PATH"

printf '%s\n' \
    "Leonardo M0 source:   $source_dir" \
    "Environment loader:   ${env_script:-<already loaded>}" \
    "NVHPC CUDA root:      $cuda_root" \
    "NVHPC NVSHMEM root:   $nvshmem_root" \
    "MPI wrapper:          $(command -v mpicxx 2>/dev/null || printf '<dry run>')" \
    "Remote transport:     $NVSHMEM_REMOTE_TRANSPORT" \
    "MPI bootstrap plugin: ${mpi_plugin:-<resolved by NVSHMEM>}"

args=(--launcher srun)
if [[ "$dry_run" == 1 ]]; then
    args+=(--dry-run)
fi
if (( ${#validation_args[@]} > 0 )); then
    args+=("${validation_args[@]}")
fi

"$generic_validator" "${args[@]}"
