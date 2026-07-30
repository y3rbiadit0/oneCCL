#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
default_source_dir=$(cd -- "$script_dir/../.." && pwd)

usage() {
    cat <<'EOF'
Build the NVSHMEM M0 spike with Leonardo's DPC++ and NVHPC 24.5 installations.

Usage:
  build_leonardo_m0.sh [options] [-- <extra CMake arguments>]

Options:
  --source-dir DIR      oneCCL source tree (default: repository containing this script)
  --build-dir DIR       Build directory (default: $SCRATCH/oneccl-nvshmem-m0-build)
  --env-script FILE     comm-playground Leonardo environment loader used by
                        leonardo_env.sh (default:
                        $COMM_PLAYGROUND_ROOT/cluster/leonardo/environment.sh)
  --skip-env            Use an already composed DPC++/CUDA/NVSHMEM environment
  --dpcpp-root DIR      Override the DPC++ installation captured from the SYCL stack
  --cuda-root DIR       Override the NVHPC CUDA 12 installation captured from the CUDA stack
  --nvshmem-root DIR    Override the NVHPC NVSHMEM installation
  --mpi-cxx FILE        Override the HPC-X MPI C++ wrapper
  --cuda-arch ARCH      CUDA architecture without sm_ prefix (default: 80)
  -j, --jobs N          Parallel build jobs (default: $SLURM_CPUS_PER_TASK or 16)
  --clean               Remove the M0 build directory before configuring (default)
  --no-clean            Reuse the existing build directory
  --configure-only      Configure without building
  --dry-run             Print generated commands without checking paths or executing them
  -h, --help            Show this help

The local leonardo_env.sh composes comm-playground's `cuda` and `sycl` stacks.

Example:
  ./examples/nvshmem/build_leonardo_m0.sh \
    --env-script "$COMM_PLAYGROUND_ROOT/cluster/leonardo/environment.sh"
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
build_dir=${NVSHMEM_M0_BUILD_DIR:-}
env_script=${ONECCL_ENV_SCRIPT:-}
dpcpp_root=
cuda_root=
nvshmem_root=
mpi_cxx=
cuda_arch=${NVSHMEM_M0_CUDA_ARCHITECTURE:-80}
jobs=${NVSHMEM_M0_JOBS:-${SLURM_CPUS_PER_TASK:-16}}
load_environment=1
clean=1
configure_only=0
dry_run=0
extra_cmake_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source-dir)
            require_value "$@"
            source_dir=$2
            shift 2
            ;;
        --build-dir)
            require_value "$@"
            build_dir=$2
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
        --dpcpp-root)
            require_value "$@"
            dpcpp_root=$2
            shift 2
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
        --mpi-cxx)
            require_value "$@"
            mpi_cxx=$2
            shift 2
            ;;
        --cuda-arch)
            require_value "$@"
            cuda_arch=${2#sm_}
            shift 2
            ;;
        -j|--jobs)
            require_value "$@"
            jobs=$2
            shift 2
            ;;
        --clean)
            clean=1
            shift
            ;;
        --no-clean)
            clean=0
            shift
            ;;
        --configure-only)
            configure_only=1
            shift
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
            extra_cmake_args=("$@")
            break
            ;;
        *)
            die "unknown option: $1 (put extra CMake arguments after --)"
            ;;
    esac
done

[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || die "jobs must be a positive integer, got: $jobs"
[[ "$cuda_arch" =~ ^[0-9]+$ ]] || die "CUDA architecture must be numeric, got: $cuda_arch"

source_dir=$(cd -- "$source_dir" && pwd -P)
generic_builder=$source_dir/examples/nvshmem/build_m0.sh
[[ -x "$generic_builder" ]] || die "M0 build driver is not executable: $generic_builder"
build_dir=${build_dir:-${SCRATCH:-$source_dir}/oneccl-nvshmem-m0-build}

if [[ "$load_environment" == 1 ]]; then
    env_script=${env_script:-${COMM_PLAYGROUND_ROOT:-$HOME/comm-playground}/cluster/leonardo/environment.sh}
    hybrid_env=$source_dir/examples/nvshmem/leonardo_env.sh
    [[ -f "$hybrid_env" ]] || die "Leonardo environment composer not found: $hybrid_env"
    source "$hybrid_env" "$env_script"
    dpcpp_root=${dpcpp_root:-${DPCPP_INSTALL:-${DPCPP_ROOT:-}}}
    cuda_root=${cuda_root:-${CUDA_ROOT:-}}
    nvshmem_root=${nvshmem_root:-${NVSHMEM_HOME:-}}
    mpi_cxx=${mpi_cxx:-${MPI_CXX_COMPILER:-${MPICXX:-mpicxx}}}
else
    dpcpp_root=${dpcpp_root:-${DPCPP_INSTALL:-${DPCPP_ROOT:-}}}
    cuda_root=${cuda_root:-${CUDA_ROOT:-${CUDA_HOME:-${CUDA_PATH:-}}}}
    nvshmem_root=${nvshmem_root:-${NVSHMEM_HOME:-${NVSHMEM_ROOT:-}}}
    mpi_cxx=${mpi_cxx:-${MPI_CXX_COMPILER:-${MPICXX:-mpicxx}}}
fi

export DPCPP_ROOT="$dpcpp_root"
export DPCPP_INSTALL="$dpcpp_root"
export CUDA_HOME="$cuda_root"
export CUDA_ROOT="$cuda_root"
export CUDA_PATH="$cuda_root"
export CUDACXX="$cuda_root/bin/nvcc"
export NVSHMEM_HOME="$nvshmem_root"
export NVSHMEM_ROOT="$nvshmem_root"
export MPI_CXX_COMPILER="$mpi_cxx"
export NVSHMEM_M0_BUILD_DIR="$build_dir"
export NVSHMEM_M0_CUDA_ARCHITECTURE="$cuda_arch"

runtime_paths=("$nvshmem_root/lib" "$nvshmem_root/lib64" "$cuda_root/lib64")
if [[ -n "${MATH_LIBS:-}" ]]; then
    runtime_paths+=("$MATH_LIBS/lib")
fi
for runtime_path in "${runtime_paths[@]}"; do
    if [[ -d "$runtime_path" ]]; then
        export LD_LIBRARY_PATH="$runtime_path:${LD_LIBRARY_PATH:-}"
    fi
done
export PATH="$dpcpp_root/bin:$cuda_root/bin:$PATH"

printf '%s\n' \
    "Leonardo M0 source:   $source_dir" \
    "Environment loader:   ${env_script:-<already loaded>}" \
    "DPC++ root:           $dpcpp_root" \
    "NVHPC CUDA root:      $cuda_root" \
    "NVHPC NVSHMEM root:   $nvshmem_root" \
    "HPC-X MPI wrapper:    $mpi_cxx" \
    "Build directory:      $build_dir" \
    "CUDA architecture:    $cuda_arch"

build_args=(
    --build-dir "$build_dir"
    --dpcpp-root "$dpcpp_root"
    --cuda-root "$cuda_root"
    --nvshmem-root "$nvshmem_root"
    --mpi-cxx "$mpi_cxx"
    --cuda-arch "$cuda_arch"
    --jobs "$jobs"
)
if [[ "$clean" == 0 ]]; then
    build_args+=(--no-clean)
fi
if [[ "$configure_only" == 1 ]]; then
    build_args+=(--configure-only)
fi
if [[ "$dry_run" == 1 ]]; then
    build_args+=(--dry-run)
fi
if (( ${#extra_cmake_args[@]} > 0 )); then
    build_args+=(-- "${extra_cmake_args[@]}")
fi

"$generic_builder" "${build_args[@]}"

if [[ "$dry_run" == 0 && "$configure_only" == 0 ]]; then
    binary=$build_dir/oneccl_nvshmem_m0
    printf 'Leonardo M0 binary: %s\n' "$binary"
    if [[ -e /dev/nvidiactl ]]; then
        if [[ -x "$nvshmem_root/bin/nvshmem-info" ]]; then
            "$nvshmem_root/bin/nvshmem-info" -v || true
        fi
        if command -v ldd >/dev/null 2>&1; then
            ldd "$binary"
        fi
    else
        printf '%s\n' \
            "GPU runtime diagnostics skipped: /dev/nvidiactl is unavailable." \
            "Run validate_leonardo_m0.sh inside a GPU allocation."
    fi
fi
