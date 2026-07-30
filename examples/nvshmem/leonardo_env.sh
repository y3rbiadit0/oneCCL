#!/usr/bin/env bash

# Compose comm-playground's validated Leonardo SYCL stack with NVHPC NVSHMEM.
# This file must be sourced so its module and environment changes persist.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf 'error: source this file instead of executing it\n' >&2
    printf 'usage: source %s [comm-playground-environment.sh]\n' "$0" >&2
    exit 2
fi

_oneccl_nvshmem_load_leonardo_env() {
    local base_env=${1:-${ONECCL_LEONARDO_BASE_ENV:-${COMM_PLAYGROUND_ROOT:-$HOME/comm-playground}/cluster/leonardo/environment.sh}}
    local mpi_module=${ONECCL_LEONARDO_MPI_MODULE:-hpcx-mpi/2.19}
    if [[ ! -f "$base_env" ]]; then
        printf 'error: comm-playground Leonardo environment loader not found: %s\n' "$base_env" >&2
        return 2
    fi

    source "$base_env" cuda

    local nvhpc_cuda_root=${NVHPC_CUDA:-${CUDA_ROOT:-${CUDA_HOME:-}}}
    local nvhpc_nvshmem_root=${NVSHMEM_HOME:-}
    local nvhpc_math_libs=${MATH_LIBS:-}
    local nvhpc_nccl_root=${NCCL_HOME:-}
    if [[ -z "$nvhpc_cuda_root" ]]; then
        printf 'error: comm-playground cuda stack did not set NVHPC_CUDA or CUDA_ROOT\n' >&2
        return 2
    fi
    if [[ -z "$nvhpc_nvshmem_root" ]]; then
        printf 'error: comm-playground cuda stack did not set NVSHMEM_HOME\n' >&2
        return 2
    fi

    source "$base_env" sycl

    if ! type module >/dev/null 2>&1; then
        printf 'error: the Leonardo module command is unavailable\n' >&2
        return 2
    fi
    module unload openmpi >/dev/null 2>&1 || true
    unset MPI_HOME MPI_ROOT OPENMPI_HOME OPAL_PREFIX
    module load "$mpi_module"

    local mpi_cxx
    mpi_cxx=$(command -v mpicxx) || {
        printf 'error: %s did not provide mpicxx\n' "$mpi_module" >&2
        return 2
    }

    export DPCPP_ROOT=${DPCPP_INSTALL:-${DPCPP_ROOT:-}}
    export NVHPC_CUDA="$nvhpc_cuda_root"
    export CUDA_HOME="$nvhpc_cuda_root"
    export CUDA_ROOT="$nvhpc_cuda_root"
    export CUDA_PATH="$nvhpc_cuda_root"
    export CUDACXX="$nvhpc_cuda_root/bin/nvcc"
    export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
    export NVSHMEM_HOME="$nvhpc_nvshmem_root"
    export NVSHMEM_ROOT="$nvhpc_nvshmem_root"
    export MATH_LIBS="$nvhpc_math_libs"
    export NCCL_HOME="$nvhpc_nccl_root"
    export MPI_CXX_COMPILER="$mpi_cxx"

    local runtime_path
    local runtime_paths=(
        "$nvhpc_nvshmem_root/lib"
        "$nvhpc_nvshmem_root/lib64"
        "$nvhpc_cuda_root/lib64"
    )
    if [[ -n "$nvhpc_nccl_root" ]]; then
        runtime_paths+=("$nvhpc_nccl_root/lib" "$nvhpc_nccl_root/lib64")
    fi
    if [[ -n "$nvhpc_math_libs" ]]; then
        runtime_paths+=("$nvhpc_math_libs/lib")
    fi
    for runtime_path in "${runtime_paths[@]}"; do
        if [[ -n "$runtime_path" && -d "$runtime_path" ]]; then
            export LD_LIBRARY_PATH="$runtime_path:${LD_LIBRARY_PATH:-}"
        fi
    done

    # DPC++'s CUDA adapter requires the newer user-provided hwloc at runtime.
    if [[ -n "${HWLOC_ROOT:-}" && -d "$HWLOC_ROOT/lib" ]]; then
        export LD_LIBRARY_PATH="$HWLOC_ROOT/lib:${LD_LIBRARY_PATH:-}"
    fi

    export PATH="${DPCPP_ROOT:+$DPCPP_ROOT/bin:}$nvhpc_cuda_root/bin:$PATH"
    export ONECCL_NVSHMEM_LEONARDO_ENV=1
    export ONECCL_LEONARDO_BASE_ENV="$base_env"
    export ONECCL_LEONARDO_MPI_MODULE="$mpi_module"
}

_oneccl_nvshmem_load_leonardo_env "$@"
unset -f _oneccl_nvshmem_load_leonardo_env
