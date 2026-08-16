#!/usr/bin/env bash

# Compose the validated comm-playground CUDA/HPC-X stack with patched OSHMPI.
# Source this file so module and environment changes persist.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf 'error: source this file instead of executing it\n' >&2
    printf 'usage: source %s [comm-playground-root]\n' "$0" >&2
    exit 2
fi

_oneccl_oshmpi_load_leonardo_env() {
    local playground_root=${1:-${COMM_PLAYGROUND_ROOT:-$HOME/Projects/hpc-comm-playground}}
    local environment="$playground_root/cluster/leonardo/environment.sh"
    if [[ ! -f "$environment" ]]; then
        printf 'error: comm-playground Leonardo environment not found: %s\n' "$environment" >&2
        return 2
    fi

    source "$environment" sycl

    # Only the cuda stack exports OSHMPI_HOME; under sycl the caller points at a
    # matching OSHMPI build itself.
    : "${OSHMPI_HOME:?set OSHMPI_HOME to an OSHMPI built against the MPI of this stack}"
    local shmem_header="$OSHMPI_HOME/include/shmem.h"
    if [[ ! -f "$shmem_header" ]]; then
        printf 'error: patched OSHMPI header not found: %s\n' "$shmem_header" >&2
        return 2
    fi
    if ! grep -q '^#define OSHMPI_PRESERVE_EXTERNAL_MPI 1$' "$shmem_header"; then
        printf 'error: OSHMPI does not include the external MPI ownership patch: %s\n' \
            "$shmem_header" >&2
        return 2
    fi

    local mpi_cxx
    mpi_cxx=$(command -v mpicxx) || {
        printf 'error: mpicxx is unavailable\n' >&2
        return 2
    }
    export MPI_CXX_COMPILER="$mpi_cxx"
    export MPI_ROOT=${MPI_ROOT:-$(dirname "$(dirname "$mpi_cxx")")}
    export OSHMPI_ROOT="$OSHMPI_HOME"
    export CCL_MPI_LIBRARY_PATH=${CCL_MPI_LIBRARY_PATH:-$MPI_ROOT/lib/libmpi.so}
    export CCL_BACKEND=oshmpi
    export CCL_OSHMPI_STAGING_SIZE=${CCL_OSHMPI_STAGING_SIZE:-64M}
    export SHMEM_SYMMETRIC_SIZE=${SHMEM_SYMMETRIC_SIZE:-1G}
    export ONECCL_OSHMPI_LEONARDO_ENV=1
}

_oneccl_oshmpi_load_leonardo_env "$@"
unset -f _oneccl_oshmpi_load_leonardo_env
