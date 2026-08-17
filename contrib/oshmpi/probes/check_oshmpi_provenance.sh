#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd -- "$script_dir/../../.." && pwd)
playground_root=${COMM_PLAYGROUND_ROOT:-$HOME/Projects/hpc-comm-playground}
source "$playground_root/cluster/leonardo/environment.sh" sycl

source_dir=${1:-${OSHMPI_SOURCE_DIR:-}}

printf 'OSHMPI_HOME=%s\n' "${OSHMPI_HOME:-unset}"
printf 'oshcc=%s\n' "$(command -v oshcc || true)"
printf 'mpicc=%s\n' "$(command -v mpicc || true)"
printf 'mpicxx=%s\n' "$(command -v mpicxx || true)"

if [[ -f "${OSHMPI_HOME:-}/include/shmem.h" ]]; then
    grep -E '^#define OSHMPI_(VERSION|NUMVERSION|BUILD_INFO|PRESERVE_EXTERNAL_MPI)' \
        "$OSHMPI_HOME/include/shmem.h" || true
else
    printf 'installed shmem.h not found\n'
fi

oshmpi_library=
for candidate in "${OSHMPI_HOME:-}/lib/liboshmpi.so" "${OSHMPI_HOME:-}/lib64/liboshmpi.so"; do
    if [[ -f "$candidate" ]]; then
        oshmpi_library=$candidate
        break
    fi
done
if [[ -n "$oshmpi_library" ]]; then
    printf 'liboshmpi=%s\n' "$(readlink -f "$oshmpi_library")"
    ldd "$oshmpi_library"
else
    printf 'liboshmpi.so not found under OSHMPI_HOME\n'
fi

if [[ -n "$source_dir" ]]; then
    if [[ ! -d "$source_dir/.git" ]]; then
        printf 'error: OSHMPI source is not a git checkout: %s\n' "$source_dir" >&2
        exit 2
    fi
    printf 'source=%s\n' "$(cd -- "$source_dir" && pwd)"
    git -C "$source_dir" rev-parse HEAD
    git -C "$source_dir" status --short --branch
    git -C "$source_dir" submodule status --recursive
else
    printf 'source=unknown (rerun with the OSHMPI source checkout as argument)\n'
fi

printf 'ownership_patch=%s\n' \
    "$project_root/contrib/oshmpi/patches/0001-preserve-external-mpi-ownership.patch"
