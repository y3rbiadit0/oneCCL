#!/usr/bin/env bash
set -euo pipefail

readonly OSHMPI_PINNED_COMMIT=ee5cf110e673c098707257bb025404e17ac0a5fc
readonly OSHMPI_PINNED_OPENPA=0475704dde41054db33562a8d17314fe0e30aaf3
readonly OSHMPI_PINNED_SHORT=ee5cf110
readonly OSHMPI_PATCH_REVISION=2

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd -- "$script_dir/../.." && pwd)
playground_root=${COMM_PLAYGROUND_ROOT:-$HOME/Projects/hpc-comm-playground}
source "$playground_root/cluster/leonardo/environment.sh" cuda

base_source=${OSHMPI_BASE_SOURCE_DIR:-$HOME/opt-src/oshmpi-main}
source_dir=${OSHMPI_SOURCE_DIR:-$SCRATCH/oshmpi-$OSHMPI_PINNED_SHORT-oneccl-patch$OSHMPI_PATCH_REVISION-src}
build_dir=${OSHMPI_BUILD_DIR:-$SCRATCH/oshmpi-$OSHMPI_PINNED_SHORT-oneccl-patch$OSHMPI_PATCH_REVISION-build}
install_prefix=${OSHMPI_INSTALL_PREFIX:-$HOME/opt/oshmpi-$OSHMPI_PINNED_SHORT-oneccl}
patch_file="$project_root/examples/oshmpi/patches/0001-preserve-external-mpi-ownership.patch"

# The patch is what makes OSHMPI leave an externally initialized MPI alone and
# stop MPI_T_init_thread from lowering HPC-X to MPI_THREAD_SINGLE. It is only
# needed when the worktree is not already patched, so the hard requirement lives
# at the point of use below; report a missing file here only as a warning.
missing_patch_message() {
    printf 'the OSHMPI external-MPI ownership patch is not in the repository: %s\n' \
        "$patch_file" >&2
    printf 'regenerate it from an already-patched worktree and commit it:\n' >&2
    printf '  git -C <patched-oshmpi-worktree> diff %s -- . ":(exclude)src/openpa" > %s\n' \
        "$OSHMPI_PINNED_COMMIT" "$patch_file" >&2
}

if [[ ! -f "$patch_file" ]]; then
    printf 'warning: ' >&2
    missing_patch_message
fi

if [[ ! -d "$base_source/.git" ]]; then
    printf 'error: base OSHMPI checkout not found: %s\n' "$base_source" >&2
    exit 2
fi

base_commit=$(git -C "$base_source" rev-parse HEAD)
if [[ "$base_commit" != "$OSHMPI_PINNED_COMMIT" ]]; then
    printf 'error: base OSHMPI checkout is at %s, expected %s\n' \
        "$base_commit" "$OSHMPI_PINNED_COMMIT" >&2
    exit 2
fi

if [[ ! -e "$source_dir" ]]; then
    mkdir -p "$(dirname "$source_dir")"
    git -C "$base_source" worktree add --detach "$source_dir" "$OSHMPI_PINNED_COMMIT"
fi

if ! git -C "$source_dir" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    printf 'error: OSHMPI worktree is invalid: %s\n' "$source_dir" >&2
    exit 2
fi

actual_commit=$(git -C "$source_dir" rev-parse HEAD)
if [[ "$actual_commit" != "$OSHMPI_PINNED_COMMIT" ]]; then
    printf 'error: OSHMPI worktree is at %s, expected %s\n' \
        "$actual_commit" "$OSHMPI_PINNED_COMMIT" >&2
    exit 2
fi

git -C "$source_dir" submodule update --init --recursive
openpa_commit=$(git -C "$source_dir/src/openpa" rev-parse HEAD)
if [[ "$openpa_commit" != "$OSHMPI_PINNED_OPENPA" ]]; then
    printf 'error: OpenPA is at %s, expected %s\n' \
        "$openpa_commit" "$OSHMPI_PINNED_OPENPA" >&2
    exit 2
fi

if grep -q 'OSHMPI_PRESERVE_EXTERNAL_MPI' "$source_dir/include/shmem.h.in.tpl"; then
    if ! grep -Fq 'MPI_T_init_thread(mpi_provided, &mpit_provided)' \
        "$source_dir/src/internal/setup_impl.c"; then
        printf 'error: OSHMPI worktree contains an older ownership patch: %s\n' \
            "$source_dir" >&2
        printf 'use a fresh OSHMPI_SOURCE_DIR and OSHMPI_BUILD_DIR\n' >&2
        exit 2
    fi
    printf 'ownership patch already applied in %s\n' "$source_dir"
    if [[ ! -f "$patch_file" ]]; then
        printf 'note: this worktree is the only copy of the patch - capture it with:\n' >&2
        printf '  git -C %s diff %s -- . ":(exclude)src/openpa" > %s\n' \
            "$source_dir" "$OSHMPI_PINNED_COMMIT" "$patch_file" >&2
    fi
else
    if [[ ! -f "$patch_file" ]]; then
        printf 'error: ' >&2
        missing_patch_message
        exit 2
    fi
    git -C "$source_dir" apply --check "$patch_file"
    git -C "$source_dir" apply "$patch_file"
fi

mkdir -p "$build_dir"
if [[ ! -x "$source_dir/configure" ]]; then
    (
        cd "$source_dir"
        ./autogen.sh
    )
fi

(
    cd "$build_dir"
    "$source_dir/configure" \
        --prefix="$install_prefix" \
        --enable-threads=multiple \
        --enable-async-thread=no \
        --enable-cuda=yes \
        --with-cuda="$CUDA_ROOT" \
        CC="$(command -v mpicc)" \
        CXX="$(command -v mpicxx)"
)

make -C "$build_dir" -j "${OSHMPI_BUILD_JOBS:-8}"
make -C "$build_dir" install

smoke_dir="$install_prefix/tests"
mkdir -p "$smoke_dir"
"$install_prefix/bin/oshc++" \
    -std=c++11 \
    -L"$CUDA_ROOT/lib64/stubs" \
    "$project_root/examples/oshmpi/oshmpi_mpi_ownership_smoke.cpp" \
    -o "$smoke_dir/oshmpi_mpi_ownership_smoke"

grep -q '^#define OSHMPI_PRESERVE_EXTERNAL_MPI 1$' "$install_prefix/include/shmem.h"
if ! nm -D "$install_prefix/lib/liboshmpi.so" | grep -q ' oshmpi_preserves_external_mpi$'; then
    printf 'error: patched ownership symbol is absent from installed liboshmpi\n' >&2
    exit 2
fi

printf 'patched OSHMPI installed at %s\n' "$install_prefix"
printf 'ownership smoke test: %s\n' "$smoke_dir/oshmpi_mpi_ownership_smoke"
printf 'export OSHMPI_HOME=%q\n' "$install_prefix"
printf 'export OSHMPI_INSTALL_PREFIX=%q\n' "$install_prefix"
