#!/usr/bin/env bash
set -euo pipefail

readonly OSHMPI_PINNED_COMMIT=ee5cf110e673c098707257bb025404e17ac0a5fc
readonly OSHMPI_PINNED_OPENPA=0475704dde41054db33562a8d17314fe0e30aaf3
readonly OSHMPI_PINNED_SHORT=ee5cf110
readonly OSHMPI_PATCH_REVISION=2

# Clones, patches and builds the OSHMPI the oneCCL backend requires. This script
# loads no modules and sources nothing: the caller supplies a prepared environment
# through the variables below, so it works on any site rather than one.
#
# Required:
#   MPI_C_COMPILER    mpicc for the MPI OSHMPI should sit on. oneCCL must later be
#                     pointed at the same one.
#   MPI_CXX_COMPILER  matching mpicxx
#
# Optional:
#   OSHMPI_BUILD_ROOT parent for source, build and clone trees; defaults to $SCRATCH
#   OSHMPI_CUDA_ROOT  CUDA install; enables OSHMPI's CUDA support when set
#   OSHMPI_INSTALL_PREFIX / OSHMPI_SOURCE_DIR / OSHMPI_BUILD_DIR
#   OSHMPI_UPSTREAM   clone URL, for mirrors or offline sites
#   OSHMPI_BUILD_JOBS

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)

mpi_c_compiler=${MPI_C_COMPILER:?set to the mpicc OSHMPI should be built against}
mpi_cxx_compiler=${MPI_CXX_COMPILER:?set to the matching mpicxx}

# Everything is built on a scratch filesystem, including the clone: nothing is
# written to the source tree or to $HOME except the final install prefix.
build_root=${OSHMPI_BUILD_ROOT:-${SCRATCH:?set SCRATCH or OSHMPI_BUILD_ROOT to a build filesystem}}
upstream=${OSHMPI_UPSTREAM:-https://github.com/pmodels/oshmpi.git}
base_source=${OSHMPI_BASE_SOURCE_DIR:-$build_root/oshmpi-upstream}
source_dir=${OSHMPI_SOURCE_DIR:-$build_root/oshmpi-$OSHMPI_PINNED_SHORT-oneccl-patch$OSHMPI_PATCH_REVISION-src}
build_dir=${OSHMPI_BUILD_DIR:-$build_root/oshmpi-$OSHMPI_PINNED_SHORT-oneccl-patch$OSHMPI_PATCH_REVISION-build}
install_prefix=${OSHMPI_INSTALL_PREFIX:-$HOME/opt/oshmpi-$OSHMPI_PINNED_SHORT-oneccl}
patch_file="$project_root/contrib/oshmpi/patches/0001-preserve-external-mpi-ownership.patch"

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

# Clone on first run. The pinned commit is fetched explicitly rather than relying
# on whatever the default branch points at today.
if [[ ! -d "$base_source/.git" ]]; then
    printf 'cloning OSHMPI %s into %s\n' "$OSHMPI_PINNED_SHORT" "$base_source"
    mkdir -p "$(dirname "$base_source")"
    git clone --quiet "$upstream" "$base_source"
fi

if ! git -C "$base_source" cat-file -e "$OSHMPI_PINNED_COMMIT^{commit}" 2>/dev/null; then
    git -C "$base_source" fetch --quiet origin "$OSHMPI_PINNED_COMMIT" ||
        git -C "$base_source" fetch --quiet origin
fi
if ! git -C "$base_source" cat-file -e "$OSHMPI_PINNED_COMMIT^{commit}" 2>/dev/null; then
    printf 'error: pinned OSHMPI commit %s is not reachable from %s\n' \
        "$OSHMPI_PINNED_COMMIT" "$upstream" >&2
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

cuda_configure_args=(--enable-cuda=no)
if [[ -n "${OSHMPI_CUDA_ROOT:-}" ]]; then
    cuda_configure_args=(--enable-cuda=yes "--with-cuda=$OSHMPI_CUDA_ROOT")
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
        "${cuda_configure_args[@]}" \
        CC="$mpi_c_compiler" \
        CXX="$mpi_cxx_compiler"
)

make -C "$build_dir" -j "${OSHMPI_BUILD_JOBS:-8}"
make -C "$build_dir" install

smoke_dir="$install_prefix/tests"
mkdir -p "$smoke_dir"
"$install_prefix/bin/oshc++" \
    -std=c++11 \
    ${OSHMPI_CUDA_ROOT:+-L"$OSHMPI_CUDA_ROOT/lib64/stubs"} \
    "$project_root/contrib/oshmpi/patches/ownership_smoke.cpp" \
    -o "$smoke_dir/ownership_smoke"

grep -q '^#define OSHMPI_PRESERVE_EXTERNAL_MPI 1$' "$install_prefix/include/shmem.h"

# The smoke test above links against oshmpi_preserves_external_mpi, so a successful
# link already proves the symbol is exported. This is a belt-and-braces check, and
# it must not report a missing symbol when the real problem is that nm could not be
# run or could not read the library.
oshmpi_library="$install_prefix/lib/liboshmpi.so"
if ! command -v nm >/dev/null 2>&1; then
    printf 'warning: nm not found in PATH; skipping the exported-symbol check\n' >&2
else
    if ! nm_output=$(nm -D "$oshmpi_library" 2>&1); then
        printf 'error: could not read symbols from %s\n' "$oshmpi_library" >&2
        printf '%s\n' "$nm_output" >&2
        exit 2
    fi
    # Defined symbols start with an address; undefined ones start with blanks.
    if ! printf '%s\n' "$nm_output" |
        grep -q '^[^[:space:]].*[[:space:]]oshmpi_preserves_external_mpi$'; then
        printf 'error: patched ownership symbol is absent from installed liboshmpi: %s\n' \
            "$oshmpi_library" >&2
        printf 'matching entries in nm -D output:\n' >&2
        printf '%s\n' "$nm_output" | grep -i 'preserve' >&2 ||
            printf '  (none - nm reported %s symbols in total)\n' \
                "$(printf '%s\n' "$nm_output" | grep -c .)" >&2
        exit 2
    fi
fi

printf 'patched OSHMPI installed at %s\n' "$install_prefix"
printf 'ownership smoke test: %s\n' "$smoke_dir/ownership_smoke"
printf 'export OSHMPI_HOME=%q\n' "$install_prefix"
printf 'export OSHMPI_INSTALL_PREFIX=%q\n' "$install_prefix"
