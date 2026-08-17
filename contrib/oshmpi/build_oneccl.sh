#!/usr/bin/env bash
set -euo pipefail

# Builds oneCCL with the OSHMPI backend. This script loads no modules and sources
# nothing: the caller supplies a prepared environment through the variables below,
# so it works on any site rather than one.
#
# Required:
#   ONECCL_C_COMPILER    SYCL-capable C compiler
#   ONECCL_CXX_COMPILER  SYCL-capable C++ compiler
#   ONECCL_SYCL_FLAGS    -fsycl flags naming the target, e.g.
#                        "-fsycl -fsycl-targets=nvptx64-nvidia-cuda"
#   MPI_C_COMPILER       the mpicc OSHMPI was built against
#   OSHMPI_ROOT          a patched OSHMPI install (see build_oshmpi.sh)
#
# Optional:
#   ONECCL_BUILD_ROOT    parent for the build tree; defaults to $SCRATCH
#   ONECCL_BUILD_DIR / ONECCL_INSTALL_PREFIX
#   ONECCL_OSHMPI_PINNED_STAGING  ON (default) pins the staging arena with CUDA
#   ONECCL_BUILD_EXAMPLES / ONECCL_BUILD_JOBS

oneccl_source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)

oneccl_c_compiler=${ONECCL_C_COMPILER:?set to a SYCL-capable C compiler}
oneccl_cxx_compiler=${ONECCL_CXX_COMPILER:?set to a SYCL-capable C++ compiler}
oneccl_sycl_flags=${ONECCL_SYCL_FLAGS:?set to the -fsycl flags for your target}
oneccl_mpi_c_compiler=${MPI_C_COMPILER:?set to the mpicc OSHMPI was built against}
oneccl_oshmpi_root=${OSHMPI_ROOT:?set to a patched OSHMPI install}

# Builds always land on a scratch filesystem, never in the source tree or $HOME.
oneccl_build_root=${ONECCL_BUILD_ROOT:-${SCRATCH:?set SCRATCH or ONECCL_BUILD_ROOT to a build filesystem}}
oneccl_build_dir=${ONECCL_BUILD_DIR:-$oneccl_build_root/oneccl-oshmpi}
oneccl_install_prefix=${ONECCL_INSTALL_PREFIX:-$HOME/opt/oneccl-oshmpi}

shmem_header="$oneccl_oshmpi_root/include/shmem.h"
if [[ ! -f "$shmem_header" ]]; then
    printf 'error: OSHMPI header not found: %s\n' "$shmem_header" >&2
    exit 2
fi
if ! grep -q '^#define OSHMPI_PRESERVE_EXTERNAL_MPI 1$' "$shmem_header"; then
    printf 'error: OSHMPI lacks the external MPI ownership patch: %s\n' "$shmem_header" >&2
    printf 'build it with contrib/oshmpi/build_oshmpi.sh\n' >&2
    exit 2
fi
oneccl_cache="$oneccl_build_dir/CMakeCache.txt"

if [[ -f "$oneccl_cache" ]]; then
    configured_source=$(grep '^CMAKE_HOME_DIRECTORY:INTERNAL=' "$oneccl_cache" || true)
    configured_source=${configured_source#*=}
    if [[ "$configured_source" != "$oneccl_source_dir" ]]; then
        printf 'error: build directory belongs to another source tree: %s\n' \
            "$configured_source" >&2
        printf 'set ONECCL_BUILD_DIR to a fresh directory\n' >&2
        exit 2
    fi
    configured_sycl=$(grep '^CCL_ENABLE_SYCL:BOOL=' "$oneccl_cache" || true)
    configured_sycl=${configured_sycl#*=}
    if [[ "$configured_sycl" != ON ]]; then
        printf 'error: build directory was configured with CCL_ENABLE_SYCL=%s\n' \
            "$configured_sycl" >&2
        printf 'set ONECCL_BUILD_DIR to a fresh directory\n' >&2
        exit 2
    fi
    configured_cxx=$(grep '^CMAKE_CXX_COMPILER:FILEPATH=' "$oneccl_cache" || true)
    configured_cxx=${configured_cxx#*=}
    if [[ "$configured_cxx" != "$oneccl_cxx_compiler" ]]; then
        printf 'error: build directory uses another C++ compiler: %s\n' \
            "$configured_cxx" >&2
        printf 'set ONECCL_BUILD_DIR to a fresh directory\n' >&2
        exit 2
    fi
fi

oneccl_cmake_args=(
    -DCMAKE_BUILD_TYPE=Release
    "-DCMAKE_C_COMPILER=$oneccl_c_compiler"
    "-DCMAKE_CXX_COMPILER=$oneccl_cxx_compiler"
    "-DCMAKE_INSTALL_PREFIX=$oneccl_install_prefix"
    "-DMPI_C_COMPILER=$oneccl_mpi_c_compiler"
    "-DOSHMPI_ROOT=$oneccl_oshmpi_root"
    -DCCL_ENABLE_OSHMPI=ON
    "-DCCL_ENABLE_OSHMPI_PINNED_STAGING=${ONECCL_OSHMPI_PINNED_STAGING:-ON}"
    -DCCL_ENABLE_NCCL=OFF
    -DCCL_ENABLE_RCCL=OFF
    -DCCL_ENABLE_ZE=OFF
)

oneccl_cmake_args+=(
    -DCCL_ENABLE_SYCL=ON
    -DCOMPUTE_BACKEND=dpcpp
    "-DCMAKE_CXX_FLAGS=$oneccl_sycl_flags"
    "-DCMAKE_CXX_FLAGS_RELEASE=$oneccl_sycl_flags -O3 -DNDEBUG"
)

cmake -S "$oneccl_source_dir" -B "$oneccl_build_dir" -G Ninja \
    "${oneccl_cmake_args[@]}" \
    -DENABLE_MPI=OFF \
    -DENABLE_MPI_TESTS=ON \
    -DENABLE_OMP=OFF \
    -DENABLE_ITT=OFF \
    -DENABLE_PROFILING=OFF \
    -DENABLE_PMIX=ON \
    -DENABLE_UMF=OFF \
    -DENABLE_STUB_BACKEND=OFF \
    -DENABLE_SYCL_INTEROP_EVENT=OFF \
    -DENABLE_OFI_HMEM=OFF \
    -DUSE_SECURITY_FLAGS=ON \
    -DBUILD_EXAMPLES=${ONECCL_BUILD_EXAMPLES:-OFF} \
    -DBUILD_FT=OFF \
    -DBUILD_CONFIG=OFF

configured_project=$(grep '^CMAKE_PROJECT_NAME:STATIC=' "$oneccl_cache" || true)
configured_project=${configured_project#*=}
configured_source=$(grep '^CMAKE_HOME_DIRECTORY:INTERNAL=' "$oneccl_cache" || true)
configured_source=${configured_source#*=}
configured_oshmpi=$(grep '^CCL_ENABLE_OSHMPI:BOOL=' "$oneccl_cache" || true)
configured_oshmpi=${configured_oshmpi#*=}
configured_native_mpi=$(grep '^ENABLE_MPI:BOOL=' "$oneccl_cache" || true)
configured_native_mpi=${configured_native_mpi#*=}
configured_stub=$(grep '^ENABLE_STUB_BACKEND:BOOL=' "$oneccl_cache" || true)
configured_stub=${configured_stub#*=}
if [[ "$configured_project" != oneCCL ||
      "$configured_source" != "$oneccl_source_dir" ||
      "$configured_oshmpi" != ON ||
      "$configured_native_mpi" != OFF ||
      "$configured_stub" != OFF ]]; then
    printf 'error: oneCCL OSHMPI CMake configuration is invalid\n' >&2
    printf 'project=%s source=%s OSHMPI=%s native_MPI=%s stub=%s\n' \
        "$configured_project" \
        "$configured_source" \
        "$configured_oshmpi" \
        "$configured_native_mpi" \
        "$configured_stub" >&2
    exit 2
fi

cmake --build "$oneccl_build_dir" --parallel "${ONECCL_BUILD_JOBS:-8}"

oneccl_test="$oneccl_build_dir/tests/oshmpi/oshmpi_collectives_test"
if [[ ! -x "$oneccl_test" ]]; then
    printf 'error: OSHMPI test executable was not built: %s\n' "$oneccl_test" >&2
    exit 2
fi

cmake --install "$oneccl_build_dir"

printf 'oneCCL OSHMPI build: %s\n' "$oneccl_build_dir"
printf 'oneCCL OSHMPI install: %s\n' "$oneccl_install_prefix"
printf 'oneCCL OSHMPI test: %s\n' "$oneccl_test"
printf 'oneCCL compilers: %s, %s\n' "$oneccl_c_compiler" "$oneccl_cxx_compiler"
