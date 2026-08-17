#!/usr/bin/env bash
set -euo pipefail

oneccl_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
oneccl_source_dir=$(cd -- "$oneccl_script_dir/../.." && pwd)
source "$oneccl_script_dir/leonardo_env.sh" \
    "${COMM_PLAYGROUND_ROOT:-$HOME/Projects/hpc-comm-playground}"

# oneCCL is built with DPC++ so callers can hand it SYCL queues and device
# communicators; OSHMPI must come from the same environment so both resolve one
# libmpi. Device support comes from SYCL itself; ONECCL_OSHMPI_PINNED_STAGING only
# turns on the optional CUDA pinning of the staging arena, which is worth ~45% of
# peak bandwidth on this machine but is not required.
oneccl_c_compiler=${ONECCL_C_COMPILER:-${DPCPP_CLANG:?leonardo_env.sh must define DPCPP_CLANG}}
oneccl_cxx_compiler=${ONECCL_CXX_COMPILER:-${DPCPP_CLANGXX:?leonardo_env.sh must define DPCPP_CLANGXX}}
oneccl_sycl_flags=${ONECCL_SYCL_FLAGS:-${SYCL_FLAGS:?leonardo_env.sh must define SYCL_FLAGS}}
oneccl_build_dir=${ONECCL_BUILD_DIR:-$SCRATCH/oneccl-oshmpi}
oneccl_install_prefix=${ONECCL_INSTALL_PREFIX:-$HOME/opt/oneccl-oshmpi}
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
    "-DMPI_C_COMPILER=$MPI_ROOT/bin/mpicc"
    "-DOSHMPI_ROOT=$OSHMPI_ROOT"
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
    -DBUILD_EXAMPLES=OFF \
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
