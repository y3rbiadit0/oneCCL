#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/leonardo_env.sh" "${COMM_PLAYGROUND_ROOT:-$HOME/Projects/hpc-comm-playground}"

source_dir=$(cd -- "$script_dir/../.." && pwd)
build_dir=${ONECCL_BUILD_DIR:-$SCRATCH/oneccl-oshmpi-$(basename "$OSHMPI_ROOT")}
install_prefix=${ONECCL_INSTALL_PREFIX:-$HOME/opt/oneccl-oshmpi}

cmake -S "$source_dir" -B "$build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$CC" \
    -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_INSTALL_PREFIX="$install_prefix" \
    -DMPI_DIR="$MPI_ROOT" \
    -DOSHMPI_ROOT="$OSHMPI_ROOT" \
    -DCCL_ENABLE_OSHMPI=ON \
    -DCCL_ENABLE_NCCL=OFF \
    -DCCL_ENABLE_RCCL=OFF \
    -DCCL_ENABLE_SYCL=OFF \
    -DCCL_ENABLE_ZE=OFF \
    -DENABLE_MPI=ON \
    -DENABLE_MPI_TESTS=ON \
    -DENABLE_OMP=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_FT=ON

cmake --build "$build_dir" --parallel "${ONECCL_BUILD_JOBS:-8}"
cmake --install "$build_dir"

printf 'oneCCL OSHMPI build: %s\n' "$build_dir"
printf 'oneCCL OSHMPI install: %s\n' "$install_prefix"
