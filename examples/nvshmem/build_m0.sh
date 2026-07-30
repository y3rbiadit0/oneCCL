#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

usage() {
    cat <<'EOF'
Configure and build the standalone oneCCL NVSHMEM M0 interoperability spike.

Usage:
  build_m0.sh [options] [-- <extra CMake arguments>]

Options:
  --build-dir DIR       Build directory (default: $SCRATCH/oneccl-nvshmem-m0-build
                        or <source-dir>/oneccl-nvshmem-m0-build)
  --dpcpp-root DIR      DPC++ installation (default: $DPCPP_ROOT or $DPCPP_INSTALL)
  --cuda-root DIR       CUDA installation (default: $CUDA_HOME, $CUDA_ROOT, or $CUDA_PATH)
  --nvshmem-root DIR    NVSHMEM installation (default: $NVSHMEM_HOME or $NVSHMEM_ROOT)
  --mpi-cxx FILE        MPI C++ wrapper used by CMake FindMPI (default:
                        $MPI_CXX_COMPILER, $MPICXX, or mpicxx)
  --cuda-arch ARCH      CUDA architecture without sm_ prefix (default: 80)
  --cuda-host FILE      NVCC host C++ compiler (default: $NVSHMEM_M0_CUDA_HOST_COMPILER,
                        or $GCC12_ROOT/bin/g++ when available)
  -j, --jobs N          Parallel build jobs (default: $SLURM_CPUS_PER_TASK or 16)
  --clean               Remove the build directory before configuring (default)
  --no-clean            Reuse the existing build directory
  --configure-only      Configure without building
  --dry-run             Print commands without checking paths or executing them
  -h, --help            Show this help

Environment overrides:
  DPCPP_ROOT, DPCPP_INSTALL, CUDA_HOME, CUDA_ROOT, CUDA_PATH,
  NVSHMEM_HOME, NVSHMEM_ROOT, NVSHMEM_DIR, MPI_CXX_COMPILER, MPICXX,
  GCC12_ROOT, GCC_HOME,
  NVSHMEM_M0_BUILD_DIR, NVSHMEM_M0_CUDA_ARCHITECTURE,
  NVSHMEM_M0_CUDA_HOST_COMPILER, NVSHMEM_M0_HOST_FLAGS, NVSHMEM_M0_JOBS

Example:
  ./examples/nvshmem/build_m0.sh --cuda-arch 80
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

require_value() {
    [[ $# -ge 2 ]] || die "option $1 requires a value"
}

print_command() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
}

build_dir=${NVSHMEM_M0_BUILD_DIR:-}
dpcpp_root=${DPCPP_ROOT:-${DPCPP_INSTALL:-}}
cuda_root=${CUDA_HOME:-${CUDA_ROOT:-${CUDA_PATH:-}}}
nvshmem_root=${NVSHMEM_HOME:-${NVSHMEM_ROOT:-}}
nvshmem_dir=${NVSHMEM_DIR:-}
mpi_cxx=${MPI_CXX_COMPILER:-${MPICXX:-mpicxx}}
gcc_root=${GCC12_ROOT:-${GCC_HOME:-}}
cuda_arch=${NVSHMEM_M0_CUDA_ARCHITECTURE:-80}
cuda_host=${NVSHMEM_M0_CUDA_HOST_COMPILER:-}
jobs=${NVSHMEM_M0_JOBS:-${SLURM_CPUS_PER_TASK:-16}}
clean=1
configure_only=0
dry_run=0
dpcpp_root_from_cli=0
cuda_root_from_cli=0
nvshmem_root_from_cli=0
extra_cmake_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir)
            require_value "$@"
            build_dir=$2
            shift 2
            ;;
        --dpcpp-root)
            require_value "$@"
            dpcpp_root=$2
            dpcpp_root_from_cli=1
            shift 2
            ;;
        --cuda-root)
            require_value "$@"
            cuda_root=$2
            cuda_root_from_cli=1
            shift 2
            ;;
        --nvshmem-root)
            require_value "$@"
            nvshmem_root=$2
            nvshmem_dir=
            nvshmem_root_from_cli=1
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
        --cuda-host)
            require_value "$@"
            cuda_host=$2
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

build_dir=${build_dir:-${SCRATCH:-$script_dir}/oneccl-nvshmem-m0-build}
if [[ "$dpcpp_root_from_cli" == 1 ]]; then
    dpcpp_cxx=$dpcpp_root/bin/clang++
else
    dpcpp_cxx=${DPCPP_CXX:-${dpcpp_root:+$dpcpp_root/bin/clang++}}
fi
if [[ "$cuda_root_from_cli" == 1 ]]; then
    nvcc=$cuda_root/bin/nvcc
else
    nvcc=${NVCC:-${cuda_root:+$cuda_root/bin/nvcc}}
fi

if [[ "$nvshmem_root_from_cli" == 1 || ( -z "$nvshmem_dir" && -n "$nvshmem_root" ) ]]; then
    if [[ -f "$nvshmem_root/lib/cmake/nvshmem/NVSHMEMConfig.cmake" ]]; then
        nvshmem_dir=$nvshmem_root/lib/cmake/nvshmem
    elif [[ -f "$nvshmem_root/lib64/cmake/nvshmem/NVSHMEMConfig.cmake" ]]; then
        nvshmem_dir=$nvshmem_root/lib64/cmake/nvshmem
    else
        nvshmem_dir=$nvshmem_root/lib/cmake/nvshmem
    fi
fi

if [[ -z "$cuda_host" && -n "$gcc_root" ]]; then
    cuda_host=$gcc_root/bin/g++
fi

if [[ "$dry_run" == 0 ]]; then
    command -v cmake >/dev/null 2>&1 || die "cmake is not available"
    command -v ninja >/dev/null 2>&1 || die "ninja is not available"
    [[ -n "$dpcpp_cxx" && -x "$dpcpp_cxx" ]] ||
        die "DPC++ clang++ was not found; set DPCPP_ROOT, DPCPP_INSTALL, or DPCPP_CXX"
    [[ -n "$nvcc" && -x "$nvcc" ]] ||
        die "NVCC was not found; set CUDA_HOME, CUDA_ROOT, CUDA_PATH, or NVCC"
    if [[ "$mpi_cxx" == */* ]]; then
        [[ -x "$mpi_cxx" ]] || die "MPI C++ wrapper is not executable: $mpi_cxx"
    else
        mpi_cxx=$(command -v "$mpi_cxx") ||
            die "MPI C++ wrapper was not found; set MPI_CXX_COMPILER or MPICXX"
    fi
    [[ -n "$cuda_root" && -f "$cuda_root/include/cuda_runtime_api.h" ]] ||
        die "CUDA headers were not found; set CUDA_HOME or CUDA_ROOT"
    if [[ -n "$nvshmem_dir" && -f "$nvshmem_dir/NVSHMEMConfig.cmake" ]]; then
        nvshmem_discovery="CMake package: $nvshmem_dir"
    elif [[ -n "$nvshmem_root" &&
            -f "$nvshmem_root/include/nvshmem.h" &&
            -f "$nvshmem_root/include/nvshmemx.h" &&
            -e "$nvshmem_root/lib/libnvshmem_host.so" &&
            -e "$nvshmem_root/lib/libnvshmem_device.a" ]]; then
        nvshmem_discovery="direct libraries: $nvshmem_root"
    elif [[ -n "$nvshmem_root" &&
            -f "$nvshmem_root/include/nvshmem.h" &&
            -f "$nvshmem_root/include/nvshmemx.h" &&
            -e "$nvshmem_root/lib64/libnvshmem_host.so" &&
            -e "$nvshmem_root/lib64/libnvshmem_device.a" ]]; then
        nvshmem_discovery="direct libraries: $nvshmem_root"
    else
        die "NVSHMEM headers and libraries were not found; set NVSHMEM_HOME, NVSHMEM_ROOT, or NVSHMEM_DIR"
    fi
    if [[ -n "$cuda_host" ]]; then
        [[ -x "$cuda_host" ]] || die "NVCC host compiler is not executable: $cuda_host"
    fi

    printf '%s\n' \
        "M0 source:        $script_dir" \
        "Build directory: $build_dir" \
        "DPC++ compiler:  $dpcpp_cxx" \
        "NVCC:            $nvcc" \
        "NVCC host:       ${cuda_host:-<NVCC default>}" \
        "MPI C++ wrapper: $mpi_cxx" \
        "CUDA root:       $cuda_root" \
        "NVSHMEM:         $nvshmem_discovery" \
        "CUDA arch:       $cuda_arch"
    "$dpcpp_cxx" --version
    "$nvcc" --version
fi

host_flags=${NVSHMEM_M0_HOST_FLAGS:-}
if [[ -z "$host_flags" && -n "$gcc_root" ]]; then
    host_flags=--gcc-toolchain=$gcc_root
fi

link_flags=
if [[ -n "$gcc_root" ]]; then
    gcc_lib=$gcc_root/lib64
    link_flags="-L$gcc_lib -Wl,-rpath,$gcc_lib"
fi
if [[ -n "$dpcpp_root" ]]; then
    link_flags="${link_flags:+$link_flags }-L$dpcpp_root/lib -Wl,-rpath,$dpcpp_root/lib"
fi

cmake_args=(
    -S "$script_dir"
    -B "$build_dir"
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    "-DCMAKE_CXX_COMPILER=$dpcpp_cxx"
    "-DCMAKE_CUDA_COMPILER=$nvcc"
    "-DCUDAToolkit_ROOT=$cuda_root"
    "-DMPI_CXX_COMPILER=$mpi_cxx"
    "-DNVSHMEM_ROOT=$nvshmem_root"
    "-DNVSHMEM_M0_CUDA_ARCHITECTURE=$cuda_arch"
)

if [[ -n "$nvshmem_dir" && -f "$nvshmem_dir/NVSHMEMConfig.cmake" ]]; then
    cmake_args+=("-DNVSHMEM_DIR=$nvshmem_dir")
fi

if [[ -n "$cuda_host" ]]; then
    cmake_args+=("-DCMAKE_CUDA_HOST_COMPILER=$cuda_host")
fi
if [[ -n "$host_flags" ]]; then
    cmake_args+=("-DCMAKE_CXX_FLAGS=$host_flags")
fi
if [[ -n "$link_flags" ]]; then
    cmake_args+=("-DCMAKE_EXE_LINKER_FLAGS=$link_flags")
fi
if (( ${#extra_cmake_args[@]} > 0 )); then
    cmake_args+=("${extra_cmake_args[@]}")
fi

if [[ "$clean" == 1 ]]; then
    if [[ "$dry_run" == 1 ]]; then
        print_command cmake -E remove_directory "$build_dir"
    else
        [[ -n "$build_dir" && "$build_dir" != / ]] || die "unsafe build directory: $build_dir"
        cmake -E make_directory "$build_dir"
        canonical_build_dir=$(cd -- "$build_dir" && pwd -P)
        canonical_script_dir=$(cd -- "$script_dir" && pwd -P)
        canonical_work_dir=$(pwd -P)
        [[ "$canonical_build_dir" != / ]] || die "refusing to clean the filesystem root"
        [[ "$canonical_script_dir" != "$canonical_build_dir" &&
           "$canonical_script_dir" != "$canonical_build_dir/"* ]] ||
            die "refusing to clean a source-tree ancestor: $canonical_build_dir"
        [[ "$canonical_work_dir" != "$canonical_build_dir" &&
           "$canonical_work_dir" != "$canonical_build_dir/"* ]] ||
            die "refusing to clean the current working directory or its ancestor: $canonical_build_dir"
        cmake -E remove_directory "$canonical_build_dir"
    fi
fi

print_command cmake "${cmake_args[@]}"
if [[ "$dry_run" == 0 ]]; then
    cmake "${cmake_args[@]}"
fi

if [[ "$configure_only" == 1 ]]; then
    exit 0
fi

print_command cmake --build "$build_dir" --parallel "$jobs"
if [[ "$dry_run" == 0 ]]; then
    cmake --build "$build_dir" --parallel "$jobs"
    printf 'M0 binary: %s\n' "$build_dir/oneccl_nvshmem_m0"
fi
