#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

usage() {
    cat <<'EOF'
Build and install oneCCL with DPC++/CUDA, NCCL, and bundled Intel MPI on Leonardo.

Usage:
  build-leonardo-sycl.sh [options] [-- <extra CMake arguments>]

Options:
  --source-dir DIR       oneCCL source tree (default: directory containing this script)
  --build-dir DIR        CMake build directory (default: $SCRATCH/oneccl-nccl-build,
                         or <source-dir>/oneccl-nccl-build when $SCRATCH is unset)
  --install-prefix DIR   Installation prefix (default: $ONECCL_NCCL_ROOT or
                         $HOME/opt/oneccl-nccl-leonardo)
  --env-script FILE      Leonardo environment loader (default:
                         $COMM_PLAYGROUND_ROOT/cluster/leonardo/environment.sh)
  --skip-env             Use the already-loaded compiler/module environment
  --mpi-root DIR         Bundled Intel MPI root (default: <source-dir>/deps/mpi)
  --nccl-root DIR        NCCL installation root (default: $NCCL_ROOT or $NCCL_HOME)
  -j, --jobs N           Parallel build jobs (default: $SLURM_CPUS_PER_TASK or 16)
  --clean                Remove the build directory before configuring (default)
  --no-clean             Reuse the existing build directory
  --examples             Build optional oneCCL examples
  --no-examples          Do not build oneCCL examples (default)
  --install              Install after building (default)
  --no-install           Build without installing
  --configure-only       Configure without building or installing
  -h, --help             Show this help

Environment overrides:
  ONECCL_SOURCE_DIR, ONECCL_BUILD_DIR, ONECCL_INSTALL_PREFIX,
  ONECCL_ENV_SCRIPT, ONECCL_MPI_ROOT, ONECCL_NCCL_ROOT, ONECCL_JOBS,
  ONECCL_CLEAN, ONECCL_BUILD_EXAMPLES, ONECCL_INSTALL,
  ONECCL_HOST_FLAGS, ONECCL_SYCL_FLAGS, ONECCL_LINK_FLAGS
  Boolean environment overrides use 1 (enabled) or 0 (disabled).

Examples:
  ./build-leonardo-sycl.sh
  ./build-leonardo-sycl.sh --source-dir $HOME/other-oneccl
  ./build-leonardo-sycl.sh --no-clean -j 32 -- -DENABLE_ITT=OFF
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

require_value() {
    [[ $# -ge 2 ]] || die "option $1 requires a value"
}

source_dir=${ONECCL_SOURCE_DIR:-}
build_dir=${ONECCL_BUILD_DIR:-}
install_prefix=${ONECCL_INSTALL_PREFIX:-}
env_script=${ONECCL_ENV_SCRIPT:-}
mpi_root=${ONECCL_MPI_ROOT:-}
nccl_root=
jobs=${ONECCL_JOBS:-}
clean=${ONECCL_CLEAN:-1}
build_examples=${ONECCL_BUILD_EXAMPLES:-0}
run_install=${ONECCL_INSTALL:-1}
load_environment=1
configure_only=0
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
        --install-prefix)
            require_value "$@"
            install_prefix=$2
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
        --mpi-root)
            require_value "$@"
            mpi_root=$2
            shift 2
            ;;
        --nccl-root)
            require_value "$@"
            nccl_root=$2
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
        --examples)
            build_examples=1
            shift
            ;;
        --no-examples)
            build_examples=0
            shift
            ;;
        --install)
            run_install=1
            shift
            ;;
        --no-install)
            run_install=0
            shift
            ;;
        --configure-only)
            configure_only=1
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

for setting in "clean:$clean" "build examples:$build_examples" "install:$run_install"; do
    value=${setting##*:}
    [[ "$value" == 0 || "$value" == 1 ]] || die "${setting%%:*} must be 0 or 1"
done

if [[ "$load_environment" == 1 ]]; then
    env_script=${env_script:-${COMM_PLAYGROUND_ROOT:-$HOME/comm-playground}/cluster/leonardo/environment.sh}
    [[ -f "$env_script" ]] || die "Leonardo environment script not found: $env_script (use --env-script or --skip-env)"
    # environment.sh accepts the stack name; direct stack scripts safely ignore it.
    source "$env_script" sycl
fi

# oneCCL uses its bundled Intel MPI. Remove OpenMPI paths loaded by the shared
# Leonardo SYCL environment before selecting the bundled MPI below.
if type module >/dev/null 2>&1; then
    module unload openmpi >/dev/null 2>&1 || true
fi
unset MPI_HOME MPI_ROOT OPENMPI_HOME OPAL_PREFIX

source_dir=${source_dir:-$script_dir}
build_dir=${build_dir:-${SCRATCH:-$source_dir}/oneccl-nccl-build}
install_prefix=${install_prefix:-${ONECCL_NCCL_ROOT:-$HOME/opt/oneccl-nccl-leonardo}}
mpi_root=${mpi_root:-$source_dir/deps/mpi}
nccl_root=${nccl_root:-${NCCL_ROOT:-${NCCL_HOME:-}}}
jobs=${jobs:-${SLURM_CPUS_PER_TASK:-16}}

[[ -f "$source_dir/CMakeLists.txt" ]] || die "oneCCL source tree not found: $source_dir"
[[ "$build_dir" != "$source_dir" ]] || die "build directory must differ from the source directory"
[[ -f "$mpi_root/include/mpi.h" ]] || die "bundled Intel MPI headers not found under $mpi_root"
[[ -e "$mpi_root/lib/libmpi.so.12" ]] || die "bundled Intel MPI library not found under $mpi_root"
[[ -n "$nccl_root" ]] || die "NCCL root is unknown (use --nccl-root or set NCCL_ROOT/NCCL_HOME)"
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || die "jobs must be a positive integer, got: $jobs"

command -v cmake >/dev/null 2>&1 || die "cmake is not available"
command -v ninja >/dev/null 2>&1 || die "ninja is not available"

dpcpp_install=${DPCPP_INSTALL:-${DPCPP_ROOT:-}}
[[ -n "$dpcpp_install" ]] || die "DPCPP_INSTALL or DPCPP_ROOT is not defined"
dpcpp_clang=${DPCPP_CLANG:-$dpcpp_install/bin/clang}
dpcpp_clangxx=${DPCPP_CLANGXX:-$dpcpp_install/bin/clang++}
[[ -x "$dpcpp_clang" ]] || die "DPC++ C compiler not found: $dpcpp_clang"
[[ -x "$dpcpp_clangxx" ]] || die "DPC++ C++ compiler not found: $dpcpp_clangxx"

gcc_root=${GCC12_ROOT:-${GCC_HOME:-}}
[[ -n "$gcc_root" ]] || die "GCC12_ROOT or GCC_HOME is not defined"
gcc_lib=${GCC12_LIB:-$gcc_root/lib64}
cuda_root=${CUDA_HOME:-${CUDA_ROOT:-${CUDA_PATH:-}}}
[[ -n "$cuda_root" ]] || die "CUDA_HOME, CUDA_ROOT, or CUDA_PATH is not defined"

sycl_target=${SYCL_TARGET:-nvptx64-nvidia-cuda}
gpu_arch=${NVIDIA_GPU_ARCH:-sm_80}
host_flags=${ONECCL_HOST_FLAGS:---gcc-toolchain=$gcc_root}
sycl_flags=${ONECCL_SYCL_FLAGS:-${SYCL_FLAGS:--fsycl $host_flags -fsycl-targets=$sycl_target -Xsycl-target-backend=$sycl_target --cuda-gpu-arch=$gpu_arch}}
link_flags=${ONECCL_LINK_FLAGS:-$host_flags -L$gcc_lib -Wl,-rpath,$gcc_lib -L$dpcpp_install/lib -Wl,-rpath,$dpcpp_install/lib -L$mpi_root/lib -Wl,-rpath,$mpi_root/lib}

export DPCPP_ROOT="$dpcpp_install"
export I_MPI_ROOT="$mpi_root"
export NCCL_ROOT="$nccl_root"
export CCL_ROOT="$install_prefix"
export ONECCL_NCCL_ROOT="$install_prefix"
export PATH="$mpi_root/bin:$PATH"
export LD_LIBRARY_PATH="$mpi_root/lib:${LD_LIBRARY_PATH:-}"

if [[ "$clean" == 1 ]]; then
    [[ -n "$build_dir" && "$build_dir" != / ]] || die "refusing to clean unsafe build directory: $build_dir"
    cmake -E remove_directory "$build_dir"
fi

if [[ "$build_examples" == 1 ]]; then
    cmake_build_examples=ON
    cmake_mpi_tests=ON
else
    cmake_build_examples=OFF
    cmake_mpi_tests=OFF
fi

printf '%s\n' \
    "oneCCL source:  $source_dir" \
    "Build directory: $build_dir" \
    "Install prefix:  $install_prefix" \
    "Bundled MPI:     $mpi_root" \
    "NCCL root:       $nccl_root" \
    "DPC++ root:      $dpcpp_install" \
    "Parallel jobs:   $jobs"

cmake_args=(
    -S "$source_dir"
    -B "$build_dir"
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    "-DCMAKE_INSTALL_PREFIX=$install_prefix"
    "-DCMAKE_C_COMPILER=$dpcpp_clang"
    "-DCMAKE_CXX_COMPILER=$dpcpp_clangxx"
    "-DCMAKE_C_FLAGS=$host_flags"
    "-DCMAKE_CXX_FLAGS=$sycl_flags"
    "-DCMAKE_CXX_FLAGS_RELEASE=$sycl_flags -O3 -DNDEBUG"
    "-DCMAKE_EXE_LINKER_FLAGS=$link_flags"
    "-DCMAKE_SHARED_LINKER_FLAGS=$link_flags"
    "-DCMAKE_MODULE_LINKER_FLAGS=$link_flags"
    "-DCMAKE_REQUIRED_FLAGS=$host_flags"
    "-DCUDAToolkit_ROOT=$cuda_root"
    "-DCUDA_ROOT=$cuda_root"
    "-DNCCL_ROOT=$nccl_root"
    "-DDPCPP_ROOT=$dpcpp_install"
    "-DMPI_DIR=$mpi_root"
    "-DI_MPI_ROOT=$mpi_root"
    -DCOMPUTE_BACKEND=dpcpp
    -DCCL_ENABLE_SYCL=ON
    -DCCL_ENABLE_NCCL=ON
    -DCCL_ENABLE_RCCL=OFF
    -DCCL_ENABLE_ZE=OFF
    -DENABLE_MPI=ON
    "-DENABLE_MPI_TESTS=$cmake_mpi_tests"
    -DENABLE_OFI_HMEM=OFF
    -DENABLE_OMP=OFF
    -DBUILD_FT=OFF
    "-DBUILD_EXAMPLES=$cmake_build_examples"
)

cmake "${cmake_args[@]}" "${extra_cmake_args[@]}"

if [[ "$configure_only" == 1 ]]; then
    exit 0
fi

cmake --build "$build_dir" --parallel "$jobs"

if [[ "$run_install" == 1 ]]; then
    cmake --install "$build_dir"
fi
