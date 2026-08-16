#!/bin/bash -l
#SBATCH -A IscrC_HIGRAPH_0
#SBATCH -p boost_usr_prod
#SBATCH --job-name=oneccl_oshmpi
#SBATCH --output=./%x-%j-stdout.txt
#SBATCH --error=./%x-%j-stderr.txt
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8

set -euo pipefail

oneccl_source_dir=${ONECCL_SOURCE_DIR:?set ONECCL_SOURCE_DIR}
oneccl_build_dir=${ONECCL_BUILD_DIR:-$SCRATCH/oneccl-oshmpi-gcc}
playground_root=${COMM_PLAYGROUND_ROOT:-$HOME/Projects/hpc-comm-playground}
test_binary="$oneccl_build_dir/tests/oshmpi/oshmpi_collectives_test"

mkdir -p ./logs
source "$oneccl_source_dir/examples/oshmpi/leonardo_env.sh" "$playground_root"
export COMM_PLAYGROUND_JOB_NODES=${SLURM_NNODES:-1}
source "$playground_root/cluster/leonardo/runtime/oshmpi.sh"

# oneCCL builds with CMAKE_SKIP_RPATH, so nothing records where libccl.so.1 or
# liboshmpi.so live. Put both on the loader path explicitly.
export LD_LIBRARY_PATH="$oneccl_build_dir/src${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if [[ -n "${OSHMPI_HOME:-}" ]]; then
    export LD_LIBRARY_PATH="$OSHMPI_HOME/lib:$LD_LIBRARY_PATH"
fi

export CCL_BACKEND=oshmpi
export CCL_LOG_LEVEL=${CCL_LOG_LEVEL:-info}

# leonardo_env.sh, sourced above, already exports a production-sized arena
# (CCL_OSHMPI_STAGING_SIZE=64M) and SHMEM_SYMMETRIC_SIZE=1G. A ${VAR:-...}
# default here would therefore never apply. Override the arena deliberately:
# the point of this job is to force every collective through the chunking path,
# which a 64M arena would skip entirely for the test's payload sizes.
# 128 bytes is the smallest value that still yields two aligned lanes.
export CCL_OSHMPI_STAGING_SIZE=${ONECCL_VALIDATE_STAGING_SIZE:-128}

[[ -x "$test_binary" ]] || {
    printf 'error: test executable not found: %s\n' "$test_binary" >&2
    exit 2
}

printf -- '---- host buffers ----\n'
srun --cpu-freq=high "$test_binary"
host_rc=$?
printf 'RESULT host exit=%d\n\n' "$host_rc"

# Present only when oneCCL was configured with CCL_ENABLE_OSHMPI_CUDA=ON.
device_binary="$oneccl_build_dir/tests/oshmpi/oshmpi_device_collectives_test"
device_rc=0
if [[ -x "$device_binary" ]]; then
    printf -- '---- device buffers ----\n'
    srun --cpu-freq=high "$device_binary"
    device_rc=$?
    printf 'RESULT device exit=%d\n\n' "$device_rc"
else
    printf 'device test not built; configure with CCL_ENABLE_OSHMPI_CUDA=ON to include it\n\n'
fi

# Present only when oneCCL was configured with CCL_ENABLE_SYCL=ON. Covers the
# device communicator, stream draining, group scope and send/recv, none of which
# the two tests above reach.
sycl_binary="$oneccl_build_dir/tests/oshmpi/oshmpi_sycl_collectives_test"
sycl_rc=0
if [[ -x "$sycl_binary" ]]; then
    printf -- '---- sycl device communicator, stream, group, pt2pt ----\n'
    srun --cpu-freq=high "$sycl_binary"
    sycl_rc=$?
    printf 'RESULT sycl exit=%d\n\n' "$sycl_rc"
else
    printf 'sycl test not built; configure with CCL_ENABLE_SYCL=ON to include it\n\n'
fi

exit $(( host_rc != 0 || device_rc != 0 || sycl_rc != 0 ))
