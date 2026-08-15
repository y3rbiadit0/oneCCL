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

export CCL_BACKEND=oshmpi
export CCL_LOG_LEVEL=${CCL_LOG_LEVEL:-info}
export CCL_OSHMPI_STAGING_SIZE=${CCL_OSHMPI_STAGING_SIZE:-128}
export SHMEM_SYMMETRIC_SIZE=${SHMEM_SYMMETRIC_SIZE:-128M}

[[ -x "$test_binary" ]] || {
    printf 'error: test executable not found: %s\n' "$test_binary" >&2
    exit 2
}

srun --cpu-freq=high "$test_binary"
