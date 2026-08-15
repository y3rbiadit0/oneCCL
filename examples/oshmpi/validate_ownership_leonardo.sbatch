#!/bin/bash -l
#SBATCH -A IscrC_HIGRAPH_0
#SBATCH -p boost_usr_prod
#SBATCH --job-name=oshmpi_ownership
#SBATCH --output=./%x-%j-stdout.txt
#SBATCH --error=./%x-%j-stderr.txt
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8

set -euo pipefail

playground_root=${COMM_PLAYGROUND_ROOT:-$HOME/Projects/hpc-comm-playground}
install_prefix=${OSHMPI_INSTALL_PREFIX:-$HOME/opt/oshmpi-ee5cf110-oneccl}
smoke_test="$install_prefix/tests/oshmpi_mpi_ownership_smoke"

source "$playground_root/cluster/leonardo/environment.sh" cuda
export OSHMPI_HOME="$install_prefix"
export OSHMPI_ROOT="$install_prefix"
export PATH="$install_prefix/bin:$PATH"
export LD_LIBRARY_PATH="$install_prefix/lib:$install_prefix/lib64:${LD_LIBRARY_PATH:-}"
export COMM_PLAYGROUND_JOB_NODES=${SLURM_NNODES:-1}
source "$playground_root/cluster/leonardo/runtime/oshmpi.sh"

[[ -x "$smoke_test" ]] || {
    printf 'error: ownership smoke test not found: %s\n' "$smoke_test" >&2
    exit 2
}

printf 'OSHMPI_HOME=%s\n' "$OSHMPI_HOME"
printf 'smoke_test=%s\n' "$smoke_test"
ldd "$smoke_test"

srun --cpu-freq=high "$smoke_test" external
sleep 1
srun --cpu-freq=high "$smoke_test" owned
