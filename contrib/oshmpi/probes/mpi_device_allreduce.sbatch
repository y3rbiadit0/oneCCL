#!/bin/bash -l
#SBATCH -A IscrC_HIGRAPH_0
#SBATCH -p boost_usr_prod
#SBATCH --job-name=mpi_dev_allreduce
#SBATCH --output=./%x-%j-%a-stdout.txt
#SBATCH --error=./%x-%j-%a-stderr.txt
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8

# Runs ONE (collective component, blocking mode) combination, because the known
# failure is a segfault that takes down the whole allocation. Submit the matrix
# with mpi_device_allreduce_matrix.sh so one crash cannot hide the rest.
#
# usage: sbatch [--export=ALL,PROBE_LABEL=...] mpi_device_allreduce.sbatch <blocking|nonblocking>

set -uo pipefail

mode=${1:?usage: sbatch mpi_device_allreduce.sbatch <blocking|nonblocking>}
label=${PROBE_LABEL:-unlabelled}

oneccl_source_dir=${ONECCL_SOURCE_DIR:?set ONECCL_SOURCE_DIR}
playground_root=${COMM_PLAYGROUND_ROOT:-$HOME/Projects/hpc-comm-playground}
source "$playground_root/cluster/leonardo/environment.sh" sycl

cuda_root=${CUDA_ROOT:-${CUDA_HOME:-}}
: "${cuda_root:?CUDA_ROOT is not set by the Leonardo environment}"

probe_src="$oneccl_source_dir/contrib/oshmpi/probes/mpi_device_allreduce.cpp"
probe_bin=${SCRATCH:-/tmp}/mpi_device_allreduce

mpicxx -std=c++11 -I"$cuda_root/include" \
    "$probe_src" -o "$probe_bin" \
    -L"$cuda_root/lib64" -lcudart || {
    printf 'error: probe failed to compile\n' >&2
    exit 2
}

printf '=== label=%s mode=%s ===\n' "$label" "$mode"
printf 'PROBE_THREAD_LEVEL=%s\n' "${PROBE_THREAD_LEVEL:-single}"
printf 'OMPI_MCA_coll_hcoll_enable=%s\n' "${OMPI_MCA_coll_hcoll_enable:-unset}"
printf 'OMPI_MCA_coll_ucc_enable=%s\n' "${OMPI_MCA_coll_ucc_enable:-unset}"
printf 'OMPI_MCA_coll_ucc_priority=%s\n\n' "${OMPI_MCA_coll_ucc_priority:-unset}"

srun --cpu-freq=high "$probe_bin" "$mode"
rc=$?
printf '\nRESULT %s %s exit=%d\n' "$label" "$mode" "$rc"
