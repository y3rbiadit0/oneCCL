#!/bin/bash -l
#SBATCH -A IscrC_HIGRAPH_0
#SBATCH -p boost_usr_prod
#SBATCH --job-name=oshmpi_cuda_probe
#SBATCH --output=./%x-%j-stdout.txt
#SBATCH --error=./%x-%j-stderr.txt
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8

# Phase 2 gate. Determines whether oneCCL can hand OSHMPI device buffers
# directly, or must stage CUDA collectives through host memory.
#
# Each combination runs as its own job step: OSHMPI aborts the job on error and
# some cases fault outright, so a shared process would mask every later result.

set -uo pipefail

oneccl_source_dir=${ONECCL_SOURCE_DIR:?set ONECCL_SOURCE_DIR}
playground_root=${COMM_PLAYGROUND_ROOT:-$HOME/Projects/hpc-comm-playground}

source "$oneccl_source_dir/examples/oshmpi/leonardo_env.sh" "$playground_root"
export COMM_PLAYGROUND_JOB_NODES=${SLURM_NNODES:-1}
source "$playground_root/cluster/leonardo/runtime/oshmpi.sh"

# The playground runtime script above disables UCC and HCOLL as part of the
# validated Leonardo stack. UCC is GPU-capable and sits at priority 100; with it
# disabled, libnbc (priority 10) services MPI_Iallreduce and reduces on the host
# CPU, which faults on device pointers. Re-enable it here, after the sourcing, to
# test whether that is the whole story -- exporting before sbatch would simply be
# overwritten.
if [[ ${PROBE_FORCE_UCC:-0} == 1 ]]; then
    export OMPI_MCA_coll_ucc_enable=1
    export OMPI_MCA_coll_ucc_priority=100
    printf 'forcing UCC on: enable=%s priority=%s\n\n' \
        "$OMPI_MCA_coll_ucc_enable" "$OMPI_MCA_coll_ucc_priority"
fi

: "${OSHMPI_HOME:?set OSHMPI_HOME to the patched OSHMPI install}"
export LD_LIBRARY_PATH="$OSHMPI_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cuda_root=${CUDA_ROOT:-${CUDA_HOME:-}}
: "${cuda_root:?CUDA_ROOT is not set by the Leonardo environment}"

probe_src="$oneccl_source_dir/examples/oshmpi/oshmpi_cuda_collectives_probe.cpp"
probe_bin=${SCRATCH:-/tmp}/oshmpi_cuda_collectives_probe

"$OSHMPI_HOME/bin/oshc++" -std=c++11 \
    -I"$cuda_root/include" \
    "$probe_src" -o "$probe_bin" \
    -L"$cuda_root/lib64" -lcudart || {
    printf 'error: probe failed to compile\n' >&2
    exit 2
}
printf 'probe built: %s\n\n' "$probe_bin"

declare -a results=()

run_probe() {
    local memory=$1 collective=$2
    printf -- '---- %s %s ----\n' "$memory" "$collective"
    srun --cpu-freq=high "$probe_bin" "$memory" "$collective"
    local rc=$?
    local line
    if [[ $rc -eq 0 ]]; then
        line="PASS  $memory-$collective"
    else
        line="FAIL  $memory-$collective (exit $rc)"
    fi
    results+=("$line")
    # Emit immediately as well: a segfault inside a job step can take the whole
    # allocation down, and a summary printed only at the end would be lost.
    printf 'RESULT %s\n\n' "$line"
}

# PROBE_ONLY runs an explicit list of "<memory>:<collective>" pairs instead of the
# default sweep, so a single case can be re-checked without an ad-hoc srun that
# would miss this script's environment setup.
#   PROBE_ONLY="raw:reduce" sbatch examples/oshmpi/probe_cuda_collectives.sbatch
if [[ -n ${PROBE_ONLY:-} ]]; then
    for pair in $PROBE_ONLY; do
        run_probe "${pair%%:*}" "${pair##*:}"
    done
    printf '==================== summary ====================\n'
    for line in "${results[@]}"; do
        printf '%s\n' "$line"
    done
    exit 0
fi

# Memory-independent sanity check that the runtime comes up at all.
run_probe space barrier

# Only the pure data-movement collectives are swept by default. The other two
# are known to segfault on device memory, for unrelated reasons, and each fault
# takes the whole job down along with every later result:
#
#   broadcast - OSHMPI_broadcast_team() ends with a host memcpy(dest, source) on
#               the root PE (src/shmem/coll.c:66). An OSHMPI defect.
#   reduce    - OSHMPI issues a non-blocking allreduce and Open MPI's libnbc
#               fallback reduces on the host CPU, in
#               ompi_op_avx_2buff_add_float_avx2(). An MPI limitation rather than
#               an OSHMPI one: this stack builds with HCOLL and UCC disabled, so
#               no accelerated component is available to handle device operands.
#
# Set PROBE_INCLUDE_FAULTING=1 to re-confirm either on a changed stack.
#
# `space`: CUDA symmetric memory via shmemx_space_create(SHMEMX_MEM_CUDA).
# `raw`:   plain cudaMalloc, deliberately not symmetric. If this passes, the
#          collectives are riding purely on CUDA-aware MPI.
for memory in space raw; do
    for collective in fcollect alltoall; do
        run_probe "$memory" "$collective"
    done
done

if [[ ${PROBE_INCLUDE_FAULTING:-0} == 1 ]]; then
    printf 'running the known-faulting collectives last; each ends the job\n\n'
    for memory in space raw; do
        for collective in broadcast reduce; do
            run_probe "$memory" "$collective"
        done
    done
fi

printf '==================== summary ====================\n'
for line in "${results[@]}"; do
    printf '%s\n' "$line"
done
