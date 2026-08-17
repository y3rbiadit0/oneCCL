#!/usr/bin/env bash
set -euo pipefail

# Submits the device-allreduce matrix as independent jobs.
#
# Each combination gets its own job on purpose: the failing case segfaults inside
# the MPI collective, which aborts the job step and takes the rest of the
# allocation with it. Separate jobs mean one crash cannot hide the others.
#
# Reading the results:
#   any PASS                  -> device-side allreduce is achievable on this stack
#   blocking PASS, nonblocking FAIL
#                             -> rebuild OSHMPI with --enable-async-thread=yes,
#                                which switches it to blocking MPI_Allreduce.
#                                Much cheaper than host staging.
#   all FAIL                  -> device allreduce is not available here; oneCCL
#                                must stage allreduce through host memory.
#
# usage: ONECCL_SOURCE_DIR=... ./mpi_device_allreduce_matrix.sh

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${ONECCL_SOURCE_DIR:?set ONECCL_SOURCE_DIR}"

submit() {
    local label=$1 mode=$2 extra=$3
    local exports="ALL,PROBE_LABEL=$label"
    [[ -n "$extra" ]] && exports="$exports,$extra"
    printf '%-18s %-12s ' "$label" "$mode"
    sbatch --parsable --export="$exports" \
        "$script_dir/mpi_device_allreduce.sbatch" "$mode"
}

# baseline: whatever the Leonardo stack selects by default (HCOLL and UCC are
# disabled there, so this is expected to fall back to tuned/libnbc)
submit default blocking ""
submit default nonblocking ""

# HCOLL implements blocking collectives only, so the nonblocking row is expected
# to be unaffected by it. Submitted anyway to confirm rather than assume.
submit hcoll blocking "OMPI_MCA_coll_hcoll_enable=1"
submit hcoll nonblocking "OMPI_MCA_coll_hcoll_enable=1"

# UCC does implement non-blocking collectives, so this is the row most likely to
# rescue OSHMPI's current MPI_Iallreduce path without an OSHMPI rebuild.
submit ucc blocking "OMPI_MCA_coll_ucc_enable=1,OMPI_MCA_coll_ucc_priority=100"
submit ucc nonblocking "OMPI_MCA_coll_ucc_enable=1,OMPI_MCA_coll_ucc_priority=100"

printf '\nwhen the jobs finish:\n'
printf '  grep -h RESULT mpi_dev_allreduce-*-stdout.txt | sort\n'
