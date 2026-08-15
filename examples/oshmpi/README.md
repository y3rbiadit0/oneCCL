# OSHMPI Backend

Phase 1 provides blocking oneCCL host collectives through OSHMPI. It supports
barrier, allgather, allreduce, alltoall, and broadcast over
`SHMEM_TEAM_WORLD`. User buffers are copied through bounded symmetric staging.

The backend requires an OSHMPI build containing
`0001-preserve-external-mpi-ownership.patch`. This prevents
`shmem_finalize()` from finalizing MPI when the application initialized MPI
and preserves the MPI thread level while OSHMPI initializes MPI_T.

## Leonardo

The validated dependency pins are:

```text
OSHMPI ee5cf110e673c098707257bb025404e17ac0a5fc
OpenPA 0475704dde41054db33562a8d17314fe0e30aaf3
```

The provenance probe can be rerun with:

```bash
export COMM_PLAYGROUND_ROOT=$HOME/Projects/hpc-comm-playground
./examples/oshmpi/probe_leonardo_oshmpi.sh "$HOME/opt-src/oshmpi-main"
```

Build the pinned revision with the ownership patch:

```bash
./examples/oshmpi/build_oshmpi_leonardo.sh
```

This creates a clean detached worktree under `$SCRATCH`, leaves the existing
source and install untouched, and installs to
`$HOME/opt/oshmpi-ee5cf110-oneccl`.

Validate both MPI ownership paths:

```bash
sbatch examples/oshmpi/validate_ownership_leonardo.sbatch
```

Select the resulting install and build oneCCL:

```bash
export OSHMPI_HOME=$HOME/opt/oshmpi-ee5cf110-oneccl
export COMM_PLAYGROUND_ROOT=$HOME/Projects/hpc-comm-playground
./examples/oshmpi/build_leonardo.sh
```

The default oneCCL build directory is `$SCRATCH/oneccl-oshmpi-gcc`. Phase 1
builds oneCCL with `gcc`/`g++` while OSHMPI remains linked to the validated
HPC-X/NVHPC CUDA stack. Override the host compilers with
`ONECCL_C_COMPILER` and `ONECCL_CXX_COMPILER`. Set `ONECCL_BUILD_DIR` to use
another directory; it must not contain a CMake cache for another source tree or
compiler.

This is an OSHMPI-focused oneCCL artifact: the native oneCCL MPI and stub
backends are disabled. HPC-X MPI remains an explicit dependency of OSHMPI and
the lifecycle/correctness tests.

### Why the build is configured this way

`ENABLE_MPI=OFF` is required, not a shortcut. oneCCL's native MPI transport
targets the MPICH ABI specifically: `mpi_api_wrapper.cpp` dlopens `libmpi.so.12`,
`atl_mpi_ctx.cpp` drives `I_MPI_*` environment variables, and
`atl_base_comm::get_mpi_comm()` returns an `int`, which cannot hold an Open MPI
`MPI_Comm`. It can never be built against HPC-X. OSHMPI brings its own MPI
transport, so nothing is lost by disabling it.

OSHMPI's MPI is discovered independently, through CMake's `FindMPI`
(`MPI::MPI_C`), and is unrelated to `MPI_DIR`/`MPI_INCLUDE_DIR` — those describe
the vendored Intel MPI that the native transport would use. Point the build at a
specific MPI with `-DMPI_C_COMPILER`; do not set `MPI_DIR`.

`ENABLE_PMIX=ON` is kept because `atl_ofi_helper.cpp` references `pmix_value_t`
and `ccl_pmix::*` unconditionally while `pmix_api_wrapper.hpp` guards them, so
`ENABLE_PMIX=OFF` does not compile. The headers are vendored in `deps/pmix`, and
the OSHMPI runtime path never initializes the native OFI transport, so this costs
nothing at runtime.

Submit the first two-rank validation:

```bash
export ONECCL_SOURCE_DIR=$PWD
sbatch examples/oshmpi/validate_leonardo.sbatch
```

The same job supports topology overrides with `sbatch --nodes=...`,
`--ntasks-per-node=...`, and `--gres=gpu:...`.

## Runtime Variables

- `CCL_BACKEND=oshmpi` selects the backend.
- `CCL_OSHMPI_STAGING_SIZE` sets total symmetric staging capacity. The default
  is `64M`, split equally between source and destination lanes.
- `SHMEM_SYMMETRIC_SIZE` must be large enough for OSHMPI and oneCCL staging.

All PEs must use identical staging settings and collective order. The final
OSHMPI communicator must be destroyed collectively before externally owned MPI
is finalized.
