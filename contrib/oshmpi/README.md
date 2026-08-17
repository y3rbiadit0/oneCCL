# OSHMPI Backend

Nothing in this directory is built or installed by oneCCL. It holds the OSHMPI
patch the backend requires, the measurements behind the design, and the scripts
used to build and validate the backend on CINECA Leonardo. The backend itself lives in
`src/common/oshmpi/` and `src/comm/oshmpi_comm.*`; its tests are in
`tests/oshmpi/`.

The backend provides blocking oneCCL collectives through OSHMPI: barrier,
allgather, allreduce, alltoall and broadcast over `SHMEM_TEAM_WORLD`, plus
send/recv. User buffers are copied through bounded symmetric staging.

Device buffers are supported through SYCL - classified with
`sycl::get_pointer_type` and staged with `queue.memcpy` - so the backend is not
tied to any one vendor. Pinning the staging arena is the only CUDA-dependent
part and is optional (`CCL_ENABLE_OSHMPI_PINNED_STAGING`, off by default).

The backend requires an OSHMPI build containing
`0001-preserve-external-mpi-ownership.patch`. This prevents
`shmem_finalize()` from finalizing MPI when the application initialized MPI
and preserves the MPI thread level while OSHMPI initializes MPI_T.

Neither behaviour can be worked around from oneCCL: the thread-level downgrade
happens inside `shmem_init_thread`, and OSHMPI finalizes MPI from an exit handler
even if `shmem_finalize()` is never called. Both are defects affecting any OSHMPI
embedder, so the patch is a candidate for upstreaming - if it lands in OSHMPI, this
directory's `patches/` and the patching step in `build_oshmpi.sh` become
unnecessary and the dependency reduces to a stock OSHMPI build.

## Building

Both scripts load no modules and source nothing. They take a prepared environment
through documented variables, so they work on any site rather than one. Builds are
written to a scratch filesystem, including the OSHMPI clone; nothing lands in the
source tree, and only the install prefixes land in `$HOME`.

### 1. Patched OSHMPI

```bash
export MPI_C_COMPILER=$(command -v mpicc)
export MPI_CXX_COMPILER=$(command -v mpicxx)
export OSHMPI_CUDA_ROOT=$CUDA_ROOT       # optional, enables OSHMPI CUDA support

./contrib/oshmpi/build_oshmpi.sh
```

Clones OSHMPI on first run, checks out the pinned revision in a detached worktree,
applies the ownership patch, builds and installs it. The pins are:

```text
OSHMPI ee5cf110e673c098707257bb025404e17ac0a5fc
OpenPA 0475704dde41054db33562a8d17314fe0e30aaf3
```

Override `OSHMPI_BUILD_ROOT` (defaults to `$SCRATCH`), `OSHMPI_INSTALL_PREFIX`, or
`OSHMPI_UPSTREAM` for a mirror.

### 2. oneCCL

```bash
export ONECCL_C_COMPILER=/path/to/clang      # SYCL-capable
export ONECCL_CXX_COMPILER=/path/to/clang++
export ONECCL_SYCL_FLAGS="-fsycl -fsycl-targets=nvptx64-nvidia-cuda"
export MPI_C_COMPILER=$(command -v mpicc)    # the same MPI as above
export OSHMPI_ROOT=$HOME/opt/oshmpi-ee5cf110-oneccl

./contrib/oshmpi/build_oneccl.sh
```

The script refuses an OSHMPI without the ownership patch. `ONECCL_BUILD_ROOT`
defaults to `$SCRATCH`; `ONECCL_BUILD_DIR` must not contain a CMake cache from
another source tree or compiler.

`ONECCL_OSHMPI_PINNED_STAGING` defaults to `ON` here and pins the staging arena
with CUDA. Set it `OFF` to build with no CUDA at all, which is the upstream
default - see the measurements below for what it costs.

This is an OSHMPI-focused artifact: the native oneCCL MPI and stub backends are
disabled. Note the tests are registered with ctest under the `oshmpi` label, so
`ctest -L oshmpi` runs them wherever a launcher is available.

### Site drivers

`comm-playground` has a Leonardo driver that supplies this environment and runs
both scripts plus its own build in one step
(`cluster/leonardo/oneccl-oshmpi/bootstrap.sh`), along with the SLURM job that
runs the test suite. Site specifics live there rather than here.

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

## Measurements

### Device pointers through OSHMPI collectives

2 PEs on one node. `space` is CUDA symmetric memory from
`shmemx_space_create(SHMEMX_MEM_CUDA)`; `raw` is plain `cudaMalloc`.

| collective | space | raw  | notes                                          |
|------------|-------|------|------------------------------------------------|
| barrier    | PASS  | n/a  | no operands                                     |
| allgather  | PASS  | PASS | `MPI_Allgather`, UCX moves device memory        |
| alltoall   | PASS  | PASS | `MPI_Alltoall`, same                            |
| broadcast  | SEGV  | SEGV | OSHMPI host `memcpy` on the root PE             |
| allreduce  | SEGV  | SEGV | no accelerated coll component; host CPU reduces |

The dividing line is moving bytes versus computing on them. OSHMPI forwards the
caller's pointers straight to MPI with no memkind handling, so pure data movement
inherits MPI's CUDA-awareness. The two failures have unrelated causes:
`OSHMPI_broadcast_team()` ends with a host `memcpy` over a device pointer
(`src/shmem/coll.c:66`, an OSHMPI defect); `allreduce` falls back to Open MPI's
`libnbc`, which reduces on the host CPU because this stack builds with HCOLL and
UCC disabled. Whether enabling either lifts the restriction is still open.

`raw` passing shows symmetric allocation is not required by this *implementation*.
It is still required by the OpenSHMEM specification, so relying on it would tie
oneCCL to OSHMPI's internals. Treat it as a measurement, not a licence.

### Device layer and staging cost

allreduce, 2 ranks on one node, peak bandwidth in GB/s:

|                     | unpinned | pinned |
|---------------------|---------:|-------:|
| CUDA device layer   |    2.553 |  4.101 |
| SYCL device layer   |    2.546 |  3.694 |

The move from CUDA to SYCL is free (2.546 against 2.553 unpinned). Pinning the
staging arena is worth 45-60%, and the SYCL copy path does honour a
`cudaHostRegister`ed pointer - the assumption that it would not was wrong, and
cost 31% of peak until it was measured. The ~10% between the pinned figures is
per-call accessor overhead, visible only once the copy is fast enough for fixed
costs to matter.

For scale, NCCL through oneCCL reaches 56 GB/s on the same pair of GPUs by staying
on the device over NVLink. No host-staged design competes with that on one node;
the comparison that matters here is multi-node, where NCCL also loses NVLink.

### C++ note

`shmemx.h` has no `extern "C"` guard of its own and the `<shmem.h>` it includes
closes its guard first, so the space API is name-mangled and fails to link from
C++. Any use of the space API needs the include wrapped.

## Runtime Variables

- `CCL_BACKEND=oshmpi` selects the backend.
- `CCL_OSHMPI_STAGING_SIZE` sets total symmetric staging capacity. The default
  is `64M`, split equally between source and destination lanes.
- `SHMEM_SYMMETRIC_SIZE` must be large enough for OSHMPI and oneCCL staging.

All PEs must use identical staging settings and collective order. The final
OSHMPI communicator must be destroyed collectively before externally owned MPI
is finalized.
