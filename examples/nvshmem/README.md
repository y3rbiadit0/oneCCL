# NVSHMEM M0 Interoperability Spike

The M0 standalone executable validates the toolchain assumptions required by
the oneCCL NVSHMEM backend. M0 remains independent of normal communicator
dispatch; the M1 skeleton is selected separately with `CCL_BACKEND=nvshmem`.

The spike verifies this path:

```text
ordinary SYCL device USM
  -> copy to NVSHMEM symmetric memory
  -> NVSHMEM float sum reduction on a CUDA stream extracted from SYCL
  -> NVSHMEM barrier on the same stream
  -> copy to ordinary SYCL device USM
  -> completion through a SYCL event
```

## Requirements

- Linux with NVIDIA GPUs
- CMake 3.20 or newer
- DPC++ with the SYCL CUDA backend
- CUDA Toolkit 12 or newer and NVCC
- An MPI implementation compatible with the NVSHMEM package
- NVSHMEM with team collectives and `nvshmemx_*_on_stream` APIs (NVHPC 24.5's
  bundled NVSHMEM 2.11 is supported by this spike)
- `mpirun` or a scheduler launcher capable of starting MPI processes, such as `srun`
- One NVSHMEM PE per GPU

The build uses `nvshmem::nvshmem_host` and `nvshmem::nvshmem_device` when an
NVSHMEM CMake package exports them. For installations without
`NVSHMEMConfig.cmake`, including the NVHPC 24.5 bundle, it discovers
`include/nvshmem*.h`, `lib/libnvshmem_host.so`, and
`lib/libnvshmem_device.a` directly under `NVSHMEM_HOME`.

## Configure

The recommended driver validates the dependency roots, prints compiler versions and exact commands, and selects DPC++ and NVCC separately:

```bash
export DPCPP_ROOT=/path/to/dpcpp
export CUDA_HOME=/path/to/cuda
export NVSHMEM_HOME=/path/to/nvshmem
export MPI_CXX_COMPILER=/path/to/mpicxx
export GCC12_ROOT=/path/to/gcc

./examples/nvshmem/build_m0.sh --cuda-arch 80
```

Use `--dry-run` to inspect the generated commands without checking paths or modifying the build directory. Run `./examples/nvshmem/build_m0.sh --help` for all root, compiler, build-directory, and CMake override options.

The equivalent manual configuration is:

```bash
cmake \
  -S examples/nvshmem \
  -B build-nvshmem-m0 \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$DPCPP_ROOT/bin/clang++" \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CUDA_HOME" \
  -DMPI_CXX_COMPILER="$MPI_CXX_COMPILER" \
  -DNVSHMEM_ROOT="$NVSHMEM_HOME" \
  -DNVSHMEM_M0_CUDA_ARCHITECTURE=80
```

Add `-DNVSHMEM_DIR="$NVSHMEM_HOME/lib/cmake/nvshmem"` when the installation
provides `NVSHMEMConfig.cmake`.

Use CUDA architecture `80` for NVIDIA A100, `90` for NVIDIA H100, or the value appropriate for the target GPU.

## Build

The build driver configures and builds by default. For a manually configured tree, run:

```bash
cmake --build build-nvshmem-m0 --parallel
```

## Leonardo

The Leonardo wrapper composes the two validated comm-playground environments.
It first loads the `cuda` stack to capture NVHPC 24.5's CUDA 12.4 and bundled
NVSHMEM 2.11 paths, then loads the `sycl` stack to establish DPC++, GCC 12,
hwloc, and the SYCL CUDA runtime. It then replaces the SYCL stack's Open MPI
with `hpcx-mpi/2.19`, matching the MPI implementation used by NVHPC NVSHMEM.
The captured NVHPC paths are restored before the generic M0 builder runs.
Module versions remain defined by comm-playground; this repository only
provides the composition layer in `leonardo_env.sh`:

```bash
export COMM_PLAYGROUND_ROOT=/path/to/comm-playground

./examples/nvshmem/build_leonardo_m0.sh \
  --env-script "$COMM_PLAYGROUND_ROOT/cluster/leonardo/environment.sh"
```

The wrappers source `leonardo_env.sh` automatically. To establish the same
environment in the current shell for inspection or several commands, source it
directly and tell the wrappers not to reload it:

```bash
source examples/nvshmem/leonardo_env.sh \
  "$COMM_PLAYGROUND_ROOT/cluster/leonardo/environment.sh"

./examples/nvshmem/build_leonardo_m0.sh --skip-env
./examples/nvshmem/validate_leonardo_m0.sh --skip-env -- \
  --nodes 1 --pes-per-node 2
```

Executing `leonardo_env.sh` instead of sourcing it is rejected because module
and environment changes from an executed child process cannot affect the
current shell.

The default build directory is
`$SCRATCH/oneccl-nvshmem-m0-build`. Use `--build-dir` to override it and
`--dry-run` to inspect the generated commands. GPU runtime diagnostics run only
when `/dev/nvidiactl` is available. A login-node build therefore does not report
the expected absence of the driver-provided `libcuda.so.1` as a build problem.

Run validation inside a Slurm allocation. The validation wrapper independently
recreates the hybrid environment because variables exported by the executed
build script do not propagate back to its parent shell:

```bash
./examples/nvshmem/validate_leonardo_m0.sh -- \
  --nodes 1 --pes-per-node 2

./examples/nvshmem/validate_leonardo_m0.sh -- \
  --nodes 2 --pes-per-node 1
```

Leonardo uses the same initialization and runtime model as comm-playground's
validated NVSHMEM programs: `MPI_Init`, CUDA device selection from the MPI
node-local rank, and `nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, ...)`.
The validator sets `NVSHMEM_BOOTSTRAP=MPI` and uses plain `srun`, leaving Slurm
MPI integration at Leonardo's site default. It also reuses
`cluster/leonardo/runtime/nvshmem.sh` when the comm-playground location is known.

## Run

Use the validation driver for repeatable one-node and multi-node checks. It runs element counts 1, 1024, and 1048576 by default:

```bash
./examples/nvshmem/validate_m0.sh \
  --launcher mpirun \
  --nodes 1 \
  --pes-per-node 2

./examples/nvshmem/validate_m0.sh \
  --launcher srun \
  --nodes 2 \
  --pes-per-node 1
```

`validate_m0.sh` supports Intel MPI/Hydra or Open MPI through `mpirun`, and
MPI processes launched through Slurm. The generic default is Open MPI:

```bash
./examples/nvshmem/validate_m0.sh --launcher mpirun --mpirun-flavor intel
./examples/nvshmem/validate_m0.sh --launcher mpirun --mpirun-flavor openmpi
./examples/nvshmem/validate_m0.sh --launcher srun
./examples/nvshmem/validate_m0.sh --launcher srun --srun-mpi pmix
```

The default Slurm command omits `--mpi`, matching comm-playground's validated
Leonardo jobs. Use `srun --mpi=list` before explicitly overriding the site
default with `--srun-mpi`.

The executable also accepts one optional element-count argument when a direct launch is useful:

```bash
mpirun -np 2 --map-by ppr:2:node \
  build-nvshmem-m0/oneccl_nvshmem_m0 1048576
```

The executable prints `PASSED` on PE 0 after collectively checking all PE results. It does not call `cudaDeviceSynchronize`; the collective result must be covered by the SYCL event submitted after the external NVSHMEM work.

MPI initialization occurs before CUDA and NVSHMEM initialization. The spike
derives a node-local rank with `MPI_Comm_split_type`, selects ordinal zero when
the scheduler exposes one GPU per process, and otherwise selects the MPI
node-local rank. CUDA device selection is complete before
`nvshmemx_init_attr`. Before initializing NVSHMEM, ranks on each node exchange
CUDA UUIDs and reject duplicate physical-GPU assignments. Set
`ONECCL_NVSHMEM_M0_DEVICE` to override the visible CUDA device ordinal for
unusual mappings. Leonardo also uses `CUDA_DEVICE_ORDER=PCI_BUS_ID` to keep CUDA
and Slurm GPU numbering consistent.

## Expected Failures

Configuration fails before compilation when:

- The C++ compiler does not support `-fsycl`.
- NVCC or the CUDA Toolkit is unavailable.
- MPI or its C++ wrapper cannot be found.
- NVSHMEM headers or either required host/device library cannot be found.
- An NVSHMEM CMake package exports incomplete application targets and no direct
  installation prefix is available.

Runtime fails collectively when:

- No CUDA device is visible.
- DPC++ does not expose the CUDA device selected for the local PE.
- More PEs than GPUs are assigned to a node.
- MPI ranks and NVSHMEM PEs do not have matching world or node-local layouts.
- Symmetric or SYCL device allocation fails.
- The NVSHMEM collective or barrier returns an error.
- The reduction result is incorrect.

## Validated Configuration

M0 completed on Leonardo on 2026-07-31 with DPC++ 21, NVHPC CUDA 12.4,
NVHPC NVSHMEM 2.11, and HPC-X 2.19. Slurm job 51172113 passed with two PEs
on one node, and job 51172241 passed with one PE on each of two nodes. Both
topologies passed element counts 1, 1024, and 1048576.

## M1 Backend Skeleton

M1 adds the optional `CCL_ENABLE_NVSHMEM` build switch, `CCL_BACKEND=nvshmem`
parsing, CUDA SYCL device/context validation, native KVS reuse, and a retained
in-order queue. NVSHMEM runtime initialization starts in M2; M1 collectives and
group calls fail explicitly instead of falling back to native algorithms.

On Leonardo, source the hybrid environment and use the existing oneCCL build
driver with NVSHMEM enabled and NCCL disabled:

```bash
source examples/nvshmem/leonardo_env.sh

./build-leonardo-sycl.sh --skip-env \
  --build-dir "$SCRATCH/oneccl-nvshmem-m1-build" \
  --install-prefix "$HOME/opt/oneccl-nvshmem-m1" \
  --no-examples --no-install -- \
  -DCCL_ENABLE_NCCL=OFF \
  -DCCL_ENABLE_NVSHMEM=ON \
  -DENABLE_MPI_TESTS=ON \
  -DBUILD_FT=ON \
  -DNVSHMEM_ROOT="$NVSHMEM_HOME"
```

Run the hardware-gated two-rank smoke test from a one-node Slurm allocation:

```bash
cmake --build "$SCRATCH/oneccl-nvshmem-m1-build" \
  --parallel 16 --target ccl nvshmem_m1_comm_test
ctest --test-dir "$SCRATCH/oneccl-nvshmem-m1-build" \
  --output-on-failure -R '^nvshmem_m1_comm_test$'
```

The CTest definition uses the MPI launcher matching `MPI_DIR` and supplies the
bundled Intel MPI, oneCCL library, Slurm bootstrap, shared-memory fabric, and
oneCCL MPI transport environment. Do not export those settings globally in
`leonardo_env.sh`: M0's NVSHMEM bootstrap intentionally uses HPC-X instead.

M1 completed on Leonardo on 2026-08-01. The disabled build passed a two-rank
native allreduce and rejected `CCL_BACKEND=nvshmem`. The enabled,
NCCL-disabled CTest passed with two ranks and two GPUs in 3.99 seconds, and
`libccl.so` had no NCCL or NVSHMEM runtime dependency.

## M2 UID Host-Library Spike

NVSHMEM 2.11 introduced socket-based UID bootstrap and host-library-only
initialization. The `oneccl_nvshmem_uid_spike` executable validates those APIs
before runtime lifecycle is added to `libccl`. MPI distributes the opaque UID
in this standalone test, but NVSHMEM is initialized with
`NVSHMEMX_INIT_WITH_UNIQUEID` and does not use an MPI bootstrap plugin.

Build the spike with the existing Leonardo M0 driver:

```bash
./examples/nvshmem/build_leonardo_m0.sh --configure-only
cmake --build "$SCRATCH/oneccl-nvshmem-m0-build" \
  --parallel 16 --target oneccl_nvshmem_uid_spike
```

The Leonardo build wrapper selects the loaded Binutils 2.42 linker explicitly.
Leonardo's system `/usr/bin/ld` cannot read GCC 12's compressed sections and
does not support CMake 4.1's linker dependency-file option. Override automatic
selection with `--linker FILE` or `NVSHMEM_M0_LINKER` when needed.

Within a one-node, two-GPU Slurm allocation, load the runtime environment and
launch two PEs:

```bash
source examples/nvshmem/leonardo_env.sh

export NVSHMEM_REMOTE_TRANSPORT=${NVSHMEM_REMOTE_TRANSPORT:-ibrc}
export NVSHMEM_IB_ENABLE_IBGDA=${NVSHMEM_IB_ENABLE_IBGDA:-0}
export NVSHMEM_DISABLE_NCCL=${NVSHMEM_DISABLE_NCCL:-1}
export NVSHMEM_IB_SL=${NVSHMEM_IB_SL:-1}
export NVSHMEM_SYMMETRIC_SIZE=${NVSHMEM_SYMMETRIC_SIZE:-1G}
unset NVSHMEM_BOOTSTRAP

srun --nodes=1 --ntasks=2 --ntasks-per-node=2 --gpus-per-task=1 \
  "$SCRATCH/oneccl-nvshmem-m0-build/oneccl_nvshmem_uid_spike"
```

The expected output is `PASSED: NVSHMEM UID host-library bootstrap`. Repeat
with `--nodes=2 --ntasks=2 --ntasks-per-node=1` to validate cross-node UID
bootstrap.

## M2 oneCCL Runtime Validation

Configure the production backend with NVCC, NVSHMEM host-library linkage, and
the hardware tests enabled:

```bash
source examples/nvshmem/leonardo_env.sh

./build-leonardo-sycl.sh --skip-env \
  --build-dir "$SCRATCH/oneccl-nvshmem-m2-build" \
  --install-prefix "$HOME/opt/oneccl-nvshmem-m2" \
  --clean --no-examples --no-install --configure-only -- \
  -DCCL_ENABLE_NCCL=OFF \
  -DCCL_ENABLE_NVSHMEM=ON \
  -DENABLE_MPI_TESTS=ON \
  -DBUILD_FT=ON \
  -DNVSHMEM_ROOT="$NVSHMEM_HOME"

cmake --build "$SCRATCH/oneccl-nvshmem-m2-build" \
  --parallel 16 --target ccl nvshmem_m1_comm_test
```

From a one-node, two-GPU allocation, run both the runtime/staging test and its
failure-path test:

```bash
ctest --test-dir "$SCRATCH/oneccl-nvshmem-m2-build" \
  --output-on-failure -L nvshmem
```

The positive test uses a 1 MiB arena and exercises shared/device USM staging
for messages below, equal to, and above the resulting lane size. Collective
APIs remain explicitly unsupported until M4.

For the two-node exit test, request exactly one task and one GPU per node:

```bash
salloc \
  -A IscrC_HIGRAPH_0 \
  -p boost_usr_prod \
  --time=00:10:00 \
  --nodes=2 \
  --ntasks-per-node=1 \
  --gres=gpu:1 \
  --cpus-per-task=8
```

The bundled Intel MPI needs libfabric for cross-node startup. The following
uses the Intel MPI libfabric installation already validated by
comm-playground's oneCCL jobs. Do not source its `oneccl-nccl.sh` runtime
directly because that script forces `CCL_BACKEND=nccl`.

```bash
repo=$PWD
build_dir="$SCRATCH/oneccl-nvshmem-m2-build"
mpi_root="$repo/deps/mpi"

export COMM_PLAYGROUND_ROOT=${COMM_PLAYGROUND_ROOT:-$HOME/Projects/hpc-comm-playground}
export ONECCL_NCCL_ROOT=${ONECCL_NCCL_ROOT:-$HOME/opt/oneccl-nccl-leonardo}
source "$repo/examples/nvshmem/leonardo_env.sh"

fabric_dir=${ONECCL_LIBFABRIC_DIR:-$ONECCL_NCCL_ROOT/opt/mpi/libfabric/lib}
provider_dir=${ONECCL_LIBFABRIC_PROVIDER_DIR:-$fabric_dir/prov-tcp-only}
test_bin="$build_dir/tests/nvshmem/nvshmem_m1_comm_test"

test -f "$fabric_dir/libfabric.so.1"
test -f "$fabric_dir/prov/libtcp-fi.so"
mkdir -p "$provider_dir"
ln -sf "$fabric_dir/prov/libtcp-fi.so" "$provider_dir/libtcp-fi.so"

export LC_ALL=C
export PATH="$mpi_root/bin:$PATH"
export LD_LIBRARY_PATH="$build_dir/src:$mpi_root/lib:$fabric_dir:$ONECCL_NCCL_ROOT/opt/mpi/lib/release:$ONECCL_NCCL_ROOT/opt/mpi/lib:${LD_LIBRARY_PATH:-}"

env -u NVSHMEM_BOOTSTRAP \
  CCL_BACKEND=nvshmem \
  CCL_ATL_TRANSPORT=mpi \
  CCL_MPI_LIBRARY_PATH="$mpi_root/lib/libmpi.so.12" \
  CCL_NVSHMEM_VALIDATE_STAGING=1 \
  CCL_NVSHMEM_STAGING_SIZE=1M \
  I_MPI_ROOT="$mpi_root" \
  I_MPI_HYDRA_BOOTSTRAP=slurm \
  I_MPI_FABRICS=shm:ofi \
  I_MPI_OFI_PROVIDER=tcp \
  I_MPI_DEBUG=5 \
  FI_PROVIDER=tcp \
  FI_PROVIDER_PATH="$provider_dir" \
  FI_LOG_LEVEL=error \
  NVSHMEM_REMOTE_TRANSPORT=ibrc \
  NVSHMEM_IB_ENABLE_IBGDA=0 \
  NVSHMEM_DISABLE_NCCL=1 \
  NVSHMEM_IB_SL=1 \
  NVSHMEM_SYMMETRIC_SIZE=1G \
  "$mpi_root/bin/mpiexec.hydra" -n 2 -ppn 1 \
  "$COMM_PLAYGROUND_ROOT/cluster/leonardo/gpu-rank-wrapper.sh" \
  "$test_bin"
```

`I_MPI_DEBUG=5` prints the rank-to-node mapping. The deliberate host-pointer
rejection emits `CCL_ERROR` diagnostics during this test and is expected.

M2 completed on Leonardo on 2026-08-01. The one-node CTest passed both tests.
Slurm job 51500159 passed the production runtime with rank 0 on `lrdn0244` and
rank 1 on `lrdn0245`, using Intel MPI 2021.17, libfabric 2.2.0-impi with the TCP
provider, NVSHMEM UID bootstrap over `ib0`, and the NVSHMEM `ibrc` transport.
