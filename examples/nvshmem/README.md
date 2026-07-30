# NVSHMEM M0 Interoperability Spike

This standalone executable validates the toolchain assumptions required by the planned oneCCL NVSHMEM backend. It is deliberately not connected to `CCL_BACKEND` or normal oneCCL communicator dispatch.

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
