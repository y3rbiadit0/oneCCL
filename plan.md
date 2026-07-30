# NVSHMEM Collective Backend Plan

## Status

- State: M0 implementation in progress
- Target API: Existing oneCCL C++ collective API
- Target compiler: DPC++ with the SYCL CUDA backend
- Target NVSHMEM version: NVHPC 24.5's bundled 2.11 for M0; 3.7 or newer remains
  the target for later arbitrary-team backend work
- Buffer policy: Transparent staging through an internal symmetric workspace
- Backend selection: `CCL_BACKEND=nvshmem`
- Initial communicator model: One process and one NVSHMEM PE per GPU

## Goal

Add an optional NVSHMEM backend that preserves the existing oneCCL public API while executing supported CUDA-device collectives through NVSHMEM. The first implementation will use host-enqueued `nvshmemx_*_on_stream` operations and return normal oneCCL events. Internal kernel-initiated algorithms may be added after the stream-based path is correct and measured.

The intended operation path is:

```text
ccl::* public API
  -> comm_interface
  -> nvshmem_comm
  -> stream-ordered copy to symmetric workspace
  -> nvshmemx_*_on_stream
  -> stream-ordered copy to the user buffer
  -> ccl::event backed by a SYCL event
```

## Non-Goals

- No compiler pass that rewrites oneCCL calls into NVSHMEM calls.
- No application-kernel-callable oneCCL API in the first implementation.
- No NVSHMEM implementation as an ATL transport.
- No OSHMPI backend in the initial work.
- No requirement that application buffers are allocated by NVSHMEM.
- No simultaneous use of multiple GPUs by one PE in the first implementation.
- No silent semantic weakening for unsupported datatypes, reductions, dependencies, or events.

## Design Requirements

- Preserve all existing public function signatures.
- Keep NVSHMEM optional and compile the existing native, NCCL, and RCCL configurations unchanged.
- Select NVSHMEM only for CUDA-backed device communicators.
- Continue using the native backend for host communicators.
- Accept arbitrary supported oneCCL buffers by staging through symmetric memory.
- Treat streamless overloads as valid by retaining an internal in-order SYCL queue.
- Honor input dependencies before staging or communication starts.
- Complete the returned event only after the result has reached the user receive buffer.
- Serialize collectives that use the same NVSHMEM team.
- Avoid collective symmetric allocation in ordinary operation submission.
- Track initialization ownership and never finalize an externally initialized NVSHMEM runtime.
- Provide explicit errors for unsupported combinations until a correct fallback exists.

## Architecture

### Backend Boundary

Add `nvshmem_comm` as a `ccl::comm_interface` implementation, following the high-level NCCL backend structure but not copying its dependency, empty-stream, temporary-allocation, or event limitations.

Expected primary files:

- `src/comm/nvshmem_comm.hpp`
- `src/comm/nvshmem_comm.cpp`
- `src/common/nvshmem/nvshmem_runtime.hpp`
- `src/common/nvshmem/nvshmem_runtime.cpp`
- `src/common/nvshmem/nvshmem_adapter.h`
- `src/common/nvshmem/nvshmem_adapter.cu`
- `cmake/FindNVSHMEM.cmake`
- `tests/nvshmem/`
- `examples/nvshmem/`

Expected integration files:

- `CMakeLists.txt`
- `cmake/helpers.cmake`
- `include/oneapi/ccl/config.h.in`
- `src/CMakeLists.txt`
- `src/common/env/env.hpp`
- `src/common/env/env.cpp`
- `src/common/env/vars.hpp`
- `src/comm/comm_selector.cpp`
- `src/common/api_wrapper/api_wrapper.cpp`, if common lifecycle hooks remain necessary
- `doc/rst/source/env-variables.rst`
- `doc/rst/source/introduction/installation.rst`

### Compiler Boundary

Use a narrow C ABI between DPC++ code and CUDA/NVSHMEM code. Compile NVSHMEM-dependent device code with NVCC and relocatable device code enabled. Link both `libnvshmem_host.so` and `libnvshmem_device.a` as required by NVSHMEM.

Do not rely on `dlopen` alone. Device symbols from `libnvshmem_device.a` require build-time device linking. Hide NVSHMEM device symbols included in the oneCCL shared library to avoid conflicts with applications that also use NVSHMEM.

### Runtime Ownership

Initialize NVSHMEM lazily during creation of the first NVSHMEM device communicator, after the CUDA device has been selected. Prefer UID-based bootstrap distributed through `kvs_interface` so the backend does not require MPI initialization.

Maintain process-global state for:

- Initialization status and ownership
- PE rank and world size
- Selected CUDA device
- Symmetric staging arena
- Communicator-to-team mappings
- Outstanding operation count
- Finalization coordination

### Staging Arena

Allocate a fixed process-global symmetric arena once after initialization and release it only during owned finalization. Divide it into source, destination, metadata, and signal regions.

Use deterministic chunking for operations larger than the arena. The initial implementation will permit one in-flight NVSHMEM collective per process to guarantee corresponding symmetric offsets and prevent reuse. Later milestones may add multiple deterministic lanes and concurrency between distinct teams.

Proposed configuration:

- `CCL_NVSHMEM_STAGING_SIZE`, default `64M`
- `CCL_NVSHMEM_LIBRARY_PATH`, when an explicit host library location is required

### Stream And Event Model

Require an in-order SYCL CUDA queue for user-supplied streams. Extract its native `cudaStream_t` using SYCL CUDA interoperability. For streamless calls, create and retain an internal in-order queue from the communicator device and context.

The operation sequence must be visible in CUDA stream order:

```text
dependency barrier
  -> user-to-symmetric copy
  -> NVSHMEM collective
  -> symmetric-to-user copy
  -> final SYCL barrier event
```

Use CUDA events or a dedicated internal stream to serialize operations on the same team across different user streams. A host mutex alone is not sufficient because CUDA execution continues after enqueueing.

## Milestones

| Milestone | Deliverable | Exit Criterion | Status |
|---|---|---|---|
| M0 | Toolchain and interoperability spike | A two-PE DPC++ program invokes an NVSHMEM on-stream collective on a native CUDA stream and observes completion through a SYCL event | In progress |
| M1 | Optional backend skeleton | oneCCL builds with and without NVSHMEM, parses `CCL_BACKEND=nvshmem`, and constructs an NVSHMEM communicator | Not started |
| M2 | Runtime, bootstrap, and symmetric staging | UID bootstrap, ownership-safe lifecycle, fixed symmetric arena, and chunked stream-ordered copies work across two nodes | Not started |
| M3 | Dependencies, events, and ordering | Stream-taking and streamless calls honor dependencies and serialize same-team operations across streams | Not started |
| M4 | Initial collective MVP | Barrier, allgather, allreduce, alltoall, and broadcast pass correctness tests on device USM buffers | Not started |
| M5 | Public API coverage and fallback | Remaining collectives and required datatype/reduction cases work directly, compositionally, or through a documented correct fallback | Not started |
| M6 | Internal kernel-initiated prototype | At least one collective is implemented by a oneCCL-owned CUDA kernel that invokes NVSHMEM device primitives | Not started |
| M7 | Production readiness | Documentation, packaging, CI coverage, performance data, and failure-path testing are complete | Not started |

## M0: Toolchain Spike

Objective: Retire the highest-risk compiler, linker, stream interoperability, and runtime assumptions before modifying normal oneCCL dispatch.

- [ ] Confirm the available NVSHMEM package version and CMake config targets.
- [ ] Confirm DPC++ can create an in-order CUDA-backed SYCL queue on the target system.
- [ ] Extract `cudaStream_t` from the SYCL queue.
- [ ] Build a CUDA adapter translation unit with NVCC and relocatable device code.
- [ ] Link `libnvshmem_host.so` and `libnvshmem_device.a` into a small mixed DPC++/CUDA executable.
- [ ] Initialize NVSHMEM using UID bootstrap or the launcher-supported bootstrap for the spike.
- [ ] Allocate a small symmetric source and destination buffer.
- [ ] Enqueue an NVSHMEM barrier and one data collective on the extracted stream.
- [ ] Submit a SYCL barrier after the NVSHMEM call and verify that waiting on the SYCL event observes the result.
- [ ] Run with two PEs on one node.
- [ ] Run with two PEs on separate nodes.
- [ ] Record exact compiler, CUDA, NVSHMEM, linker, and launcher commands in `examples/nvshmem/README.md` or the implementation notes.

M0 acceptance tests:

- The build uses DPC++ for SYCL code and NVCC only for the adapter.
- No global device synchronization is needed for correctness.
- The final SYCL event covers NVSHMEM work on the extracted stream.
- The executable exits without an initialization or finalization hang.
- A deliberately incorrect PE count or device mapping fails with a useful error.

M0 implementation checkpoint, 2026-07-29:

- Added the standalone mixed DPC++/NVCC spike under `examples/nvshmem/`.
- Added CMake checks for DPC++ `-fsycl`, CUDA Toolkit 12+, NVCC, and both
  NVSHMEM host/device libraries, using package targets or direct discovery.
- Added native CUDA stream extraction, symmetric staging, on-stream float sum reduction, on-stream barrier, SYCL completion event, count consensus, and collective result validation.
- Added launcher, GPU mapping, configure, build, and run instructions in `examples/nvshmem/README.md`.
- Added `build_m0.sh` for dependency diagnostics and reproducible mixed-toolchain builds.
- Added `validate_m0.sh` for one-node and multi-node validation with NVSHMEM, Intel MPI/Open MPI, or Slurm launchers.
- Local configure validation stops at NVCC discovery because the current development host is macOS/arm64 without DPC++, CUDA, NVCC, NVSHMEM, MPI, or an NVSHMEM launcher.
- M0 remains in progress until the documented build and the one-node and multi-node runs pass on the target Linux GPU environment.

M0 Leonardo compatibility checkpoint, 2026-07-31:

- Added direct NVSHMEM header/library discovery for NVHPC installations that do
  not ship `NVSHMEMConfig.cmake`; package targets remain preferred when present.
- Added `leonardo_env.sh` as a reusable composition layer over
  comm-playground's existing `cuda` and `sycl` environments. It preserves
  NVHPC 24.5 CUDA 12.4/NVSHMEM 2.11 while retaining the validated DPC++/GCC
  12/hwloc stack.
- Added `build_leonardo_m0.sh` to source that environment and invoke the generic
  mixed-toolchain build reproducibly.
- Added `validate_leonardo_m0.sh` to recreate the hybrid runtime environment,
  restore NVSHMEM transport/plugin paths, and generate MPI-bootstrapped Slurm
  launches for one-node and multi-node validation.
- Leonardo's NVHPC 24.5 package was confirmed to lack the PMI bootstrap plugin.
  M0 now follows comm-playground's validated path: HPC-X 2.19, explicit
  `MPI_Init`, CUDA selection from an MPI shared-memory rank, and
  `nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, ...)`.
- The adapter verifies MPI world rank/size against NVSHMEM PE rank/size and
  exchanges CUDA UUIDs within each node to reject duplicate physical-GPU
  assignments before NVSHMEM initialization. It finalizes NVSHMEM before owned
  MPI state.
- The initial mixed DPC++/NVCC executable built successfully on Leonardo. The
  MPI-bootstrap revision still requires one-node and multi-node runtime runs.
- Local shell syntax, command generation, and whitespace checks pass. The local
  macOS host cannot compile the target; the revised build and runtime validation
  remain pending on Leonardo.

## M1: Backend Skeleton

Objective: Establish optional build and runtime selection without implementing data collectives.

- [ ] Add `CCL_ENABLE_NVSHMEM`, defaulting to `OFF`.
- [ ] Add `FindNVSHMEM.cmake` with an imported target and configuration-fatal checks when explicitly enabled dependencies are absent.
- [ ] Add `CCL_ENABLE_NVSHMEM` to generated configuration headers.
- [ ] Add `backend_mode::nvshmem` and the `"nvshmem"` parser mapping.
- [ ] Add NVSHMEM source registration and target-local compile definitions.
- [ ] Add `nvshmem_comm` implementing the required `comm_interface` surface.
- [ ] Add device communicator dispatch beside NCCL and RCCL.
- [ ] Keep host communicator creation on the native backend.
- [ ] Validate that the communicator uses a SYCL CUDA device and context.
- [ ] Add rank, size, device, context, and internal stream handling.
- [ ] Return explicit not-supported errors from unimplemented operations.
- [ ] Add a build-only test for NVSHMEM disabled.
- [ ] Add a build-and-create smoke test for NVSHMEM enabled.

M1 acceptance tests:

- Existing default builds do not require CUDA or NVSHMEM.
- `CCL_BACKEND=nvshmem` is rejected when support was not compiled.
- `CCL_BACKEND=nvshmem` rejects non-CUDA device communicators.
- Native host communicator behavior is unchanged.
- Communicator construction and destruction complete on all PEs.

## M2: Runtime And Staging

Objective: Support arbitrary oneCCL device buffers without per-operation symmetric allocation.

- [ ] Implement process-global NVSHMEM runtime state.
- [ ] Select and validate the CUDA device before the first NVSHMEM operation.
- [ ] Implement UID distribution through `kvs_interface`.
- [ ] Verify oneCCL rank and size against NVSHMEM PE rank and world size.
- [ ] Track external versus oneCCL-owned initialization.
- [ ] Allocate the fixed symmetric arena collectively once.
- [ ] Parse and validate `CCL_NVSHMEM_STAGING_SIZE`.
- [ ] Partition source, destination, metadata, and signal regions.
- [ ] Implement checked size arithmetic and deterministic chunk boundaries.
- [ ] Implement stream-ordered copies for device and shared USM pointers.
- [ ] Detect unsupported pointer residence and report it before communication starts.
- [ ] Prevent arena reuse until the returned operation event completes.
- [ ] Drain outstanding operations before owned finalization.

M2 acceptance tests:

- User buffers do not need to be NVSHMEM allocations.
- Messages smaller than, equal to, and larger than the staging lane work.
- Repeated operations do not grow the symmetric heap.
- Allocation and free calls occur in matching collective order.
- External NVSHMEM initialization is not finalized by oneCCL.
- Initialization failure leaves no partially usable communicator.

## M3: Events And Ordering

Objective: Match the oneCCL asynchronous operation contract.

- [ ] Convert compatible oneCCL dependency events into SYCL dependency barriers.
- [ ] Treat default events as already complete.
- [ ] Define a safe fallback for dependencies that cannot expose a SYCL event.
- [ ] Return a SYCL-backed event after the destination copy.
- [ ] Make `event::test`, `event::wait`, and `get_native` behave consistently.
- [ ] Return `false` for cancellation when NVSHMEM work cannot be canceled.
- [ ] Implement the internal queue for streamless overloads.
- [ ] Serialize same-team collectives across different user streams with CUDA-visible dependencies.
- [ ] Ensure failed enqueueing does not leave a staging lane permanently occupied.
- [ ] Define grouped-call behavior before enabling grouped NVSHMEM operations.

M3 acceptance tests:

- A producer SYCL kernel can feed a collective through `deps` without a host wait.
- A consumer SYCL kernel can depend on the returned native SYCL event.
- The receive buffer is not reported complete before the copy-back finishes.
- Two threads and two streams cannot overlap collectives on the same team.
- Streamless calls complete correctly.
- Event behavior remains valid on error paths.

## M4: Initial Collective MVP

Objective: Deliver a useful backend subset with transparent staging.

- [ ] Implement barrier with `nvshmemx_barrier_on_stream`.
- [ ] Implement allgather with `nvshmemx_fcollectmem_on_stream`.
- [ ] Implement alltoall with `nvshmemx_alltoallmem_on_stream`.
- [ ] Implement broadcast with `nvshmemx_broadcastmem_on_stream`.
- [ ] Implement allreduce for supported NVSHMEM datatype and reduction pairs.
- [ ] Implement `avg` only where sum plus stream-ordered division preserves oneCCL semantics.
- [ ] Add centralized datatype and reduction mapping with version guards.
- [ ] Reject custom reductions and unsupported low-precision behavior explicitly.
- [ ] Support in-place forms only where NVSHMEM and staging semantics are proven correct.
- [ ] Add examples for barrier and allreduce.
- [ ] Register two-PE and multi-PE CTests where GPU test infrastructure is available.

M4 acceptance matrix:

| Dimension | Required coverage |
|---|---|
| PE count | 2, local multi-PE, and multi-node |
| Message size | 0 where legal, 1 element, staging boundary, multi-chunk |
| Buffers | Device USM and shared USM |
| Streams | User in-order stream and internal stream |
| Dependencies | None, one producer event, chained collective event |
| Operations | Barrier, allgather, allreduce, alltoall, broadcast |
| Reductions | Sum, product, minimum, and maximum where supported |
| Errors | Wrong backend, wrong device, unsupported datatype, insufficient staging size |

## M5: API Coverage And Fallback

Objective: Make backend selection compatible with the intended public operation surface.

- [ ] Implement allgatherv using metadata exchange and chunked RMA.
- [ ] Implement alltoallv using metadata exchange and chunked RMA.
- [ ] Implement reduce using allreduce plus root extraction or a dedicated algorithm.
- [ ] Implement reduce-scatter with the NVSHMEM extension where supported.
- [ ] Provide an allreduce-plus-slice reduce-scatter fallback.
- [ ] Implement both public broadcast forms.
- [ ] Support vector-buffer overloads through packing and unpacking.
- [ ] Define send and receive through a put-with-signal protocol or route them to a native fallback.
- [ ] Add dynamically registered datatype handling for transfer-only operations.
- [ ] Route custom callback reductions to a correct native fallback.
- [ ] Implement or fall back for pre-multiplied reductions.
- [ ] Match documented FP16 and BF16 conversion behavior.
- [ ] Map arbitrary oneCCL communicators to NVSHMEM teams.
- [ ] Implement communicator split with arbitrary team initialization.
- [ ] Implement safe team destruction after outstanding events complete.
- [ ] Add grouped-call support or document and enforce its temporary exclusion.

M5 acceptance tests:

- Every public collective either completes correctly or takes a documented correct fallback.
- No public operation silently ignores attributes or dependencies.
- Parent and split communicators can coexist.
- Variable-count operations handle asymmetric but valid count vectors.
- Custom and low-precision reductions preserve oneCCL semantics.
- Point-to-point matching does not confuse concurrent peers or group IDs.

## M6: Kernel-Initiated Prototype

Objective: Evaluate true GPU-initiated communication without changing the public API.

- [ ] Select one operation where kernel initiation has a measurable expected benefit.
- [ ] Implement the operation in a oneCCL-owned CUDA kernel.
- [ ] Invoke NVSHMEM device primitives from the kernel.
- [ ] Use `nvshmemx_collective_launch` when the kernel contains synchronization or collective calls.
- [ ] Preserve the same staging, team, stream, dependency, and event contracts as M4.
- [ ] Add a runtime selector to compare on-stream and kernel-initiated implementations.
- [ ] Measure launch latency, overlap, bandwidth, and small-message scaling.
- [ ] Retain the on-stream implementation when kernel initiation is not beneficial.

M6 acceptance tests:

- The kernel path passes the same correctness suite as the on-stream path.
- The build does not expose NVSHMEM device symbols through the oneCCL public API.
- The application does not need to compile its own source with NVCC.
- Cooperative launch failures produce actionable errors rather than deadlocks.
- Performance data justifies retaining the kernel path.

## M7: Production Readiness

Objective: Prepare the backend for normal optional distribution and maintenance.

- [ ] Document installation, required versions, launchers, supported hardware, and environment variables.
- [ ] Document operation, datatype, reduction, buffer, and communicator support.
- [ ] Add third-party licensing and redistribution notices.
- [ ] Export or intentionally hide transitive CMake dependencies.
- [ ] Validate shared and static oneCCL builds.
- [ ] Validate coexistence with applications that also link NVSHMEM.
- [ ] Test initialization and finalization under normal exit and exceptions.
- [ ] Test invalid rank maps, device oversubscription, and bootstrap failures.
- [ ] Add sanitizer or equivalent host-side lifecycle coverage where possible.
- [ ] Add performance baselines against NCCL and native GPU-aware MPI.
- [ ] Add CI jobs or documented hardware-gated test procedures.
- [ ] Review documentation and code against oneCCL contribution requirements.

## Test Strategy

Use dedicated NVSHMEM tests rather than relying initially on the general benchmark, because the benchmark currently creates a host service communicator and may use out-of-order queues.

Test layers:

| Layer | Purpose |
|---|---|
| Adapter unit tests | Validate status conversion, datatype mapping, checked arithmetic, and environment parsing |
| Runtime tests | Validate bootstrap, ownership, arena lifecycle, and team lifecycle |
| Event tests | Validate dependency import, completion export, streamless calls, and cross-stream ordering |
| Collective tests | Validate values, in-place behavior, chunking, roots, counts, and reductions |
| Failure tests | Validate unsupported configurations and cleanup after partial failure |
| Performance tests | Compare latency and bandwidth with NCCL and native paths |

All GPU collectives should be tested with rank-dependent input patterns so rank-ordering mistakes cannot pass accidentally.

## Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| DPC++ and NVCC device-link incompatibility | Blocks device adapter integration | Resolve in M0 before backend wiring |
| SYCL does not observe externally enqueued CUDA work as expected | Incorrect completion events | Prove same-stream ordering in M0 and use explicit CUDA events if needed |
| Collective ordering differs across local threads | Deadlock | Dedicated internal stream and process-wide serialization initially |
| Symmetric arena is too small | Operation failure or excess memory use | Configurable arena plus deterministic chunking |
| User buffers cannot be classified | Invalid copies | Validate SYCL pointer residence and reject before enqueueing |
| NVSHMEM initialization conflicts with MPI or application ownership | Hangs or double finalization | UID bootstrap and explicit ownership tracking |
| Multiple communicators map inconsistently to teams | Wrong rank ordering or deadlock | Use KVS-distributed team unique IDs and verify rank maps |
| Low-precision reduction semantics differ | Numerically incorrect results | Use explicit conversion kernels or native fallback |
| Static NVSHMEM device symbols collide in a consumer | Link or runtime failure | Hide bundled symbols and test consumer coexistence |
| Native fallback requires unavailable ATL support | Incomplete API coverage | Define required deployment profile before M5 and keep errors explicit |

## First Implementation Slice

Start with M0 only. Do not add normal oneCCL backend selection until the mixed compiler and event experiment passes.

The first slice is complete when this sequence works on two nodes:

```text
DPC++ creates an in-order CUDA-backed SYCL queue
  -> native cudaStream_t is extracted
  -> input is copied into NVSHMEM symmetric memory
  -> NVSHMEM allreduce is enqueued on that stream
  -> output is copied to ordinary SYCL device USM
  -> a SYCL event reports completion
  -> result is verified
  -> NVSHMEM finalizes cleanly
```

After M0 passes, implement M1 and M2 together as the first reviewable backend pull request. Keep M3 and M4 as a second pull request so event semantics and collective behavior can be reviewed independently from build and lifecycle wiring.

## Definition Of Done

The NVSHMEM backend is complete when:

- It is optional and does not regress non-NVSHMEM builds.
- Existing oneCCL API calls require no source changes.
- Arbitrary supported oneCCL buffers work through transparent staging.
- Dependencies and returned events compose with SYCL work.
- All intended public collectives are implemented or use a correct documented fallback.
- Communicator and runtime lifecycles do not leak, hang, or double-finalize.
- Unsupported configurations fail before partially enqueueing communication.
- Multi-node correctness and performance results are reproducible.
- Build, runtime, environment, and support limitations are documented.

## References

- [oneAPI oneCCL specification](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/latest/elements/oneccl/source/)
- [NVSHMEM documentation](https://docs.nvidia.com/nvshmem/api/index.html)
- [NVSHMEM collective communication](https://docs.nvidia.com/nvshmem/api/gen/api/collectives.html)
- [NVSHMEM memory management](https://docs.nvidia.com/nvshmem/api/gen/api/memory.html)
- [NVSHMEM team management](https://docs.nvidia.com/nvshmem/api/gen/api/teams.html)
- [NVSHMEM and the CUDA model](https://docs.nvidia.com/nvshmem/api/cuda-interactions.html)
- [OSHMPI](https://github.com/pmodels/oshmpi)
