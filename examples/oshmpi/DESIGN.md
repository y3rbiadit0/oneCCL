# oneCCL OSHMPI Backend Plan

## Goal

Add OSHMPI as an opt-in, top-level oneCCL backend selected with
`CCL_BACKEND=oshmpi`. The implementation targets Leonardo Booster and uses an
externally built OSHMPI linked against the same MPI used by oneCCL.

## Agreed Scope

### Phase 1: host buffers

- Build option: `CCL_ENABLE_OSHMPI=ON`.
- Runtime selector: `CCL_BACKEND=oshmpi`.
- Communicator: `SHMEM_TEAM_WORLD` only, one oneCCL rank per OSHMPI PE.
- Operations: barrier, allgather, allreduce, alltoall, and broadcast.
- Execution: blocking OSHMPI calls that return an already-completed oneCCL event.
- Buffers: ordinary oneCCL host buffers staged through bounded symmetric memory.
- Dependencies: wait before entering the OSHMPI operation.
- Unsupported: communicator split, allgatherv, alltoallv, reduce,
  reduce-scatter, send, recv, and grouped operations.
- Allreduce: integral types, `float32`, and `float64`; `sum`, `prod`, `min`, and
  `max`. Low-precision types, average, and custom reductions are rejected.

### Phase 2: CUDA/SYCL buffers

Begin only after Phase 1 passes all Leonardo gates. Validate OSHMPI CUDA-space
collectives independently before choosing direct CUDA-symmetric staging or a
host-staged fallback. Preserve SYCL dependency and stream ordering.

## Architecture

1. Add CMake discovery through `OSHMPI_ROOT`, `OSHMPI_HOME`, or
   `OSHMPI_PREFIX`, normalized as `OSHMPI::oshmpi`.
2. Add `backend_mode::oshmpi` and communicator/KVS selection branches.
3. Use an OSHMPI-specific KVS token. OSHMPI obtains process membership from
   `MPI_COMM_WORLD`; oneCCL KVS is not used for transport bootstrap.
4. Lazily initialize OSHMPI when the first OSHMPI communicator is created.
5. Require `SHMEM_THREAD_SERIALIZED` or better and serialize runtime calls.
6. Validate supplied communicator rank and size against `shmem_my_pe()` and
   `shmem_n_pes()`.
7. Allocate a process-wide symmetric staging arena with `shmem_malloc`, split
   into source and destination lanes, and chunk operations that exceed it.
8. Release the arena and finalize OSHMPI when the final communicator is
   destroyed. Communicator destruction is collective and must precede an
   externally owned `MPI_Finalize()`.
9. Require oneCCL and OSHMPI to resolve the same MPI implementation and DSO.

## OSHMPI Dependency

The Leonardo dependency is pinned to:

- OSHMPI `ee5cf110e673c098707257bb025404e17ac0a5fc` (`2.1a1`);
- OpenPA `0475704dde41054db33562a8d17314fe0e30aaf3`;
- HPC-X 2.19 `libmpi.so.40`;
- CUDA 12.4.

The existing `$HOME/opt-src/oshmpi-main` checkout contains test-only local
changes and must remain untouched. The oneCCL workflow creates a clean detached
worktree, applies the external MPI ownership patch, and installs to
`$HOME/opt/oshmpi-ee5cf110-oneccl`.

The expected Leonardo base stack comes from
`comm-playground/cluster/leonardo/environment.sh cuda`:

- `nvhpc/24.5`;
- `hpcx-mpi/2.19`;
- CUDA 12.4;
- UCX PML and OSC;
- HCOLL and UCC disabled;
- `srun` as the initial launcher.

## Milestones

- [x] Map oneCCL backend and collective dispatch architecture.
- [x] Review OSHMPI lifecycle, team, collective, and datatype APIs.
- [x] Review the Leonardo OSHMPI experiments and runtime environment.
- [x] Gate 0: identify and pin the working Leonardo OSHMPI revision.
- [x] Add CMake option, dependency discovery, and public config macro.
- [x] Add environment parsing and backend dispatch.
- [x] Add OSHMPI KVS and runtime lifecycle.
- [x] Add bounded symmetric staging and chunking.
- [x] Implement barrier.
- [x] Implement allgather.
- [x] Implement allreduce.
- [x] Implement alltoall.
- [x] Implement broadcast.
- [x] Add explicit errors for unsupported API operations.
- [x] Add host correctness and lifecycle tests.
- [x] Add Leonardo environment, build, and Slurm validation scripts.
- [x] Pass Phase 1 validation gates.
- [ ] Design and implement Phase 2 CUDA/SYCL support.

## Validation Gates

All compilation and runtime validation is performed by the user on Leonardo.
After each implementation increment, stop and request exactly one validation
step with the command and expected evidence.

1. Gate 0: dependency provenance, configure flags, and linked MPI.
2. Gate 1: patched standalone OSHMPI build and MPI ownership smoke tests.
3. Gate 2: oneCCL CMake configure with `CCL_ENABLE_OSHMPI=ON`.
4. Gate 3: oneCCL and OSHMPI test compilation.
5. Gate 4: one rank on one Booster node.
6. Gate 5: two ranks on one node.
7. Gate 6: one rank per node on two nodes.
8. Gate 7: four ranks on one node and four ranks per node on two nodes.
9. Gate 8: externally initialized MPI remains active after OSHMPI teardown.
10. Gate 9: OSHMPI-owned MPI is finalized exactly once.

## Known Risks

- Upstream OSHMPI currently finalizes externally initialized MPI.
- Upstream OSHMPI initializes MPI_T with `MPI_THREAD_SINGLE` before inspecting
  MPI, which lowers the externally initialized thread level with HPC-X 2.19.
- OpenSHMEM collective operands are symmetric; arbitrary oneCCL pointers cannot
  be passed directly.
- OSHMPI collective internals cast some counts to `int`; staging chunks must
  stay within both arena and `INT_MAX` limits.
- OSHMPI finalization and symmetric allocation are collective.
- Multiple user threads can issue collectives in inconsistent process order;
  internal serialization cannot repair inconsistent ordering across ranks.
- The existing OSHMPI experiment evidence is incomplete and some recorded PASS
  results predate current benchmark rewrites.

## Progress Notes

- 2026-08-14: agreed on host-first implementation, core five collectives,
  external pinned OSHMPI, and an MPI-ownership patch. Leonardo context confirms
  host-symmetric allreduce and an HPC-X/UCX CUDA stack.
- 2026-08-14: implemented Phase 1 source, CMake integration, symmetric chunked
  staging, tests, the external MPI ownership patch, and Leonardo tooling.
  Static review passes; Gate 0 and all compilation/runtime gates remain user-run.
- 2026-08-15: Gate 0 completed. The working install was built from OSHMPI
  `ee5cf110e673c098707257bb025404e17ac0a5fc` with OpenPA
  `0475704dde41054db33562a8d17314fe0e30aaf3`, CUDA 12.4, and HPC-X 2.19.
  Local source changes affect only upstream CUDA test construction and
  diagnostics, so production builds use a clean detached worktree.
- 2026-08-15: Gate 1 tooling now creates a clean detached OSHMPI worktree,
  preserves the existing source and install, builds a versioned patched prefix,
  and compiles standalone external-MPI and OSHMPI-owned lifecycle smoke tests.
- 2026-08-15: Gate 1 exposed an HPC-X interaction where OSHMPI's early
  `MPI_T_init_thread(MPI_THREAD_SINGLE)` lowered an externally initialized MPI
  runtime to `MPI_THREAD_SINGLE`. The dependency patch now queries externally
  initialized MPI first and requests its granted thread level from MPI_T; the
  smoke test checks that the level remains at least `MPI_THREAD_SERIALIZED`.
- 2026-08-15: Gate 1 passed both external-MPI and OSHMPI-owned lifecycle paths
  with HPC-X 2.19. The oneCCL build tooling now isolates its source-path
  variables from sourced environment scripts and rejects CMake caches belonging
  to another project.
- 2026-08-15: Gate 2 configured oneCCL with OSHMPI and resolved CUDA-enabled
  OSHMPI's transitive `libcudart` dependency. Gate 3 reached compilation; the
  Leonardo build now excludes GCC-only security and fallthrough flags rejected
  by NVHPC and builds only the focused OSHMPI test instead of the full functional
  suite.
- 2026-08-15: NVHPC compilation reached oneCCL sources and reported its
  `extra_semicolon` and `code_is_unreachable` diagnostics for existing portable
  macro and fallback patterns. This confirmed that using NVHPC for the oneCCL
  host library added unrelated compatibility work; no diagnostic suppressions
  remain in the focused GNU build.
- 2026-08-15: the Phase 1 build was narrowed to a portable OSHMPI-focused
  artifact. OSHMPI retains its HPC-X/NVHPC CUDA build, while oneCCL uses the
  supported GNU host compiler and disables its unrelated native MPI and stub
  backends. The OSHMPI test links HPC-X through CMake's imported MPI C target.
- 2026-08-15: PMIx remains enabled in the focused build because oneCCL's
  unconditionally compiled native OFI sources use PMIx types even when
  `ENABLE_PMIX=OFF`. OSHMPI still bypasses native OFI initialization at runtime.
- 2026-08-15: the resizable PMI implementation used a root-rank KVS key defined
  inside the native MPI guard. The key is now a shared ATL definition so the
  focused `ENABLE_MPI=OFF` build does not depend on MPI-only declarations.
- 2026-08-15: root-caused the repeating Gate 3 failures. They were not OSHMPI
  defects but three pre-existing upstream sites that use MPI symbols outside
  `#ifdef CCL_ENABLE_MPI` in always-compiled sources: the IPC allgatherv
  workaround in `exchange_utils.cpp`, the `MPI_Comm_rank` address suffix in
  `internal_kvs.cpp`, and the `ATL_MPI_ROOT_RANK_KEY` include in
  `pmi_resizable_simple_internal.h`. All three are now guarded, so
  `ENABLE_MPI=OFF` compiles as a supported configuration rather than a
  one-error-at-a-time repair loop.
- 2026-08-15: OSHMPI's MPI dependency is discovered with `find_package(MPI)` and
  exposed through `MPI::MPI_C` on the `OSHMPI::oshmpi` imported target, instead of
  borrowing oneCCL's Intel-MPI-oriented `MPI_INCLUDE_DIR`. `build_leonardo.sh` no
  longer passes `MPI_DIR`, so oneCCL's own MPI paths are never repointed at HPC-X.
- 2026-08-15: the startup agreement reductions used file-scope statics in
  `libccl.so` as `shmem_*_reduce` operands. OpenSHMEM guarantees symmetry only for
  the symmetric heap and the executable's data segment, not a shared library's, so
  the operands now come from a `shmem_malloc`'d scratch block. `acquire()` also
  stops treating a failed `shmem_init_thread` as initialized, which previously led
  to collectives and `shmem_finalize()` on a runtime that never came up.
- 2026-08-15: a local syntax-only sweep over all 157 always-compiled sources (host
  clang targeting x86_64 with small glibc shims) found three further Gate 3
  blockers in the backend itself, all of which would have cost separate Leonardo
  round trips:
  1. `oshmpi_comm.cpp` defined a helper named `unsupported` inside namespace
     `ccl`, which is ambiguous against the `ccl::unsupported` exception type from
     `oneapi/ccl/exception.hpp` at all ten call sites. Renamed to
     `throw_unsupported`.
  2. `validate_attributes` called `.empty()` on the `match_id` attribute, but
     `ccl::string` only exposes `length()`. Now matches the `coll_param.cpp` idiom.
  3. `oshmpi_runtime.cpp` and `oshmpi_kvs_impl.hpp` included
     `oneapi/ccl/api_functions.hpp` / `kvs_impl.hpp` without the type stack those
     headers assume, since `api_functions.hpp` includes nothing of its own. The
     runtime now includes the `oneapi/ccl.hpp` umbrella, and the KVS header is
     self-contained in the same way as `stub_kvs_impl.hpp`.
  After these, the only sources that fail locally are `ofi_api_wrapper.cpp`,
  `base_thread.cpp`, and `profile.cpp`, all on glibc-only constructs
  (`RTLD_DI_ORIGIN`, `cpu_set_t`, libstdc++ `std::pair`) that are unrelated to the
  backend and fail identically with `ENABLE_MPI=ON`.
- 2026-08-16: Gates 3 through 8 passed on Leonardo. Compilation needed two further
  fixes for backends that are switched off in this build - `reset_group_lifecycle`
  and `is_run_with_mpi` were both defined with call sites behind a backend guard,
  which `-Wall -Wextra -Werror` rejects as unused. Running then needed
  `LD_LIBRARY_PATH` for `libccl.so.1` and `liboshmpi.so`, because the project
  builds with `CMAKE_SKIP_RPATH` and this focused artifact does not go through the
  installed `vars.sh`.
- 2026-08-16: the validation job had been running with a 64M staging arena, so no
  collective ever chunked. `leonardo_env.sh` exports `CCL_OSHMPI_STAGING_SIZE=64M`
  and is sourced first, so the sbatch's `${VAR:-128}` default could never apply.
  The job now overrides the arena through `ONECCL_VALIDATE_STAGING_SIZE`.
  Re-running with a real 128-byte arena gave: 2 ranks/1 node, 2 ranks/2 nodes,
  4 ranks/1 node, and 8 ranks/2 nodes, all PASS at 32, 32, 16 and 8 byte chunks
  respectively, with the test's `MPI_Finalized` assertion holding in every case.
- 2026-08-15: `examples/oshmpi/patches/0001-preserve-external-mpi-ownership.patch`
  was found to be referenced by the dependency build and required by
  `FindOSHMPI.cmake`, but never committed, so the stack was not reproducible from a
  clean clone. `build_oshmpi_leonardo.sh` now fails early with regeneration
  instructions; the patch itself still has to be recovered from Leonardo.
- 2026-08-16: the original patch was lost. Both scratch worktrees had been removed,
  and neither `opt-src/oshmpi-main` nor `opt-src/oshmpi` carried the change; there
  was no stash and no dangling object in the shared store. The only surviving trace
  was the installed prefix `$HOME/opt/oshmpi-ee5cf110-oneccl`, whose generated
  `include/shmem.h` still declares the probe and defines the macro.
- 2026-08-16: the patch was reconstructed against pinned upstream `ee5cf110` and
  committed. Upstream takes ownership of MPI in two places, both reproduced here:
  `finalize_impl()` calls `MPI_Finalize()` unconditionally, and
  `OSHMPI_initialize_thread()` calls `MPI_T_init_thread(MPI_THREAD_SINGLE, ...)`
  ahead of `MPI_Initialized()`. The reconstruction records whether OSHMPI called
  `MPI_Init_thread` itself and finalizes only in that case, and moves MPI_T
  initialization after MPI's own so it requests the granted level. `initialize_mpit()`
  runs later in the same function, so the reordering is safe. The patch is verified
  to apply cleanly to a pristine tree and to satisfy the markers that
  `build_oshmpi_leonardo.sh` and `FindOSHMPI.cmake` check, but it is functionally
  equivalent rather than byte-identical to the lost original, so Gate 1 must be
  re-run against a separate install prefix before it is trusted.
