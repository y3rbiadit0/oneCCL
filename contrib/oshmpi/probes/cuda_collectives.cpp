/*
 Copyright 2026 Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

/*
 * Phase 2 gate: does OSHMPI support the five collectives oneCCL needs when the
 * operands live in CUDA device memory?
 *
 * OSHMPI's team collectives forward the caller's pointers straight to MPI
 * (fcollect -> MPI_Allgather, alltoall -> MPI_Alltoall, broadcast -> MPI_Bcast,
 * reduce -> MPI_Allreduce), with no memkind handling anywhere in the path. Two
 * consequences are worth measuring rather than assuming:
 *
 *   1. Whether device operands work at all reduces to whether the underlying MPI
 *      is CUDA-aware. Hence the `raw` mode, which passes plain cudaMalloc memory
 *      that is not symmetric at all. If that works, oneCCL could skip symmetric
 *      staging entirely -- at the cost of relying on an implementation detail
 *      rather than the OpenSHMEM specification.
 *   2. OSHMPI_broadcast_team() finishes with a host memcpy(dest, source) on the
 *      root PE. On device pointers that is expected to fault, independently of
 *      how CUDA-aware MPI is.
 *
 * One collective per process run, because OSHMPI aborts the job on error and a
 * crash in one would otherwise mask the rest.
 *
 * usage: cuda_collectives <space|raw> <barrier|fcollect|alltoall|broadcast|reduce>
 */

#include <mpi.h>
#include <shmem.h>

/* shmemx.h declares the space API with no extern "C" guard of its own, and the
 * <shmem.h> it includes at its top opens and closes its own guard before those
 * declarations are reached. From C++ they would therefore be name-mangled and
 * fail to link against the C library. shmem.h is already included above, so the
 * nested include below is a no-op and only shmemx's declarations are wrapped.
 * Anything in oneCCL that uses the space API will need the same treatment. */
extern "C" {
#include <shmemx.h>
}

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr std::size_t element_count = 32;

int fail(const char* message, int rank) {
    std::cerr << "FAIL rank " << rank << ": " << message << std::endl;
    return 1;
}

bool cuda_ok(cudaError_t status, const char* what, int rank) {
    if (status != cudaSuccess) {
        std::cerr << "FAIL rank " << rank << ": " << what << ": " << cudaGetErrorString(status)
                  << std::endl;
        return false;
    }
    return true;
}

int local_rank_from_env(int fallback) {
    for (const char* name : { "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID" }) {
        const char* value = std::getenv(name);
        if (value && *value) {
            return std::atoi(value);
        }
    }
    return fallback;
}

struct device_buffers {
    void* source = nullptr;
    void* destination = nullptr;
    shmemx_space_t space = nullptr;
    bool from_space = false;
};

bool allocate(device_buffers& buffers, bool use_space, std::size_t bytes, int rank) {
    if (use_space) {
        shmemx_space_config_t config;
        std::memset(&config, 0, sizeof(config));
        config.sheap_size = 1 << 20;
        config.num_contexts = 0;
        config.memkind = SHMEMX_MEM_CUDA;
        config.device_handle = nullptr;
        config.hints = 0;

        shmemx_space_create(config, &buffers.space);
        shmemx_space_attach(buffers.space);
        buffers.source = shmemx_space_malloc(buffers.space, bytes);
        buffers.destination = shmemx_space_malloc(buffers.space, bytes);
        buffers.from_space = true;
    }
    else {
        if (!cuda_ok(cudaMalloc(&buffers.source, bytes), "cudaMalloc source", rank) ||
            !cuda_ok(cudaMalloc(&buffers.destination, bytes), "cudaMalloc destination", rank)) {
            return false;
        }
    }
    return buffers.source != nullptr && buffers.destination != nullptr;
}

void release(device_buffers& buffers) {
    if (buffers.from_space) {
        // Space allocations are released with plain shmem_free; there is no
        // shmemx_space_free. Destroying a space that still holds live chunks
        // trips an assertion in OSHMPI's memory pool teardown.
        if (buffers.destination) {
            shmem_free(buffers.destination);
        }
        if (buffers.source) {
            shmem_free(buffers.source);
        }
        if (buffers.space) {
            shmemx_space_detach(buffers.space);
            shmemx_space_destroy(buffers.space);
        }
    }
    else {
        if (buffers.source) {
            cudaFree(buffers.source);
        }
        if (buffers.destination) {
            cudaFree(buffers.destination);
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: " << argv[0]
                  << " <space|raw> <barrier|fcollect|alltoall|broadcast|reduce>" << std::endl;
        return 2;
    }
    const std::string memory_mode = argv[1];
    const std::string collective = argv[2];
    if (memory_mode != "space" && memory_mode != "raw") {
        std::cerr << "unknown memory mode: " << memory_mode << std::endl;
        return 2;
    }

    // Mirror how oneCCL drives OSHMPI: the application owns MPI.
    int mpi_provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_provided) != MPI_SUCCESS ||
        mpi_provided < MPI_THREAD_SERIALIZED) {
        return fail("MPI_THREAD_SERIALIZED is unavailable", -1);
    }

    int world_rank = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int device_count = 0;
    if (!cuda_ok(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount", world_rank) ||
        device_count == 0) {
        return fail("no CUDA devices visible", world_rank);
    }
    const int device = local_rank_from_env(world_rank) % device_count;
    if (!cuda_ok(cudaSetDevice(device), "cudaSetDevice", world_rank)) {
        return fail("could not select a CUDA device", world_rank);
    }

    int shmem_provided = SHMEM_THREAD_SINGLE;
    if (shmem_init_thread(SHMEM_THREAD_SERIALIZED, &shmem_provided) != SHMEM_SUCCESS ||
        shmem_provided < SHMEM_THREAD_SERIALIZED) {
        return fail("SHMEM_THREAD_SERIALIZED is unavailable", world_rank);
    }

    const int rank = shmem_my_pe();
    const int size = shmem_n_pes();
    const std::size_t bytes_per_peer = element_count * sizeof(std::int32_t);
    // fcollect and alltoall need room for one contribution per PE.
    const std::size_t bytes = bytes_per_peer * static_cast<std::size_t>(size);

    if (collective == "barrier") {
        shmem_barrier_all();
        if (rank == 0) {
            std::cout << "PASS " << memory_mode << "-barrier" << std::endl;
        }
        shmem_finalize();
        MPI_Finalize();
        return 0;
    }

    device_buffers buffers;
    if (!allocate(buffers, memory_mode == "space", bytes, rank)) {
        return fail("device allocation failed", rank);
    }

    std::vector<std::int32_t> host_source(element_count * static_cast<std::size_t>(size), 0);
    std::vector<std::int32_t> host_destination(host_source.size(), -1);
    bool ok = true;

    if (collective == "fcollect") {
        for (std::size_t i = 0; i < element_count; ++i) {
            host_source[i] = rank + 1;
        }
    }
    else if (collective == "alltoall") {
        for (int peer = 0; peer < size; ++peer) {
            for (std::size_t i = 0; i < element_count; ++i) {
                host_source[static_cast<std::size_t>(peer) * element_count + i] =
                    rank * 1000 + peer;
            }
        }
    }
    else if (collective == "broadcast") {
        for (std::size_t i = 0; i < element_count; ++i) {
            host_source[i] = (rank == 0) ? 42 : -7;
        }
    }
    else if (collective == "reduce") {
        float* host_floats = reinterpret_cast<float*>(host_source.data());
        for (std::size_t i = 0; i < element_count; ++i) {
            host_floats[i] = static_cast<float>(rank + 1);
        }
    }
    else {
        release(buffers);
        return fail("unknown collective", rank);
    }

    if (!cuda_ok(cudaMemcpy(buffers.source, host_source.data(), bytes, cudaMemcpyHostToDevice),
                 "cudaMemcpy H2D",
                 rank)) {
        release(buffers);
        return 1;
    }
    cudaDeviceSynchronize();
    shmem_barrier_all();

    int status = SHMEM_SUCCESS;
    if (collective == "fcollect") {
        status = shmem_fcollectmem(
            SHMEM_TEAM_WORLD, buffers.destination, buffers.source, bytes_per_peer);
    }
    else if (collective == "alltoall") {
        status = shmem_alltoallmem(
            SHMEM_TEAM_WORLD, buffers.destination, buffers.source, bytes_per_peer);
    }
    else if (collective == "broadcast") {
        // Expected to fault on the root: OSHMPI_broadcast_team() ends with a host
        // memcpy(dest, source) when PE_root == my_pe.
        status = shmem_broadcastmem(
            SHMEM_TEAM_WORLD, buffers.destination, buffers.source, bytes_per_peer, 0);
    }
    else if (collective == "reduce") {
        status = shmem_float_sum_reduce(SHMEM_TEAM_WORLD,
                                        static_cast<float*>(buffers.destination),
                                        static_cast<const float*>(buffers.source),
                                        element_count);
    }

    if (status != SHMEM_SUCCESS) {
        release(buffers);
        return fail((collective + " returned a non-success status").c_str(), rank);
    }

    cudaDeviceSynchronize();
    if (!cuda_ok(
            cudaMemcpy(
                host_destination.data(), buffers.destination, bytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy D2H",
            rank)) {
        release(buffers);
        return 1;
    }

    if (collective == "fcollect") {
        for (int peer = 0; peer < size && ok; ++peer) {
            if (host_destination[static_cast<std::size_t>(peer) * element_count] != peer + 1) {
                ok = false;
            }
        }
    }
    else if (collective == "alltoall") {
        for (int peer = 0; peer < size && ok; ++peer) {
            if (host_destination[static_cast<std::size_t>(peer) * element_count] !=
                peer * 1000 + rank) {
                ok = false;
            }
        }
    }
    else if (collective == "broadcast") {
        for (std::size_t i = 0; i < element_count && ok; ++i) {
            if (host_destination[i] != 42) {
                ok = false;
            }
        }
    }
    else if (collective == "reduce") {
        const float expected = static_cast<float>(size * (size + 1) / 2);
        const float* results = reinterpret_cast<const float*>(host_destination.data());
        for (std::size_t i = 0; i < element_count && ok; ++i) {
            if (results[i] != expected) {
                ok = false;
            }
        }
    }

    release(buffers);

    int all_ok = 0;
    const int local_ok = ok ? 1 : 0;
    MPI_Allreduce(&local_ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    shmem_finalize();
    if (rank == 0) {
        std::cout << (all_ok ? "PASS " : "FAIL ") << memory_mode << "-" << collective << std::endl;
    }
    MPI_Finalize();
    return all_ok ? 0 : 1;
}
