/*
 * Copyright 2026 UXL Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "nvshmem_uid_adapter.h"

#include <mpi.h>

#include <cstdlib>
#include <iostream>
#include <vector>

namespace {
[[noreturn]] void abort_job(int rank, const char* operation) {
    std::cerr << "rank " << rank << ": " << operation << ": "
              << oneccl_nvshmem_uid_last_error() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}
} // namespace

int main(int argc, char** argv) {
    if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
        std::cerr << "MPI initialization failed" << std::endl;
        return EXIT_FAILURE;
    }

    int rank = -1;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Comm node_comm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
    int node_rank = -1;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_free(&node_comm);

    int device_count = 0;
    if (oneccl_nvshmem_uid_cuda_device_count(&device_count) != 0) {
        abort_job(rank, "CUDA device discovery failed");
    }
    const int device = (device_count == 1) ? 0 : node_rank;
    if (device >= device_count) {
        std::cerr << "rank " << rank << ": node rank " << node_rank
                  << " exceeds visible CUDA device count " << device_count << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        std::abort();
    }
    if (oneccl_nvshmem_uid_cuda_set_device(device) != 0) {
        abort_job(rank, "CUDA device selection failed");
    }

    std::vector<unsigned char> uid(oneccl_nvshmem_uid_size());
    int uid_status = 0;
    if (rank == 0) {
        uid_status = oneccl_nvshmem_uid_get(uid.data(), uid.size());
    }
    MPI_Bcast(&uid_status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (uid_status != 0) {
        abort_job(rank, "NVSHMEM UID creation failed");
    }
    MPI_Bcast(uid.data(), static_cast<int>(uid.size()), MPI_BYTE, 0, MPI_COMM_WORLD);

    if (oneccl_nvshmem_uid_init(rank, size, uid.data(), uid.size()) != 0) {
        abort_job(rank, "NVSHMEM UID initialization failed");
    }
    if (!oneccl_nvshmem_uid_is_initialized() || oneccl_nvshmem_uid_my_pe() != rank ||
        oneccl_nvshmem_uid_n_pes() != size) {
        std::cerr << "rank " << rank << ": NVSHMEM topology or initialization status mismatch"
                  << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        std::abort();
    }

    constexpr size_t arena_size = 4096;
    void* arena = oneccl_nvshmem_uid_malloc(arena_size);
    if (arena == nullptr) {
        abort_job(rank, "NVSHMEM symmetric allocation failed");
    }
    oneccl_nvshmem_uid_barrier_all();
    oneccl_nvshmem_uid_free(arena);
    oneccl_nvshmem_uid_finalize();

    if (rank == 0) {
        std::cout << "PASSED: NVSHMEM UID host-library bootstrap" << std::endl;
    }
    MPI_Finalize();
    return EXIT_SUCCESS;
}
