/*
 * Copyright 2026 Contributors
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

/* allgather over the OSHMPI backend, host buffers.
 *
 *   CCL_BACKEND=oshmpi mpirun -n 2 ./oshmpi_allgather_test
 */

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <mpi.h>
#include <oneapi/ccl.hpp>

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const char* backend = std::getenv("CCL_BACKEND");
    if (!backend || std::string(backend) != "oshmpi") {
        if (rank == 0) {
            std::cerr << "this example requires CCL_BACKEND=oshmpi" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    ccl::init();

    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type addr{};
    if (rank == 0) {
        kvs = ccl::create_main_kvs();
        addr = kvs->get_address();
    }
    MPI_Bcast(addr.data(), static_cast<int>(addr.size()), MPI_BYTE, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        kvs = ccl::create_kvs(addr);
    }

    auto comm = ccl::create_communicator(size, rank, kvs);

    /* recv[peer * count ..] == peer */
    const size_t count = 128;
    std::vector<int> send_data(count, rank);
    std::vector<int> recv_data(count * static_cast<size_t>(size), -1);

    ccl::allgather(send_data.data(), recv_data.data(), count, comm).wait();

    bool success = true;
    for (int peer = 0; peer < size && success; peer++) {
        for (size_t i = 0; i < count; i++) {
            const int value = recv_data[static_cast<size_t>(peer) * count + i];
            if (value != peer) {
                std::cerr << "Rank " << rank << ": ERROR at peer " << peer << " index " << i
                          << ": got " << value << ", expected " << peer << std::endl;
                success = false;
                break;
            }
        }
    }

    int local_ok = success ? 1 : 0;
    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << (global_ok ? "PASSED" : "FAILED") << std::endl;
    }

    MPI_Finalize();

    return global_ok ? 0 : 1;
}
