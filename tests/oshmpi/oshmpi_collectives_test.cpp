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

#include <mpi.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "oneapi/ccl.hpp"

namespace {

int failures = 0;

void expect(bool condition, int rank, const char* message) {
    if (!condition) {
        std::cerr << "Rank " << rank << ": " << message << std::endl;
        ++failures;
    }
}

} // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided) != MPI_SUCCESS ||
        provided < MPI_THREAD_SERIALIZED) {
        std::cerr << "MPI_THREAD_SERIALIZED is required" << std::endl;
        return 1;
    }

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const char* backend = std::getenv("CCL_BACKEND");
    if (!backend || std::string(backend) != "oshmpi") {
        if (rank == 0) {
            std::cerr << "Set CCL_BACKEND=oshmpi" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    ccl::init();

    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type address{};
    if (rank == 0) {
        kvs = ccl::create_main_kvs();
        address = kvs->get_address();
    }
    MPI_Bcast(address.data(), static_cast<int>(address.size()), MPI_BYTE, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        kvs = ccl::create_kvs(address);
    }

    {
        auto comm = ccl::create_communicator(size, rank, kvs);

        auto barrier_event = ccl::barrier(comm);
        expect(barrier_event.test(), rank, "blocking barrier event is incomplete");

        const std::size_t count = 33;
        std::vector<int> gather_send(count, rank + 1);
        std::vector<int> gather_recv(count * static_cast<std::size_t>(size), -1);
        ccl::allgather(gather_send.data(), gather_recv.data(), count, comm).wait();
        for (int peer = 0; peer < size; ++peer) {
            for (std::size_t index = 0; index < count; ++index) {
                expect(gather_recv[static_cast<std::size_t>(peer) * count + index] == peer + 1,
                       rank,
                       "allgather result mismatch");
            }
        }

        std::vector<std::vector<int>> gather_parts(
            static_cast<std::size_t>(size), std::vector<int>(count, -1));
        ccl::vector_class<int*> gather_buffers;
        for (auto& part : gather_parts) {
            gather_buffers.push_back(part.data());
        }
        ccl::allgather(gather_send.data(), gather_buffers, count, comm).wait();
        for (int peer = 0; peer < size; ++peer) {
            expect(gather_parts[static_cast<std::size_t>(peer)][0] == peer + 1,
                   rank,
                   "vector allgather result mismatch");
        }

        std::vector<float> reduce_send(count, static_cast<float>(rank + 1));
        std::vector<float> reduce_recv(count, 0.0f);
        ccl::allreduce(
            reduce_send.data(), reduce_recv.data(), count, ccl::reduction::sum, comm)
            .wait();
        const float expected_sum = static_cast<float>(size * (size + 1) / 2);
        for (const float value : reduce_recv) {
            expect(value == expected_sum, rank, "allreduce result mismatch");
        }

        std::vector<int> alltoall_send(static_cast<std::size_t>(size) * count);
        std::vector<int> alltoall_recv(static_cast<std::size_t>(size) * count, -1);
        for (int destination = 0; destination < size; ++destination) {
            for (std::size_t index = 0; index < count; ++index) {
                alltoall_send[static_cast<std::size_t>(destination) * count + index] =
                    rank * 1000 + destination;
            }
        }
        ccl::alltoall(alltoall_send.data(), alltoall_recv.data(), count, comm).wait();
        for (int source = 0; source < size; ++source) {
            expect(alltoall_recv[static_cast<std::size_t>(source) * count] ==
                       source * 1000 + rank,
                   rank,
                   "alltoall result mismatch");
        }

        std::vector<int> broadcast_buffer(count, rank == 0 ? 42 : -1);
        ccl::broadcast(broadcast_buffer.data(), count, 0, comm).wait();
        for (const int value : broadcast_buffer) {
            expect(value == 42, rank, "broadcast result mismatch");
        }

        bool unsupported_threw = false;
        try {
            std::vector<std::size_t> counts(static_cast<std::size_t>(size), count);
            ccl::allgatherv(
                gather_send.data(), count, gather_recv.data(), counts, comm)
                .wait();
        }
        catch (const ccl::exception&) {
            unsupported_threw = true;
        }
        expect(unsupported_threw, rank, "unsupported allgatherv did not throw");
    }

    int mpi_finalized = 0;
    MPI_Finalized(&mpi_finalized);
    expect(mpi_finalized == 0, rank, "OSHMPI finalized externally owned MPI");
    if (mpi_finalized != 0) {
        return 1;
    }
    MPI_Barrier(MPI_COMM_WORLD);

    int global_failures = 0;
    MPI_Allreduce(&failures, &global_failures, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
        std::cout << (global_failures == 0 ? "PASS" : "FAIL") << std::endl;
    }

    MPI_Finalize();
    return global_failures == 0 ? 0 : 1;
}
