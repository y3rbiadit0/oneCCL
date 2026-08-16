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
 * Device-buffer counterpart of oshmpi_collectives_test.
 *
 * Deliberately the same collectives, counts and expected values as the host
 * test, so a failure here that the host test does not reproduce points squarely
 * at the device staging path rather than at the collective itself.
 *
 * Buffers are plain cudaMalloc allocations passed straight to the ordinary
 * oneCCL API. There is no stream: the backend classifies pointers with
 * cudaPointerGetAttributes and stages device operands through its host symmetric
 * arena. Requires a oneCCL built with CCL_ENABLE_OSHMPI_CUDA=ON.
 */

#include <mpi.h>

#include <cuda_runtime.h>

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

bool cuda_ok(cudaError_t status, const char* what, int rank) {
    if (status != cudaSuccess) {
        std::cerr << "Rank " << rank << ": " << what << ": " << cudaGetErrorString(status)
                  << std::endl;
        ++failures;
        return false;
    }
    return true;
}

int local_rank_from_env(int fallback) {
    for (const char* name : { "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK" }) {
        const char* value = std::getenv(name);
        if (value && *value) {
            return std::atoi(value);
        }
    }
    return fallback;
}

// Small owning wrapper so every early return still frees the allocation.
template <class T>
class device_buffer {
public:
    device_buffer() = default;
    device_buffer(const device_buffer&) = delete;
    device_buffer& operator=(const device_buffer&) = delete;

    ~device_buffer() {
        if (pointer) {
            cudaFree(pointer);
        }
    }

    bool allocate(std::size_t count, int rank) {
        bytes = count * sizeof(T);
        return cuda_ok(cudaMalloc(&pointer, bytes), "cudaMalloc", rank);
    }

    bool fill_from(const std::vector<T>& host, int rank) {
        return cuda_ok(cudaMemcpy(pointer, host.data(), bytes, cudaMemcpyHostToDevice),
                       "cudaMemcpy H2D",
                       rank);
    }

    bool read_into(std::vector<T>& host, int rank) {
        return cuda_ok(cudaMemcpy(host.data(), pointer, bytes, cudaMemcpyDeviceToHost),
                       "cudaMemcpy D2H",
                       rank);
    }

    T* get() {
        return pointer;
    }

private:
    T* pointer = nullptr;
    std::size_t bytes = 0;
};

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

    int device_count = 0;
    if (!cuda_ok(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount", rank) ||
        device_count == 0) {
        std::cerr << "Rank " << rank << ": no CUDA devices visible" << std::endl;
        MPI_Finalize();
        return 1;
    }
    if (!cuda_ok(cudaSetDevice(local_rank_from_env(rank) % device_count), "cudaSetDevice", rank)) {
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

        const std::size_t count = 33;
        const std::size_t total = count * static_cast<std::size_t>(size);

        // allgather: every PE contributes rank + 1
        {
            device_buffer<int> send;
            device_buffer<int> recv;
            if (send.allocate(count, rank) && recv.allocate(total, rank)) {
                std::vector<int> host_send(count, rank + 1);
                std::vector<int> host_recv(total, -1);
                send.fill_from(host_send, rank);
                recv.fill_from(host_recv, rank);

                ccl::allgather(send.get(), recv.get(), count, comm).wait();

                recv.read_into(host_recv, rank);
                for (int peer = 0; peer < size; ++peer) {
                    expect(host_recv[static_cast<std::size_t>(peer) * count] == peer + 1,
                           rank,
                           "device allgather result mismatch");
                }
            }
        }

        // allreduce: sum of 1..size
        {
            device_buffer<float> send;
            device_buffer<float> recv;
            if (send.allocate(count, rank) && recv.allocate(count, rank)) {
                std::vector<float> host_send(count, static_cast<float>(rank + 1));
                std::vector<float> host_recv(count, 0.0f);
                send.fill_from(host_send, rank);
                recv.fill_from(host_recv, rank);

                ccl::allreduce(send.get(), recv.get(), count, ccl::reduction::sum, comm).wait();

                recv.read_into(host_recv, rank);
                const float expected = static_cast<float>(size * (size + 1) / 2);
                for (const float value : host_recv) {
                    expect(value == expected, rank, "device allreduce result mismatch");
                }
            }
        }

        // alltoall: PE r sends r * 1000 + destination to each destination
        {
            device_buffer<int> send;
            device_buffer<int> recv;
            if (send.allocate(total, rank) && recv.allocate(total, rank)) {
                std::vector<int> host_send(total, 0);
                std::vector<int> host_recv(total, -1);
                for (int destination = 0; destination < size; ++destination) {
                    for (std::size_t index = 0; index < count; ++index) {
                        host_send[static_cast<std::size_t>(destination) * count + index] =
                            rank * 1000 + destination;
                    }
                }
                send.fill_from(host_send, rank);
                recv.fill_from(host_recv, rank);

                ccl::alltoall(send.get(), recv.get(), count, comm).wait();

                recv.read_into(host_recv, rank);
                for (int source = 0; source < size; ++source) {
                    expect(host_recv[static_cast<std::size_t>(source) * count] ==
                               source * 1000 + rank,
                           rank,
                           "device alltoall result mismatch");
                }
            }
        }

        // broadcast: root 0 holds 42
        {
            device_buffer<int> buffer;
            if (buffer.allocate(count, rank)) {
                std::vector<int> host(count, rank == 0 ? 42 : -1);
                buffer.fill_from(host, rank);

                ccl::broadcast(buffer.get(), count, 0, comm).wait();

                buffer.read_into(host, rank);
                for (const int value : host) {
                    expect(value == 42, rank, "device broadcast result mismatch");
                }
            }
        }

        // mixed host/device operands must still work: the backend classifies each
        // buffer independently rather than assuming both sides match.
        {
            device_buffer<int> send;
            if (send.allocate(count, rank)) {
                std::vector<int> host_send(count, rank + 1);
                std::vector<int> host_recv(total, -1);
                send.fill_from(host_send, rank);

                ccl::allgather(send.get(), host_recv.data(), count, comm).wait();

                for (int peer = 0; peer < size; ++peer) {
                    expect(host_recv[static_cast<std::size_t>(peer) * count] == peer + 1,
                           rank,
                           "mixed device-send host-recv allgather mismatch");
                }
            }
        }
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
