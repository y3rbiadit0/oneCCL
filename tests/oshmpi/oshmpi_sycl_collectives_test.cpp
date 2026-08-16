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
 * Exercises the parts of the OSHMPI backend that only exist under SYCL, none of
 * which the host or device collectives tests touch:
 *
 *   - a device communicator built from a SYCL device and context, rather than
 *     the host create_communicator(size, rank, kvs) overload
 *   - a ccl::stream wrapping a sycl::queue, which the backend has to drain
 *     before staging: buffers are filled by a kernel here, not by memcpy, so a
 *     missing drain shows up as wrong data rather than as a crash
 *   - group_start/group_end around collectives
 *   - send/recv, implemented over shmem_putmem_signal
 *
 * Buffers are SYCL USM device allocations, matching how comm-playground drives
 * oneCCL. Values and counts mirror oshmpi_collectives_test so a failure here
 * that the host test does not reproduce points at the SYCL path specifically.
 */

#include <mpi.h>

#include <sycl/sycl.hpp>

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

int local_rank_from_env(int fallback) {
    for (const char* name : { "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK" }) {
        const char* value = std::getenv(name);
        if (value && *value) {
            return std::atoi(value);
        }
    }
    return fallback;
}

sycl::device device_for_rank(int rank) {
    const auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devices.empty()) {
        throw std::runtime_error("no SYCL GPU devices available");
    }
    return devices[static_cast<std::size_t>(local_rank_from_env(rank)) % devices.size()];
}

/* Fills a device buffer from a kernel rather than a host copy. This is the point
 * of the test: the fill is queued on the caller's stream, so a backend that does
 * not drain that stream before staging will read the buffer too early. */
template <class T>
void fill_on_device(sycl::queue& queue, T* buffer, std::size_t count, T value) {
    queue.parallel_for(sycl::range<1>(count), [=](sycl::id<1> index) {
        buffer[index] = value;
    });
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

    try {
        ccl::init();

        sycl::device device = device_for_rank(rank);
        sycl::context context(device);
        sycl::queue queue(context, device, sycl::property::queue::in_order());

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

        // The device overload: rejected by the backend before this work.
        auto ccl_device = ccl::create_device(queue.get_device());
        auto ccl_context = ccl::create_context(queue.get_context());
        auto comm = ccl::create_communicator(size, rank, ccl_device, ccl_context, kvs);
        auto stream = ccl::create_stream(queue);

        const std::size_t count = 33;
        const std::size_t total = count * static_cast<std::size_t>(size);

        // allreduce, with the send buffer produced by a kernel on the stream
        {
            float* send = sycl::malloc_device<float>(count, queue);
            float* recv = sycl::malloc_device<float>(count, queue);
            fill_on_device(queue, send, count, static_cast<float>(rank + 1));

            ccl::allreduce(send, recv, count, ccl::datatype::float32, ccl::reduction::sum,
                           comm, stream)
                .wait();

            std::vector<float> host(count, 0.0f);
            queue.memcpy(host.data(), recv, count * sizeof(float)).wait();
            const float expected = static_cast<float>(size * (size + 1) / 2);
            for (const float value : host) {
                expect(value == expected, rank, "sycl allreduce result mismatch");
            }
            sycl::free(send, queue);
            sycl::free(recv, queue);
        }

        // allgather inside a group scope
        {
            int* send = sycl::malloc_device<int>(count, queue);
            int* recv = sycl::malloc_device<int>(total, queue);
            fill_on_device(queue, send, count, rank + 1);

            ccl::group_start();
            ccl::allgather(send, recv, count, ccl::datatype::int32, comm, stream).wait();
            ccl::group_end();

            std::vector<int> host(total, -1);
            queue.memcpy(host.data(), recv, total * sizeof(int)).wait();
            for (int peer = 0; peer < size; ++peer) {
                expect(host[static_cast<std::size_t>(peer) * count] == peer + 1,
                       rank,
                       "sycl grouped allgather result mismatch");
            }
            sycl::free(send, queue);
            sycl::free(recv, queue);
        }

        // broadcast from rank 0
        {
            int* buffer = sycl::malloc_device<int>(count, queue);
            fill_on_device(queue, buffer, count, rank == 0 ? 42 : -1);

            ccl::broadcast(buffer, count, ccl::datatype::int32, 0, comm, stream).wait();

            std::vector<int> host(count, 0);
            queue.memcpy(host.data(), buffer, count * sizeof(int)).wait();
            for (const int value : host) {
                expect(value == 42, rank, "sycl broadcast result mismatch");
            }
            sycl::free(buffer, queue);
        }

        // send/recv ping-pong between rank 0 and rank 1
        if (size >= 2 && (rank == 0 || rank == 1)) {
            const int peer = (rank == 0) ? 1 : 0;
            int* send = sycl::malloc_device<int>(count, queue);
            int* recv = sycl::malloc_device<int>(count, queue);
            fill_on_device(queue, send, count, rank + 100);
            fill_on_device(queue, recv, count, -1);

            if (rank == 0) {
                ccl::send(send, count, ccl::datatype::int32, peer, comm, stream).wait();
                ccl::recv(recv, count, ccl::datatype::int32, peer, comm, stream).wait();
            }
            else {
                ccl::recv(recv, count, ccl::datatype::int32, peer, comm, stream).wait();
                ccl::send(send, count, ccl::datatype::int32, peer, comm, stream).wait();
            }

            std::vector<int> host(count, 0);
            queue.memcpy(host.data(), recv, count * sizeof(int)).wait();
            for (const int value : host) {
                expect(value == peer + 100, rank, "sycl send/recv result mismatch");
            }
            sycl::free(send, queue);
            sycl::free(recv, queue);
        }
    }
    catch (const std::exception& error) {
        std::cerr << "Rank " << rank << ": exception: " << error.what() << std::endl;
        ++failures;
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
