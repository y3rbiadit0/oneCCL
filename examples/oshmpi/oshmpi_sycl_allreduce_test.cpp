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

/* allreduce over the OSHMPI backend on SYCL device memory. The stream is
 * required: the backend drains the caller's queue before reading a device
 * buffer, and stages through it.
 *
 *   CCL_BACKEND=oshmpi mpirun -n 2 ./oshmpi_sycl_allreduce_test
 */

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <mpi.h>
#include <sycl/sycl.hpp>
#include <oneapi/ccl.hpp>

sycl::device get_device_for_rank(int rank) {
    for (const char* name : { "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK" }) {
        const char* value = std::getenv(name);
        if (value && *value) {
            rank = std::atoi(value);
            break;
        }
    }
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devices.empty()) {
        std::cerr << "No GPU devices found!" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return devices[static_cast<size_t>(rank) % devices.size()];
}

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

    /* create sycl queue */
    sycl::device dev = get_device_for_rank(rank);
    sycl::context ctx(dev);
    sycl::queue q(ctx, dev, sycl::property::queue::in_order());

    /* create kvs */
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

    /* create device communicator and stream */
    auto ccl_dev = ccl::create_device(q.get_device());
    auto ccl_ctx = ccl::create_context(q.get_context());
    auto comm = ccl::create_communicator(size, rank, ccl_dev, ccl_ctx, kvs);
    auto ccl_stream = ccl::create_stream(q);

    /* create buffers */
    const size_t count = 1024;
    float* d_send = sycl::malloc_device<float>(count, q);
    float* d_recv = sycl::malloc_device<float>(count, q);

    /* filled by a kernel: the collective must drain this queue before staging */
    const float contribution = static_cast<float>(rank + 1);
    q.parallel_for(sycl::range<1>(count), [=](sycl::id<1> i) {
        d_send[i] = contribution;
    });

    /* invoke allreduce */
    ccl::allreduce(d_send,
                   d_recv,
                   count,
                   ccl::datatype::float32,
                   ccl::reduction::sum,
                   comm,
                   ccl_stream)
        .wait();

    std::vector<float> recv_data(count, 0.0f);
    q.memcpy(recv_data.data(), d_recv, count * sizeof(float)).wait();

    /* check result: 1 + 2 + ... + size */
    const float expected = static_cast<float>(size * (size + 1) / 2);
    bool success = true;
    for (size_t i = 0; i < count; i++) {
        if (std::fabs(recv_data[i] - expected) > 1e-5f) {
            std::cerr << "Rank " << rank << ": ERROR at index " << i << ": got " << recv_data[i]
                      << ", expected " << expected << std::endl;
            success = false;
            break;
        }
    }

    sycl::free(d_send, q);
    sycl::free(d_recv, q);

    int local_ok = success ? 1 : 0;
    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << (global_ok ? "PASSED" : "FAILED") << std::endl;
    }

    MPI_Finalize();

    return global_ok ? 0 : 1;
}
