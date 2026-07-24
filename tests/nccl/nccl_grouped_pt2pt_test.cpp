/*
 * Copyright 2016-2020 Intel Corporation
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

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include <mpi.h>
#include <oneapi/ccl.hpp>
#include <sycl/sycl.hpp>

class ccl_group_scope {
public:
    ccl_group_scope() {
        ccl::group_start();
        active = true;
    }

    ccl_group_scope(const ccl_group_scope&) = delete;
    ccl_group_scope& operator=(const ccl_group_scope&) = delete;

    ~ccl_group_scope() {
        if (active) {
            try {
                ccl::group_end();
            }
            catch (...) {
            }
        }
    }

    void end() {
        active = false;
        ccl::group_end();
    }

private:
    bool active = false;
};

sycl::device get_device_for_rank(int local_rank, int local_size) {
    auto devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (devices.size() < static_cast<std::size_t>(local_size)) {
        if (local_rank == 0) {
            std::cerr << "Grouped NCCL test requires " << local_size
                      << " visible GPU devices for the local MPI ranks, found " << devices.size()
                      << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return devices[local_rank];
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            std::cout << "Need at least 2 ranks to run grouped P2P test" << std::endl;
        }
        MPI_Finalize();
        return 0;
    }

    const char* backend = std::getenv("CCL_BACKEND");
    if (backend == nullptr || std::string(backend) != "nccl") {
        if (rank == 0) {
            std::cerr << "Set CCL_BACKEND=nccl to run grouped NCCL P2P test" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    ccl::init();

    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank = 0;
    int local_size = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_size);
    MPI_Comm_free(&local_comm);

    sycl::device dev = get_device_for_rank(local_rank, local_size);
    sycl::context ctx(dev);
    sycl::queue q(ctx, dev, sycl::property::queue::in_order());

    ccl::shared_ptr_class<ccl::kvs> kvs;
    ccl::kvs::address_type addr;
    if (rank == 0) {
        kvs = ccl::create_main_kvs();
        addr = kvs->get_address();
        MPI_Bcast(addr.data(), addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    else {
        MPI_Bcast(addr.data(), addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        kvs = ccl::create_kvs(addr);
    }

    auto ccl_dev = ccl::create_device(q.get_device());
    auto ccl_ctx = ccl::create_context(q.get_context());
    auto comm = ccl::create_communicator(size, rank, ccl_dev, ccl_ctx, kvs);
    auto stream = ccl::create_stream(q);

    const int left = (rank - 1 + size) % size;
    const int right = (rank + 1) % size;
    const size_t count = 1024;

    std::vector<float> send_left_data(count, static_cast<float>(2 * rank));
    std::vector<float> send_right_data(count, static_cast<float>((2 * rank) + 1));
    std::vector<float> recv_left_data(count, 0.0f);
    std::vector<float> recv_right_data(count, 0.0f);

    float* d_send_left = sycl::malloc_device<float>(count, q);
    float* d_send_right = sycl::malloc_device<float>(count, q);
    float* d_recv_left = sycl::malloc_device<float>(count, q);
    float* d_recv_right = sycl::malloc_device<float>(count, q);
    if (d_send_left == nullptr || d_send_right == nullptr || d_recv_left == nullptr ||
        d_recv_right == nullptr) {
        std::cerr << "Rank " << rank << ": failed to allocate device buffers" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    q.memcpy(d_send_left, send_left_data.data(), count * sizeof(float)).wait();
    q.memcpy(d_send_right, send_right_data.data(), count * sizeof(float)).wait();
    q.memset(d_recv_left, 0, count * sizeof(float)).wait();
    q.memset(d_recv_right, 0, count * sizeof(float)).wait();

    ccl_group_scope outer_group;
    auto recv_left = ccl::recv(d_recv_left, count, ccl::datatype::float32, left, comm, stream);

    ccl_group_scope inner_group;
    auto recv_right = ccl::recv(d_recv_right, count, ccl::datatype::float32, right, comm, stream);
    auto send_right = ccl::send(d_send_right, count, ccl::datatype::float32, right, comm, stream);
    if (recv_left.test() || recv_right.test() || send_right.test()) {
        std::cerr << "Rank " << rank << ": grouped event completed inside nested group"
                  << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    inner_group.end();
    if (recv_left.test() || recv_right.test() || send_right.test()) {
        std::cerr << "Rank " << rank << ": grouped event completed at nested group_end"
                  << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    auto send_left = ccl::send(d_send_left, count, ccl::datatype::float32, left, comm, stream);
    if (send_left.test()) {
        std::cerr << "Rank " << rank << ": outer grouped event completed before group_end"
                  << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    bool wait_threw = false;
    try {
        recv_left.wait();
    }
    catch (const std::exception&) {
        wait_threw = true;
    }
    if (!wait_threw) {
        std::cerr << "Rank " << rank << ": grouped event wait succeeded before group_end"
                  << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    bool get_native_threw = false;
    try {
        (void)recv_left.get_native();
    }
    catch (const std::exception&) {
        get_native_threw = true;
    }
    if (!get_native_threw) {
        std::cerr << "Rank " << rank
                  << ": grouped event native handle was available before group_end" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    outer_group.end();

    recv_left.get_native().wait();
    recv_left.wait();
    recv_right.wait();
    send_right.wait();
    send_left.wait();

    bool unmatched_end_threw = false;
    try {
        ccl::group_end();
    }
    catch (const std::exception&) {
        unmatched_end_threw = true;
    }
    if (!unmatched_end_threw) {
        std::cerr << "Rank " << rank << ": group_end without group_start succeeded" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    q.memcpy(recv_left_data.data(), d_recv_left, count * sizeof(float)).wait();
    q.memcpy(recv_right_data.data(), d_recv_right, count * sizeof(float)).wait();

    const float expected_left = static_cast<float>((2 * left) + 1);
    const float expected_right = static_cast<float>(2 * right);
    bool success = true;
    for (size_t i = 0; i < count; ++i) {
        if (std::fabs(recv_left_data[i] - expected_left) > 1e-5f ||
            std::fabs(recv_right_data[i] - expected_right) > 1e-5f) {
            std::cerr << "Rank " << rank << ": ERROR at index " << i << std::endl;
            success = false;
            break;
        }
    }

    int local_ok = success ? 1 : 0;
    int global_ok = 0;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    sycl::free(d_send_left, q);
    sycl::free(d_send_right, q);
    sycl::free(d_recv_left, q);
    sycl::free(d_recv_right, q);

    if (rank == 0) {
        std::cout << (global_ok ? "PASSED" : "FAILED") << std::endl;
    }

    MPI_Finalize();
    return global_ok ? 0 : 1;
}
