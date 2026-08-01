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

#include <mpi.h>
#include <oneapi/ccl.hpp>
#include <sycl/sycl.hpp>

#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    int provided_thread_level = MPI_THREAD_SINGLE;
    const int init_status =
        MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided_thread_level);
    if (init_status != MPI_SUCCESS) {
        std::cerr << "MPI_Init_thread failed with status " << init_status << std::endl;
        return EXIT_FAILURE;
    }
    if (provided_thread_level < MPI_THREAD_MULTIPLE) {
        std::cerr << "MPI does not provide MPI_THREAD_MULTIPLE" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    int rank = -1;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int result = EXIT_SUCCESS;
    try {
        ccl::init();

        MPI_Comm node_comm = MPI_COMM_NULL;
        MPI_Comm_split_type(
            MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
        int node_rank = -1;
        MPI_Comm_rank(node_comm, &node_rank);
        MPI_Comm_free(&node_comm);

        std::optional<sycl::device> selected_device;
        std::string setup_error;
        try {
            std::vector<sycl::device> cuda_devices;
            for (const auto& device :
                 sycl::device::get_devices(sycl::info::device_type::gpu)) {
                if (device.get_backend() == sycl::backend::ext_oneapi_cuda) {
                    cuda_devices.push_back(device);
                }
            }
            if (cuda_devices.empty()) {
                throw std::runtime_error("no CUDA-backed SYCL devices are visible");
            }
            selected_device = cuda_devices.size() == 1
                                  ? cuda_devices.front()
                                  : cuda_devices.at(static_cast<size_t>(node_rank));
        }
        catch (const std::exception& error) {
            setup_error = error.what();
        }

        int setup_ok = setup_error.empty() ? 1 : 0;
        int all_setup_ok = 0;
        MPI_Allreduce(&setup_ok, &all_setup_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        if (all_setup_ok == 0) {
            throw std::runtime_error(setup_error.empty()
                                         ? "another rank failed CUDA device selection"
                                         : setup_error);
        }

        const sycl::device device = *selected_device;
        const sycl::context context(device);
        sycl::queue queue(
            context, device, sycl::property::queue::in_order{});

        ccl::shared_ptr_class<ccl::kvs> kvs;
        ccl::kvs::address_type address{};
        if (rank == 0) {
            kvs = ccl::create_main_kvs();
            address = kvs->get_address();
        }
        MPI_Bcast(address.data(), address.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
        if (rank != 0) {
            kvs = ccl::create_kvs(address);
        }

        auto host_comm = ccl::create_communicator(size, rank, kvs);
        if (host_comm.rank() != rank || host_comm.size() != size) {
            throw std::runtime_error("native host communicator topology does not match MPI");
        }

        auto ccl_device = ccl::create_device(device);
        auto ccl_context = ccl::create_context(context);
        auto comm = ccl::create_communicator(
            size, rank, ccl_device, ccl_context, kvs);
        if (comm.rank() != rank || comm.size() != size) {
            throw std::runtime_error("communicator topology does not match MPI");
        }
        if (comm.get_device().get_native() != device ||
            comm.get_context().get_native() != context) {
            throw std::runtime_error("communicator device or context changed during construction");
        }

        auto stream = ccl::create_stream(queue);
        bool unsupported = false;
        try {
            ccl::barrier(comm, stream);
        }
        catch (const std::exception& error) {
            const std::string message = error.what();
            unsupported = message.find("NVSHMEM M1") != std::string::npos;
        }
        if (!unsupported) {
            throw std::runtime_error("NVSHMEM M1 barrier did not fail explicitly");
        }
    }
    catch (const std::exception& error) {
        std::cerr << "rank " << rank << ": " << error.what() << std::endl;
        result = EXIT_FAILURE;
    }

    int global_result = EXIT_FAILURE;
    MPI_Allreduce(&result, &global_result, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0 && global_result == EXIT_SUCCESS) {
        std::cout << "PASSED: NVSHMEM M1 communicator skeleton" << std::endl;
    }
    MPI_Finalize();
    return global_result;
}
