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

#include "nvshmem_m0_adapter.h"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
size_t parse_count(int argc, char** argv) {
    if (argc == 1) {
        return 1024;
    }
    if (argc != 2) {
        throw std::invalid_argument("usage: oneccl_nvshmem_m0 [element-count]");
    }

    const std::string value(argv[1]);
    size_t parsed_chars = 0;
    const unsigned long long parsed = std::stoull(value, &parsed_chars);
    if (parsed_chars != value.size() || parsed == 0 ||
        parsed > std::numeric_limits<size_t>::max() / sizeof(float)) {
        throw std::invalid_argument("element-count must be a positive representable value");
    }
    return static_cast<size_t>(parsed);
}

int get_cuda_ordinal(int node_rank, int device_count) {
    const char* configured_device = std::getenv("ONECCL_NVSHMEM_M0_DEVICE");
    if (configured_device != nullptr) {
        const std::string value(configured_device);
        size_t parsed_chars = 0;
        const long parsed = std::stol(value, &parsed_chars);
        if (parsed_chars != value.size() || parsed < 0 || parsed >= device_count) {
            throw std::invalid_argument(
                "ONECCL_NVSHMEM_M0_DEVICE is not a visible CUDA device ordinal");
        }
        return static_cast<int>(parsed);
    }

    // Schedulers commonly expose one assigned GPU as ordinal zero in each process.
    if (device_count == 1) {
        return 0;
    }

    if (node_rank >= device_count) {
        throw std::runtime_error(
            "MPI node-local rank exceeds the number of visible CUDA devices");
    }
    return node_rank;
}

sycl::device get_cuda_device(int cuda_ordinal) {
    for (const sycl::device& device :
         sycl::device::get_devices(sycl::info::device_type::gpu)) {
        if (device.get_backend() != sycl::backend::ext_oneapi_cuda) {
            continue;
        }

        const auto native_device =
            sycl::get_native<sycl::backend::ext_oneapi_cuda>(device);
        if (static_cast<int>(native_device) == cuda_ordinal) {
            return device;
        }
    }

    throw std::runtime_error("DPC++ did not expose the selected CUDA device");
}

void check_adapter(int status, const char* operation) {
    if (status == 0) {
        return;
    }

    const char* detail = oneccl_nvshmem_m0_last_error();
    throw std::runtime_error(std::string(operation) + " failed" +
                             ((detail != nullptr && detail[0] != '\0')
                                  ? std::string(": ") + detail
                                  : std::string()));
}

void verify_matching_count(size_t count, void* native_stream) {
    double* count_source =
        static_cast<double*>(oneccl_nvshmem_m0_malloc(sizeof(double)));
    double* count_minimum =
        static_cast<double*>(oneccl_nvshmem_m0_malloc(sizeof(double)));
    double* count_maximum =
        static_cast<double*>(oneccl_nvshmem_m0_malloc(sizeof(double)));
    if (count_source == nullptr || count_minimum == nullptr || count_maximum == nullptr) {
        throw std::runtime_error("NVSHMEM count-consensus allocation failed");
    }

    const double local_count = static_cast<double>(count);
    double minimum = 0.0;
    double maximum = 0.0;
    check_adapter(oneccl_nvshmem_m0_copy_async(
                      count_source, &local_count, sizeof(local_count), native_stream),
                  "CUDA count upload");
    check_adapter(oneccl_nvshmem_m0_double_min_reduce(
                      count_minimum, count_source, 1, native_stream),
                  "NVSHMEM count minimum");
    check_adapter(oneccl_nvshmem_m0_double_max_reduce(
                      count_maximum, count_source, 1, native_stream),
                  "NVSHMEM count maximum");
    check_adapter(oneccl_nvshmem_m0_copy_async(
                      &minimum, count_minimum, sizeof(minimum), native_stream),
                  "CUDA count minimum download");
    check_adapter(oneccl_nvshmem_m0_copy_async(
                      &maximum, count_maximum, sizeof(maximum), native_stream),
                  "CUDA count maximum download");
    check_adapter(oneccl_nvshmem_m0_stream_synchronize(native_stream),
                  "CUDA count synchronization");

    oneccl_nvshmem_m0_free(count_maximum);
    oneccl_nvshmem_m0_free(count_minimum);
    oneccl_nvshmem_m0_free(count_source);

    if (minimum != maximum || minimum != local_count) {
        throw std::runtime_error("element-count differs across NVSHMEM PEs");
    }
}
} // namespace

int main(int argc, char** argv) {
    bool mpi_initialized = false;
    bool nvshmem_initialized = false;
    float* symmetric_source = nullptr;
    float* symmetric_destination = nullptr;

    try {
        const size_t count = parse_count(argc, argv);

        int mpi_rank = -1;
        int mpi_size = 0;
        int mpi_node_rank = -1;
        mpi_initialized = true;
        check_adapter(oneccl_nvshmem_m0_mpi_init(
                          &argc, &argv, &mpi_rank, &mpi_size, &mpi_node_rank),
                      "MPI initialization");

        int device_count = 0;
        check_adapter(oneccl_nvshmem_m0_cuda_device_count(&device_count),
                      "CUDA device discovery");
        if (device_count <= 0) {
            throw std::runtime_error("CUDA did not report an available GPU");
        }
        const int cuda_ordinal = get_cuda_ordinal(mpi_node_rank, device_count);
        check_adapter(oneccl_nvshmem_m0_cuda_set_device(cuda_ordinal),
                      "CUDA device selection");
        check_adapter(oneccl_nvshmem_m0_validate_cuda_device(cuda_ordinal),
                      "CUDA device mapping validation");

        check_adapter(oneccl_nvshmem_m0_init(), "NVSHMEM initialization");
        nvshmem_initialized = true;

        const size_t bytes = count * sizeof(float);

        const int rank = oneccl_nvshmem_m0_my_pe();
        const int size = oneccl_nvshmem_m0_n_pes();
        const int node_rank = oneccl_nvshmem_m0_node_pe();
        if (rank < 0 || size <= 0 || node_rank < 0) {
            throw std::runtime_error("NVSHMEM returned an invalid PE topology");
        }
        if (rank != mpi_rank || size != mpi_size) {
            throw std::runtime_error("NVSHMEM PE layout does not match MPI_COMM_WORLD");
        }
        if (device_count > 1 &&
            std::getenv("ONECCL_NVSHMEM_M0_DEVICE") == nullptr &&
            (node_rank != mpi_node_rank || node_rank != cuda_ordinal)) {
            throw std::runtime_error(
                "MPI and NVSHMEM node-local ranks selected different CUDA devices");
        }

        const sycl::device device = get_cuda_device(cuda_ordinal);
        const sycl::context context(device);
        const auto async_handler = [](sycl::exception_list exceptions) {
            for (const std::exception_ptr& exception : exceptions) {
                std::rethrow_exception(exception);
            }
        };
        sycl::queue queue(context,
                          device,
                          async_handler,
                          sycl::property::queue::in_order());
        const auto native_stream =
            sycl::get_native<sycl::backend::ext_oneapi_cuda>(queue);
        void* native_stream_ptr = reinterpret_cast<void*>(native_stream);

        verify_matching_count(count, native_stream_ptr);

        symmetric_source = static_cast<float*>(oneccl_nvshmem_m0_malloc(bytes));
        if (symmetric_source == nullptr) {
            throw std::runtime_error(oneccl_nvshmem_m0_last_error());
        }
        symmetric_destination = static_cast<float*>(oneccl_nvshmem_m0_malloc(bytes));
        if (symmetric_destination == nullptr) {
            throw std::runtime_error(oneccl_nvshmem_m0_last_error());
        }

        float* user_source = sycl::malloc_device<float>(count, queue);
        float* user_destination = sycl::malloc_device<float>(count, queue);
        if (user_source == nullptr || user_destination == nullptr) {
            if (user_source != nullptr) {
                sycl::free(user_source, queue);
            }
            if (user_destination != nullptr) {
                sycl::free(user_destination, queue);
            }
            throw std::runtime_error("SYCL device USM allocation failed");
        }

        std::vector<float> input(count, static_cast<float>(rank + 1));
        std::vector<float> output(count, 0.0f);

        queue.memcpy(user_source, input.data(), bytes).wait_and_throw();
        check_adapter(oneccl_nvshmem_m0_copy_async(
                          symmetric_source,
                          user_source,
                          bytes,
                          native_stream_ptr),
                      "CUDA user-to-symmetric copy");

        check_adapter(oneccl_nvshmem_m0_float_sum_reduce(
                          symmetric_destination,
                          symmetric_source,
                          count,
                          native_stream_ptr),
                      "NVSHMEM on-stream allreduce");
        check_adapter(oneccl_nvshmem_m0_barrier(native_stream_ptr),
                      "NVSHMEM on-stream barrier");

        check_adapter(oneccl_nvshmem_m0_copy_async(
                          user_destination,
                          symmetric_destination,
                          bytes,
                          native_stream_ptr),
                      "CUDA symmetric-to-user copy");
        sycl::event completion = queue.ext_oneapi_submit_barrier();
        completion.wait_and_throw();
        queue.memcpy(output.data(), user_destination, bytes).wait_and_throw();

        sycl::free(user_source, queue);
        sycl::free(user_destination, queue);

        const float expected =
            static_cast<float>(size * (size + 1) / 2);
        bool valid = true;
        for (size_t index = 0; index < count; index++) {
            if (std::fabs(output[index] - expected) > 1.0e-5f) {
                std::cerr << "PE " << rank << " mismatch at index " << index
                          << ": expected " << expected << ", received "
                          << output[index] << std::endl;
                valid = false;
                break;
            }
        }

        const float local_valid = valid ? 1.0f : 0.0f;
        float valid_pe_count = 0.0f;
        check_adapter(oneccl_nvshmem_m0_copy_async(
                          symmetric_source,
                          &local_valid,
                          sizeof(local_valid),
                          native_stream_ptr),
                      "CUDA validation upload");
        check_adapter(oneccl_nvshmem_m0_float_sum_reduce(
                          symmetric_destination,
                          symmetric_source,
                          1,
                          native_stream_ptr),
                      "NVSHMEM validation reduction");
        check_adapter(oneccl_nvshmem_m0_copy_async(
                          &valid_pe_count,
                          symmetric_destination,
                          sizeof(valid_pe_count),
                          native_stream_ptr),
                      "CUDA validation download");
        check_adapter(oneccl_nvshmem_m0_stream_synchronize(native_stream_ptr),
                      "CUDA validation synchronization");
        const bool globally_valid =
            std::fabs(valid_pe_count - static_cast<float>(size)) <= 1.0e-5f;

        oneccl_nvshmem_m0_free(symmetric_destination);
        symmetric_destination = nullptr;
        oneccl_nvshmem_m0_free(symmetric_source);
        symmetric_source = nullptr;
        oneccl_nvshmem_m0_finalize();
        nvshmem_initialized = false;
        mpi_initialized = false;

        if (rank == 0) {
            std::cout << (globally_valid ? "PASSED" : "FAILED") << ": " << size
                      << " PEs, " << count << " float elements, CUDA device "
                      << cuda_ordinal << std::endl;
        }
        return globally_valid ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    catch (const std::exception& error) {
        std::cerr << "NVSHMEM M0 spike failed: " << error.what() << std::endl;
        if (nvshmem_initialized || mpi_initialized) {
            // A local exception may leave peers in a collective, so terminate the job.
            oneccl_nvshmem_m0_global_exit(EXIT_FAILURE);
        }
        return EXIT_FAILURE;
    }
}
