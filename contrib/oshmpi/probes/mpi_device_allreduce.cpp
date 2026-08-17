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
 * Can this MPI reduce CUDA device memory at all?
 *
 * OSHMPI is deliberately not involved. The device allreduce failure seen through
 * OSHMPI was inside Open MPI: libnbc's software fallback ran the arithmetic on
 * the host CPU (ompi_op_avx_2buff_add_float_avx2) over a device pointer. That is
 * an MPI-layer property, so it is cheaper to measure it directly than through
 * OSHMPI, and the result decides whether oneCCL must host-stage allreduce.
 *
 * Two axes matter, and they interact:
 *
 *   blocking vs nonblocking - OSHMPI picks MPI_Allreduce when built with
 *       --enable-async-thread=yes and MPI_Iallreduce otherwise. The Leonardo
 *       build uses `no`, so it takes the non-blocking path. HCOLL implements
 *       only blocking collectives, so it can never service MPI_Iallreduce no
 *       matter how it is configured; UCC's non-blocking coverage varies. If
 *       blocking works and non-blocking does not, rebuilding OSHMPI with
 *       --enable-async-thread=yes is a cheaper fix than host staging.
 *
 *   the collective component - selected by the caller through OMPI_MCA_* in the
 *       environment, so this program does not need to know about it.
 *
 * usage: mpi_device_allreduce <blocking|nonblocking>
 */

#include <mpi.h>

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int element_count = 1024;

bool cuda_ok(cudaError_t status, const char* what, int rank) {
    if (status != cudaSuccess) {
        std::cerr << "rank " << rank << ": " << what << ": " << cudaGetErrorString(status)
                  << std::endl;
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

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: " << argv[0] << " <blocking|nonblocking|nonblocking-test>" << std::endl;
        return 2;
    }
    const std::string mode = argv[1];
    if (mode != "blocking" && mode != "nonblocking" && mode != "nonblocking-test") {
        std::cerr << "unknown mode: " << mode << std::endl;
        return 2;
    }

    // PROBE_THREAD_LEVEL selects how MPI is initialised. This is not incidental:
    // Open MPI picks its collective module per communicator at creation time, and
    // UCC (priority 100, GPU-capable) disqualifies itself above MPI_THREAD_SINGLE
    // in some HPC-X builds. When it declines, libnbc (priority 10) handles the
    // allreduce and reduces on the host CPU, faulting on device pointers. OSHMPI
    // always requests SERIALIZED, so this reproduces its selection from a plain
    // MPI program.
    const char* requested = std::getenv("PROBE_THREAD_LEVEL");
    const std::string thread_level = requested && *requested ? requested : "single";
    int want = MPI_THREAD_SINGLE;
    if (thread_level == "serialized") {
        want = MPI_THREAD_SERIALIZED;
    }
    else if (thread_level == "multiple") {
        want = MPI_THREAD_MULTIPLE;
    }
    else if (thread_level == "funneled") {
        want = MPI_THREAD_FUNNELED;
    }

    int granted = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, want, &granted);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int device_count = 0;
    if (!cuda_ok(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount", rank) ||
        device_count == 0) {
        if (rank == 0) {
            std::cout << "FAIL " << mode << " (no CUDA device)" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    if (!cuda_ok(cudaSetDevice(local_rank_from_env(rank) % device_count), "cudaSetDevice", rank)) {
        MPI_Finalize();
        return 1;
    }

    float* device_source = nullptr;
    float* device_result = nullptr;
    const std::size_t bytes = static_cast<std::size_t>(element_count) * sizeof(float);
    if (!cuda_ok(cudaMalloc(&device_source, bytes), "cudaMalloc source", rank) ||
        !cuda_ok(cudaMalloc(&device_result, bytes), "cudaMalloc result", rank)) {
        MPI_Finalize();
        return 1;
    }

    std::vector<float> host(element_count, static_cast<float>(rank + 1));
    if (!cuda_ok(cudaMemcpy(device_source, host.data(), bytes, cudaMemcpyHostToDevice),
                 "cudaMemcpy H2D",
                 rank)) {
        MPI_Finalize();
        return 1;
    }
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);

    // Reaching the line after this call is itself part of the result: the known
    // failure mode is a segfault inside the host reduction kernel, not an error
    // return.
    if (mode == "blocking") {
        MPI_Allreduce(device_source, device_result, element_count, MPI_FLOAT, MPI_SUM,
                      MPI_COMM_WORLD);
    }
    else if (mode == "nonblocking") {
        MPI_Request request = MPI_REQUEST_NULL;
        MPI_Iallreduce(device_source, device_result, element_count, MPI_FLOAT, MPI_SUM,
                       MPI_COMM_WORLD, &request);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
    }
    else {
        // Same call, driven the way OSHMPI drives it when built with
        // --enable-async-thread=no: a MPI_Test spin rather than MPI_Wait. This is
        // the only remaining difference between the direct MPI_Iallreduce that
        // passed and OSHMPI's, which faulted in libnbc's host reduction.
        MPI_Request request = MPI_REQUEST_NULL;
        MPI_Iallreduce(device_source, device_result, element_count, MPI_FLOAT, MPI_SUM,
                       MPI_COMM_WORLD, &request);
        int done = 0;
        while (!done) {
            MPI_Test(&request, &done, MPI_STATUS_IGNORE);
        }
    }

    cudaDeviceSynchronize();
    std::vector<float> result(element_count, 0.0f);
    if (!cuda_ok(cudaMemcpy(result.data(), device_result, bytes, cudaMemcpyDeviceToHost),
                 "cudaMemcpy D2H",
                 rank)) {
        MPI_Finalize();
        return 1;
    }

    const float expected = static_cast<float>(size * (size + 1) / 2);
    int ok = 1;
    for (int i = 0; i < element_count; ++i) {
        if (result[i] != expected) {
            ok = 0;
            break;
        }
    }

    int all_ok = 0;
    MPI_Allreduce(&ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    cudaFree(device_source);
    cudaFree(device_result);

    if (rank == 0) {
        std::cout << (all_ok ? "PASS " : "FAIL ") << mode << " thread=" << thread_level
                  << " granted=" << granted << std::endl;
    }
    MPI_Finalize();
    return all_ok ? 0 : 1;
}
