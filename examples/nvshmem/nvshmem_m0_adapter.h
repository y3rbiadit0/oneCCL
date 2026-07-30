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

#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int oneccl_nvshmem_m0_mpi_init(int* argc,
                               char*** argv,
                               int* rank,
                               int* size,
                               int* node_rank);
int oneccl_nvshmem_m0_init(void);
int oneccl_nvshmem_m0_my_pe(void);
int oneccl_nvshmem_m0_n_pes(void);
int oneccl_nvshmem_m0_node_pe(void);

int oneccl_nvshmem_m0_cuda_device_count(int* count);
int oneccl_nvshmem_m0_cuda_set_device(int device);
int oneccl_nvshmem_m0_validate_cuda_device(int device);

void* oneccl_nvshmem_m0_malloc(size_t size);
void oneccl_nvshmem_m0_free(void* ptr);

int oneccl_nvshmem_m0_copy_async(void* destination,
                                 const void* source,
                                 size_t size,
                                 void* native_stream);
int oneccl_nvshmem_m0_stream_synchronize(void* native_stream);

int oneccl_nvshmem_m0_float_sum_reduce(float* destination,
                                       const float* source,
                                       size_t count,
                                       void* native_stream);
int oneccl_nvshmem_m0_double_min_reduce(double* destination,
                                        const double* source,
                                        size_t count,
                                        void* native_stream);
int oneccl_nvshmem_m0_double_max_reduce(double* destination,
                                        const double* source,
                                        size_t count,
                                        void* native_stream);
int oneccl_nvshmem_m0_barrier(void* native_stream);

void oneccl_nvshmem_m0_finalize(void);
void oneccl_nvshmem_m0_global_exit(int status);
const char* oneccl_nvshmem_m0_last_error(void);

#ifdef __cplusplus
}
#endif
