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

size_t oneccl_nvshmem_uid_size(void);
int oneccl_nvshmem_uid_get(void* uid, size_t size);

int oneccl_nvshmem_uid_cuda_device_count(int* count);
int oneccl_nvshmem_uid_cuda_set_device(int device);

int oneccl_nvshmem_uid_init(int rank, int size, const void* uid, size_t uid_size);
int oneccl_nvshmem_uid_is_initialized(void);
int oneccl_nvshmem_uid_my_pe(void);
int oneccl_nvshmem_uid_n_pes(void);

void* oneccl_nvshmem_uid_malloc(size_t size);
void oneccl_nvshmem_uid_barrier_all(void);
void oneccl_nvshmem_uid_free(void* ptr);
void oneccl_nvshmem_uid_finalize(void);

const char* oneccl_nvshmem_uid_last_error(void);

#ifdef __cplusplus
}
#endif
