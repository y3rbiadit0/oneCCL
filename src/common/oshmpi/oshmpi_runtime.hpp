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
#pragma once

#ifdef CCL_ENABLE_OSHMPI

#include <cstddef>
#include <mutex>

#include "oneapi/ccl/types.hpp"

namespace ccl {

class oshmpi_runtime final {
public:
    static oshmpi_runtime& instance();

    oshmpi_runtime(const oshmpi_runtime&) = delete;
    oshmpi_runtime& operator=(const oshmpi_runtime&) = delete;

    void acquire(std::size_t size, std::size_t rank);
    void release() noexcept;

    void barrier();
    void allgather(const void* send_buf, void* recv_buf, std::size_t bytes);
    void allgather(const void* send_buf,
                   const ccl::vector_class<void*>& recv_bufs,
                   std::size_t bytes);
    void allreduce(const void* send_buf,
                   void* recv_buf,
                   std::size_t count,
                   ccl::datatype dtype,
                   ccl::reduction reduction);
    void alltoall(const void* send_buf, void* recv_buf, std::size_t bytes_per_peer);
    void broadcast(const void* send_buf, void* recv_buf, std::size_t bytes, int root);

    int rank() const noexcept {
        return world_rank;
    }

    int size() const noexcept {
        return world_size;
    }

private:
    oshmpi_runtime() = default;

    void check_ready() const;
    void reduce_chunk(void* destination,
                      const void* source,
                      std::size_t count,
                      ccl::datatype dtype,
                      ccl::reduction reduction);

    std::mutex state_mutex;
    std::mutex operation_mutex;
    std::size_t users = 0;
    bool initialized = false;
    bool finalized = false;
    int world_rank = -1;
    int world_size = 0;
    char* staging = nullptr;
    std::size_t lane_size = 0;
};

} // namespace ccl

#endif // CCL_ENABLE_OSHMPI
