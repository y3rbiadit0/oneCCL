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
#include <cstdint>
#include <mutex>
#include <vector>

#include "oneapi/ccl/types.hpp"
#include "common/oshmpi/oshmpi_device.hpp"

namespace ccl {

class oshmpi_runtime final {
public:
    static oshmpi_runtime& instance();

    oshmpi_runtime(const oshmpi_runtime&) = delete;
    oshmpi_runtime& operator=(const oshmpi_runtime&) = delete;

    void acquire(std::size_t size, std::size_t rank);
    void release() noexcept;

    void barrier();

    /* Every operand-carrying entry point takes the caller's device accessor. The
     * runtime owns host symmetric memory and knows nothing about devices on its
     * own; the accessor is how a device operand gets classified and copied, and a
     * default constructed one makes every buffer host memory. */
    void allgather(const void* send_buf,
                   void* recv_buf,
                   std::size_t bytes,
                   const oshmpi_device::accessor& device);
    void allgather(const void* send_buf,
                   const ccl::vector_class<void*>& recv_bufs,
                   std::size_t bytes,
                   const oshmpi_device::accessor& device);
    void allreduce(const void* send_buf,
                   void* recv_buf,
                   std::size_t count,
                   ccl::datatype dtype,
                   ccl::reduction reduction,
                   const oshmpi_device::accessor& device);
    void alltoall(const void* send_buf,
                  void* recv_buf,
                  std::size_t bytes_per_peer,
                  const oshmpi_device::accessor& device);
    void broadcast(const void* send_buf,
                   void* recv_buf,
                   std::size_t bytes,
                   int root,
                   const oshmpi_device::accessor& device);

    /* Two-sided point to point over one-sided RMA. Both calls block until the
     * transfer is complete, which is what lets the protocol stay stop-and-wait:
     * with blocking semantics a PE has at most one transfer in flight per peer,
     * so a sequence number per peer pair is enough to match chunks without tags.
     * oneCCL pt2pt carries no tag, so matching is by peer and program order. */
    void send(const void* send_buf,
              std::size_t bytes,
              int peer,
              const oshmpi_device::accessor& device);
    void recv(void* recv_buf,
              std::size_t bytes,
              int peer,
              const oshmpi_device::accessor& device);

    int rank() const noexcept {
        return world_rank;
    }

    int size() const noexcept {
        return world_size;
    }

private:
    oshmpi_runtime() = default;

    void check_ready() const;
    void release_pt2pt() noexcept;
    // Undoes the pinning done at acquire(); safe to call when nothing was pinned.
    void unpin_staging() noexcept;
    // Validates the peer and that pt2pt is configured; returns the slot base.
    char* pt2pt_slot_for(int peer) const;
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
    // Whether the staging arena was pinned and so needs unregistering on teardown.
    bool staging_pinned = false;
    // symmetric scratch for the startup agreement reductions
    std::uint64_t* scratch = nullptr;
    std::size_t lane_size = 0;

    /* Point-to-point symmetric state. One landing slot per possible sender, so
     * concurrent senders to the same receiver cannot collide, plus a signal per
     * direction. Sized world_size * pt2pt_slot_size, which grows linearly with
     * the job - CCL_OSHMPI_PT2PT_SLOT_SIZE bounds it, and 0 disables pt2pt. */
    char* pt2pt_slots = nullptr;
    std::uint64_t* pt2pt_data_signal = nullptr;
    std::uint64_t* pt2pt_ack_signal = nullptr;
    std::size_t pt2pt_slot_size = 0;
    // Chunk counters, kept in step by matching send/recv program order.
    std::vector<std::uint64_t> pt2pt_send_seq;
    std::vector<std::uint64_t> pt2pt_recv_seq;
};

} // namespace ccl

#endif // CCL_ENABLE_OSHMPI
