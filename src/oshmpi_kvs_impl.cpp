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
#ifdef CCL_ENABLE_OSHMPI

#include "oshmpi_kvs_impl.hpp"

#include <algorithm>
#include <array>

#include "common/global/global.hpp"

namespace ccl {
namespace {

constexpr std::array<char, 8> address_signature{ { 'O', 'S', 'H', 'M', 'P', 'I', 1, 0 } };

kvs::address_type make_address() {
    kvs::address_type addr{};
    std::copy(address_signature.begin(), address_signature.end(), addr.begin());
    return addr;
}

} // namespace

oshmpi_kvs_impl::oshmpi_kvs_impl() : base_kvs_impl(make_address()) {
    CCL_THROW_IF_NOT(ccl::global_data::env().backend == backend_mode::oshmpi,
                     "unexpected backend");
}

oshmpi_kvs_impl::oshmpi_kvs_impl(const kvs::address_type& addr) : base_kvs_impl(addr) {
    CCL_THROW_IF_NOT(ccl::global_data::env().backend == backend_mode::oshmpi,
                     "unexpected backend");
    CCL_THROW_IF_NOT(is_valid_address(addr), "invalid OSHMPI KVS address");
}

bool oshmpi_kvs_impl::is_valid_address(const kvs::address_type& addr) {
    return std::equal(address_signature.begin(), address_signature.end(), addr.begin());
}

} // namespace ccl

#endif // CCL_ENABLE_OSHMPI
