/*
 Copyright 2016-2020 Intel Corporation

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

#include "gtest/gtest.h"
#include "oneapi/ccl.hpp"

TEST(group_test, nested_groups_close_at_outermost_end) {
    EXPECT_NO_THROW(ccl::group_start());
    EXPECT_NO_THROW(ccl::group_start());
    EXPECT_NO_THROW(ccl::group_end());
    EXPECT_NO_THROW(ccl::group_end());
    EXPECT_THROW(ccl::group_end(), ccl::exception);
}

int main(int argc, char** argv) {
    ccl::init();
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
