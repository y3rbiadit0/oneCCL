if(NOT DEFINED MPIEXEC OR NOT DEFINED TEST_EXECUTABLE OR NOT DEFINED EXPECTED_TEXT)
    message(FATAL_ERROR "MPIEXEC, TEST_EXECUTABLE, and EXPECTED_TEXT are required")
endif()

execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env --unset=NVSHMEM_BOOTSTRAP
            "${MPIEXEC}" -n 2 "${TEST_EXECUTABLE}"
    RESULT_VARIABLE test_result
    OUTPUT_VARIABLE test_output
    ERROR_VARIABLE test_error
)

set(combined_output "${test_output}${test_error}")
if("${test_result}" STREQUAL "0")
    message(FATAL_ERROR "test unexpectedly succeeded\n${combined_output}")
endif()

string(FIND "${combined_output}" "${EXPECTED_TEXT}" expected_position)
if(expected_position EQUAL -1)
    message(FATAL_ERROR
        "test failed without the expected diagnostic: ${EXPECTED_TEXT}\n${combined_output}")
endif()

message(STATUS "observed expected failure: ${EXPECTED_TEXT}")
