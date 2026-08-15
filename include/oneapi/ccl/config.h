#pragma once

#define CCL_FORCEINLINE   inline __attribute__((always_inline))
#define CCL_FORCENOINLINE __attribute__((noinline))

#if (__GNUC__ >= 6) || defined(__clang__)
#   define CCL_DEPRECATED_ENUM_FIELD __attribute__((deprecated))
#else
#   define CCL_DEPRECATED_ENUM_FIELD
#endif

#if defined(__GNUC__)
#   define CCL_DEPRECATED __attribute__((deprecated))
#else
#   define CCL_DEPRECATED
#endif

/* All symbols shall be internal unless marked as CCL_API */
#ifdef __linux__
#   if __GNUC__ >= 4
#       define CCL_HELPER_DLL_EXPORT __attribute__ ((visibility ("default")))
#   else
#       define CCL_HELPER_DLL_EXPORT
#   endif
#else
#error "unexpected OS"
#endif

#define CCL_API CCL_HELPER_DLL_EXPORT

#define ONECCL_SPEC_VERSION "1.0"

#define CCL_MAJOR_VERSION           2021
#define CCL_MINOR_VERSION           17
#define CCL_UPDATE_VERSION          2
#define CCL_PRODUCT_STATUS     "Gold"
#define CCL_PRODUCT_BUILD_DATE "2026-08-14T 09:16:40Z"
#define CCL_PRODUCT_FULL       "Gold-2021.17.2 2026-08-14T 09:16:40Z (feat/oshmpi/82d9b24)"

/* Enable SYCL support for:
 * 1. Intel oneAPI DPC++ compiler (__INTEL_LLVM_COMPILER)
 * 2. Open-source Intel LLVM/DPC++ compiler (clang with SYCL_LANGUAGE_VERSION)
 */
#if defined(SYCL_LANGUAGE_VERSION)
#if defined(__INTEL_LLVM_COMPILER) || defined(__clang__)
#define CCL_ENABLE_SYCL
/* Only enable Level Zero for Intel compiler by default.
 * For open-source DPC++ with NVIDIA backend, CCL_ENABLE_ZE should be
 * controlled via CMake (-DCCL_ENABLE_ZE=OFF) */
#if defined(__INTEL_LLVM_COMPILER)
#define CCL_ENABLE_ZE
#endif
#endif
#endif

/* NCCL support is configured by CMake (-DCCL_ENABLE_NCCL=ON) */
#ifndef CCL_ENABLE_NCCL
/* #undef CCL_ENABLE_NCCL */
#endif

/* RCCL support is configured by CMake (-DCCL_ENABLE_RCCL=ON) */
#ifndef CCL_ENABLE_RCCL
/* #undef CCL_ENABLE_RCCL */
#endif

/* OSHMPI support is configured by CMake (-DCCL_ENABLE_OSHMPI=ON) */
#ifndef CCL_ENABLE_OSHMPI
/* #undef CCL_ENABLE_OSHMPI */
#endif
