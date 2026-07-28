cmake_minimum_required(VERSION 3.21)
project(sequant_ta_repro CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# TiledArray install prefix. Build it with ./setup-tiledarray.sh, which
# pins commit cd53bd3 -- the TiledArray revision real MPQC tracks -- whose
# TA::einsum correctly handles the multi-pair tensor-of-tensor tiles this
# code's coarse occupied tiling produces (older commits crashed on them).
# BLAS (OpenBLAS), MPI, Boost, umpire, range-v3, etc. all resolve
# transitively through this prefix's lib/cmake, so no separate find_package
# is needed. Build the repro with the SAME compiler as TA (clang++-21) --
# see README.md.
set(TA_INSTALL_DIR "${CMAKE_SOURCE_DIR}/third_party/tiledarray-cd53bd3-clang/install"
    CACHE PATH "TiledArray install prefix")
list(PREPEND CMAKE_PREFIX_PATH "${TA_INSTALL_DIR}")
find_package(tiledarray QUIET HINTS "${TA_INSTALL_DIR}/lib/cmake/tiledarray")

# Optional diagnostic / benchmark tools (see tools/README.md); off by
# default so the core build is just the residual driver.
option(SPTC_BUILD_TOOLS "Build the investigation tools under tools/" OFF)

if(tiledarray_FOUND)
    message(STATUS "TiledArray found — building ta_sequant_native_residual_main")

    # The explicit SeQuant -> TiledArray contraction sequence
    # (generated_t{1,2}_residual.cpp, pasted verbatim; see
    # src/ta_sequant_native_residual.h for provenance and regeneration),
    # driven against real leaf data via src/ta_tensor_loader.h.
    add_executable(ta_sequant_native_residual_main
        src/ta_sequant_native_residual_main.cpp
        src/generated_t1_residual.cpp
        src/generated_t2_residual.cpp)
    target_link_libraries(ta_sequant_native_residual_main PRIVATE tiledarray)
    target_include_directories(ta_sequant_native_residual_main
        PRIVATE "${CMAKE_SOURCE_DIR}/src")

    if(SPTC_BUILD_TOOLS AND EXISTS "${CMAKE_SOURCE_DIR}/tools/gen_split/t2_split.cpp")
        # Warm-loop T2 benchmark: persistent cache of the t-independent
        # DF/CSV block (MPQC's volatile/persistent split), SSA-split from
        # the generated T2. See tools/README.md.
        add_executable(ta_warm_t2
            tools/gen_split/ta_warm_t2_main.cpp
            tools/gen_split/t2_split.cpp)
        target_link_libraries(ta_warm_t2 PRIVATE tiledarray)
        target_include_directories(ta_warm_t2
            PRIVATE "${CMAKE_SOURCE_DIR}/src" "${CMAKE_SOURCE_DIR}/tools/gen_split")
    endif()
else()
    message(STATUS "TiledArray not found — run ./setup-tiledarray.sh first")
endif()
