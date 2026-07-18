cmake_minimum_required(VERSION 3.21)

project(AMcoli
    VERSION   0.1.0
    LANGUAGES C CXX
    DESCRIPTION "Universal MoE Disk-Streaming Inference Engine"
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ── Options ──────────────────────────────────────────────────────────────

option(AMCOLI_BUILD_TESTS    "Build unit tests"                     ON)
option(AMCOLI_IO_URING       "Enable io_uring backend (Linux 5.6+)" OFF)
option(AMCOLI_ASAN           "Enable AddressSanitizer"              OFF)

# ── Compiler flags ───────────────────────────────────────────────────────

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Wno-unused-parameter)
    if(AMCOLI_ASAN)
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address)
    endif()
elseif(MSVC)
    add_compile_options(/W4 /wd4100 /wd4996)
endif()

# ── Platform detection ───────────────────────────────────────────────────

if(WIN32)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    add_definitions(-DNOMINMAX)
endif()

if(AMCOLI_IO_URING)
    find_library(URING_LIB uring)
    if(URING_LIB)
        add_definitions(-DAMCOLI_HAS_IO_URING)
        message(STATUS "io_uring: enabled (${URING_LIB})")
    else()
        message(WARNING "io_uring requested but liburing not found, falling back to mmap")
        set(AMCOLI_IO_URING OFF)
    endif()
endif()

# ── Core library ─────────────────────────────────────────────────────────

set(AMCOLI_SOURCES
    src/amcoli.cpp
    src/llama-moe-cache.cpp
    src/llama-disk-streamer.cpp
    src/llama-moe-router.cpp
    src/llama-moe-prefetch.cpp
    src/amcoli-sys-info.cpp
    src/amcoli-bench.cpp
    src/amcoli-downloader.cpp
)

set(AMCOLI_HEADERS
    include/amcoli.h
    src/llama-moe-cache.h
    src/llama-disk-streamer.h
    src/llama-moe-router.h
    src/llama-moe-prefetch.h
    src/amcoli-sys-info.h
    src/amcoli-bench.h
    src/amcoli-downloader.h
)

add_library(amcoli_core STATIC ${AMCOLI_SOURCES} ${AMCOLI_HEADERS})

target_include_directories(amcoli_core
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Platform libraries
if(WIN32)
    # Windows: no extra libs needed (uses Win32 API)
elseif(APPLE)
    # macOS: no madvise flags for MADV_WILLNEED need extra libs
else()
    # Linux: link rt for clock_gettime (older glibc)
    target_link_libraries(amcoli_core PRIVATE rt)
endif()

if(AMCOLI_IO_URING AND URING_LIB)
    target_link_libraries(amcoli_core PRIVATE ${URING_LIB})
endif()

add_executable(amcoli tools/run/main.cpp)
target_link_libraries(amcoli PRIVATE amcoli_core)
target_include_directories(amcoli PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Utility generator
add_executable(generate_synthetic tools/convert/generate_synthetic.cpp)
target_include_directories(generate_synthetic PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ── Tests ────────────────────────────────────────────────────────────────

if(AMCOLI_BUILD_TESTS)
    enable_testing()

    # Cache unit tests
    add_executable(test-moe-cache tests/test-moe-cache.cpp)
    target_link_libraries(test-moe-cache PRIVATE amcoli_core)
    target_include_directories(test-moe-cache PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    add_test(NAME MoECacheTests COMMAND test-moe-cache)

    # Disk streamer tests
    add_executable(test-disk-streamer tests/test-disk-streamer.cpp)
    target_link_libraries(test-disk-streamer PRIVATE amcoli_core)
    target_include_directories(test-disk-streamer PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    add_test(NAME DiskStreamerTests COMMAND test-disk-streamer)

    # Router config tests
    add_executable(test-router-config tests/test-router-config.cpp)
    target_link_libraries(test-router-config PRIVATE amcoli_core)
    target_include_directories(test-router-config PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    add_test(NAME RouterConfigTests COMMAND test-router-config)

    # Benchmark tests
    add_executable(test-bench tests/test-bench.cpp)
    target_link_libraries(test-bench PRIVATE amcoli_core)
    target_include_directories(test-bench PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    add_test(NAME BenchmarkTests COMMAND test-bench)
endif()

# ── Install ──────────────────────────────────────────────────────────────

install(TARGETS amcoli amcoli_core
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(FILES include/amcoli.h DESTINATION include)

# ── Summary ──────────────────────────────────────────────────────────────

message(STATUS "")
message(STATUS "AMcoli v${PROJECT_VERSION}")
message(STATUS "  Build type:    ${CMAKE_BUILD_TYPE}")
message(STATUS "  Tests:         ${AMCOLI_BUILD_TESTS}")
message(STATUS "  io_uring:      ${AMCOLI_IO_URING}")
message(STATUS "  ASan:          ${AMCOLI_ASAN}")
message(STATUS "")
