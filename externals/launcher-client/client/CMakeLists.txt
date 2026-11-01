# launcher-client: the emulator side of the launcher's plugin protocol (README.md).
#
# Emulators add this folder (add_subdirectory) and link `launcher_client`. Built on its own, it
# also builds the tests: cmake -S client -B build && cmake --build build && ctest --test-dir build
cmake_minimum_required(VERSION 3.16)
project(launcher_client LANGUAGES CXX)

option(LAUNCHER_CLIENT_VULKAN "Build the optional Vulkan overlay helper (launcher_overlay_vk.h)" ON)
option(LAUNCHER_CLIENT_TESTS "Build the tests" ${PROJECT_IS_TOP_LEVEL})

add_library(launcher_client STATIC
    src/json.cpp
    src/protocol.cpp
    src/session.cpp
    src/runner.cpp
    src/overlay.cpp
    src/transport_linux.cpp
    src/transport_macos.cpp
    src/transport_none.cpp
)
target_include_directories(launcher_client PUBLIC include PRIVATE src)
target_compile_features(launcher_client PRIVATE cxx_std_17)
set_target_properties(launcher_client PROPERTIES CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)
find_package(Threads REQUIRED)
target_link_libraries(launcher_client PRIVATE Threads::Threads)
if(APPLE)
    target_link_libraries(launcher_client PRIVATE "-framework CoreFoundation" "-framework IOSurface")
endif()
if(MSVC)
    target_compile_options(launcher_client PRIVATE /W4)
else()
    target_compile_options(launcher_client PRIVATE -Wall -Wextra -Wpedantic)
endif()

# The Vulkan helper needs only Vulkan's headers: it loads every function through the pointers the
# emulator passes, so it links no loader. An emulator that has its own headers points
# LAUNCHER_CLIENT_VULKAN_INCLUDE at them (the folder that holds vulkan/vulkan.h); otherwise they're
# looked for, in the Vulkan SDK ($VULKAN_SDK) first.
if(LAUNCHER_CLIENT_VULKAN)
    if(NOT LAUNCHER_CLIENT_VULKAN_INCLUDE)
        find_path(LAUNCHER_CLIENT_VULKAN_HEADERS vulkan/vulkan.h HINTS "$ENV{VULKAN_SDK}/include")
        if(LAUNCHER_CLIENT_VULKAN_HEADERS)
            set(LAUNCHER_CLIENT_VULKAN_INCLUDE "${LAUNCHER_CLIENT_VULKAN_HEADERS}")
        endif()
    endif()
    if(LAUNCHER_CLIENT_VULKAN_INCLUDE)
        target_sources(launcher_client PRIVATE src/overlay_vk.cpp)
        if(NOT MSVC)
            # Vulkan structs are written {sType} and filled in after
            set_source_files_properties(src/overlay_vk.cpp PROPERTIES COMPILE_OPTIONS -Wno-missing-field-initializers)
        endif()
        target_include_directories(launcher_client PUBLIC ${LAUNCHER_CLIENT_VULKAN_INCLUDE})
        target_compile_definitions(launcher_client PUBLIC LAUNCHER_CLIENT_VULKAN=1)
    else()
        message(STATUS "launcher_client: no Vulkan headers found; building without the Vulkan helper")
    endif()
endif()

if(LAUNCHER_CLIENT_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
