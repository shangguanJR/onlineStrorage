cmake_minimum_required(VERSION 3.16)
project(VelRegistrationApp VERSION 0.1.0 LANGUAGES C CXX)

add_compile_options(-std=c++17)

if(NOT DEFINED VTK_DIR)
    set(VTK_DIR "/home/nzl/vela/build/VTK-build")
endif()

find_package(VTK REQUIRED)
find_package(Threads REQUIRED)

add_subdirectory(VelRegistration)

if(NOT DEFINED BUILD_TESTING)
    option(BUILD_TESTING "Build Testing" ON)
    message(STATUS "BUILD_TESTING Default set to ON.")
endif()

if(BUILD_TESTING)
    include(CTest)
    enable_testing()
    add_subdirectory(VelRegistrationTest)
endif(BUILD_TESTING)
