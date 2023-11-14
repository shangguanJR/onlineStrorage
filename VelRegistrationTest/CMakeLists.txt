cmake_minimum_required(VERSION 3.16)
project(VelRegistrationTest)

include_directories(
    ${CMAKE_SOURCE_DIR}/VelRegistration
)

add_executable(testDRR
    testDRR.cxx
)

target_link_libraries(testDRR
    VelRegistration
)