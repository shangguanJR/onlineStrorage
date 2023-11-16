cmake_minimum_required(VERSION 3.16)
project(VelRegistration VERSION 0.1.0 LANGUAGES C CXX)

set(${PROJECT_NAME}_SRCS
    VelRayCastInterpolator.h
    VelRayCastInterpolator.cxx
    VelTwoImageToOneMetric.h
    VelTwoImageToOneMetric.cxx
    VelRegistrationMacro.h
)

add_library(${PROJECT_NAME} SHARED
    ${${PROJECT_NAME}_SRCS}
)

target_link_libraries(${PROJECT_NAME}
    pthread
    VTK::CommonCore
    VTK::ImagingCore
    VTK::IOImage
    Eigen3::Eigen
)