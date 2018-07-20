cmake_minimum_required(VERSION 2.8.3)
set(PROJECT_NAME_STR image_drivers)
project(${PROJECT_NAME_STR} C CXX)

set(ALL_SRCS
    src/mock_driver.cpp)
find_package(OpenCV REQUIRED)
add_library(3df_image_drivers_lib ${ALL_SRCS})
target_link_libraries(3df_image_drivers_lib
    ${OpenCV_LIBS}
    3df_io_lib)
