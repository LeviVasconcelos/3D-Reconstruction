cmake_minimum_required(VERSION 2.8.3)
set(PROJECT_NAME_STR io)
project(${PROJECT_NAME_STR} C CXX)

#-------------------
# set common include folder for module
#-------------------
set(ALL_SRCS
    src/io.cpp)
add_library(3df_io_lib ${ALL_SRCS})
target_link_libraries(3df_io_lib
    ${Boost_LIBRARIES})


