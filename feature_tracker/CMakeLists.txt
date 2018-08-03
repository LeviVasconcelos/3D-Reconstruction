cmake_minimum_required(VERSION 2.8.3)
set(PROJECT_NAME_STR feature_track)
project(${PROJECT_NAME_STR} C CXX)

set(ALL_SRCS
    src/feature_tracker_common.cpp
    src/feature_tracker_core.cpp
    src/feature_tracker_utils.cpp
    src/feature_tracker_visualizer.cpp)

add_library(3df_feature_track_lib ${ALL_SRCS})


