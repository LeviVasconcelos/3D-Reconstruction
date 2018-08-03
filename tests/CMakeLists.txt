cmake_minimum_required(VERSION 2.8.3)
set(PROJECT_NAME_STR tests)
project(${PROJECT_NAME_STR} C CXX)

#-------------------
# set common include folder for module
#-------------------
set(TEST_DRIVER_SRCS
    src/image_driver_test.cpp)
set(TEST_HARRIS_SRCS
    src/harris_detection_test.cpp)
set(TEST_TRACK_SRCS
    src/track_test.cpp)
add_executable(test_img_driver ${TEST_DRIVER_SRCS})
target_link_libraries(test_img_driver
    3df_image_drivers_lib
    ${OpenCV_LIBS})
add_executable(harris_detection ${TEST_HARRIS_SRCS})
target_link_libraries(harris_detection
    3df_image_drivers_lib
    3df_feature_track_lib
    ${OpenCV_LIBS})
add_executable(track_test ${TEST_TRACK_SRCS})
target_link_libraries(track_test
    3df_image_drivers_lib
    3df_feature_track_lib)
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")


