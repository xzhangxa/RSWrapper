cmake_minimum_required(VERSION 2.8)
project(realsense_publisher)

find_package(catkin REQUIRED COMPONENTS librealsense image_transport cv_bridge roscpp)

include_directories(${catkin_INCLUDE_DIRS})

catkin_package(
    CATKIN_DEPENDS librealsense image_transport cv_bridge roscpp
)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

if(WIN32)
    add_compile_options(/W4)
else()
    add_compile_options(--std=c++11 -Wall)
endif()

include(../lib/CMakeLists.txt)

add_executable(realsense_publisher_node node.cpp)
target_link_libraries(realsense_publisher_node RSWrapper ${catkin_LIBRARIES})

install(TARGETS realsense_publisher_node
    ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
    RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch
    DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
