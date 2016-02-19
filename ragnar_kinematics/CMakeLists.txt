cmake_minimum_required(VERSION 2.8.3)
project(ragnar_kinematics)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_srvs
  cv_bridge
  sensor_msgs
)

find_package(OpenCV)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ragnar_kinematics
  CATKIN_DEPENDS roscpp
)

include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(ragnar_kinematics
 src/ragnar_kinematic_defs.cpp
 src/ragnar_kinematics.cpp
)

target_link_libraries(ragnar_kinematics
  ${catkin_LIBRARIES}
)

add_library(spline_creator
  src/spline_creator.cpp)

add_executable(ragnar_kinematics_node src/ragnar_test_demo.cpp)

add_executable(ragnar_demo_motions src/ragnar_demo_motions.cpp
                src/spline_creator.cpp)

add_executable(ragnar_signature_demo src/ragnar_signature_demo.cpp
                src/spline_creator.cpp)

## add_executable(spline_creator src/spline_creator.cpp)
target_link_libraries(spline_creator
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

target_link_libraries(ragnar_kinematics_node
  ragnar_kinematics
  spline_creator
  ${catkin_LIBRARIES}
)

target_link_libraries(ragnar_demo_motions
  ragnar_kinematics
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

target_link_libraries(ragnar_signature_demo
  ragnar_kinematics
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS ragnar_kinematics ragnar_kinematics_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_ragnar_kinematics.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
