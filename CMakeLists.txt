cmake_minimum_required(VERSION 3.0.2)
project(visual_place_recognition_ros)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  roscpp
  sensor_msgs
  std_msgs
  tf2
  dbow3_ros
  my_research_tools
)
find_package(OpenCV REQUIRED)

add_message_files(FILES Feature.msg)
generate_messages(DEPENDENCIES sensor_msgs)

###################################
## catkin specific configuration ##
###################################
catkin_package(CATKIN_DEPENDS message_runtime)

###########
## Build ##
###########
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(realtime_visual_place_recognition_node src/realtime_visual_place_recognition.cpp)
target_link_libraries(realtime_visual_place_recognition_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

add_executable(visual_place_recognition_node src/visual_place_recognition.cpp)
target_link_libraries(visual_place_recognition_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

#############
## Testing ##
#############
if(CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)
  find_package(roslint REQUIRED)
  roslint_cpp()
  roslint_add_test()
endif()
