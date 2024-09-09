cmake_minimum_required(VERSION 3.0.2)
project(diff_policy)

find_package(catkin REQUIRED COMPONENTS
  rospy
  # add other required components here
  std_msgs
  geometry_msgs
  sensor_msgs
  message_generation
)

catkin_python_setup()

add_message_files(
  FILES
  X2RobotState.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime
)

catkin_install_python(PROGRAMS
  nodes/action_predictor.py
  nodes/csv_data_publisher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
