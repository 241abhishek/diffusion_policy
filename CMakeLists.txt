cmake_minimum_required(VERSION 3.0.2)
project(diffusion_policy)

find_package(catkin REQUIRED COMPONENTS
  rospy
  # add other required components here
)

catkin_python_setup()

catkin_package()

catkin_install_python(PROGRAMS
  nodes/action_predictor.py
  nodes/csv_data_publisher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
