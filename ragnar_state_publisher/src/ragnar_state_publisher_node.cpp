#include <ros/ros.h>

#include "ragnar_state_publisher/ragnar_state_publisher.h"

int main(int argc, char** argv)
{
  ros::init(argc, argv, "ragnar_state_publisher");

  ragnar_state_publisher::RagnarStatePublisher pub ("joint_states");

  ros::spin();
}
