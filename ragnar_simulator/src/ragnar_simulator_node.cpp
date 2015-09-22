#include <ros/ros.h>
#include <sensor_msgs/JointState.h> // for publishing robot current position

#include "ragnar_simulator/ragnar_simulator.h"


void publishCurrentState(const ros::TimerEvent&,
                         ros::Publisher& pub,
                         const ragnar_simulator::RagnarSimulator& sim)
{
  sensor_msgs::JointState joint_state;
  joint_state.header.frame_id = "world";
  joint_state.header.stamp = ros::Time::now();
  joint_state.name = sim.getJointNames();
  joint_state.position = sim.getJointPositions();
  pub.publish(joint_state);
}

void setCurrentState(const sensor_msgs::JointStateConstPtr& state_command,
                     ragnar_simulator::RagnarSimulator& sim)
{
  sim.setJointPositions(state_command->position);
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "ragnar_simulator_node");
  ros::NodeHandle nh;
  ros::NodeHandle pnh ("~");

  // pnh loads configuration parameters
  // instantiate simulation
  ragnar_simulator::RagnarSimulator sim;

  // create pub/subscribers and wire them up
  ros::Publisher current_state_pub = nh.advertise<sensor_msgs::JointState>("joint_states", 1);
  ros::Subscriber command_state_sub = 
      nh.subscribe<sensor_msgs::JointState>("joint_command", 1, boost::bind(setCurrentState, 
                                                                            _1, 
                                                                            boost::ref(sim)));
  ros::Timer state_publish_timer =
      nh.createTimer(ros::Duration(0.1), boost::bind(publishCurrentState,
                                                     _1,
                                                     boost::ref(current_state_pub),
                                                     boost::cref(sim)));

  ros::spin();
  return 0;
}