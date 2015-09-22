#include "ragnar_simulator/ragnar_simulator.h"

ragnar_simulator::RagnarSimulator::RagnarSimulator()
{
  joint_names_.push_back("ragnar_joint1");
  joint_names_.push_back("ragnar_joint2");
  joint_names_.push_back("ragnar_joint3");
  joint_names_.push_back("ragnar_joint4");
  joint_positions_.resize(joint_names_.size());
}

void ragnar_simulator::RagnarSimulator::setJointPositions(const std::vector<double>& new_state)
{
  joint_positions_ = new_state;
}

