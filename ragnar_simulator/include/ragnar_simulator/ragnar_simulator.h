#ifndef RAGNAR_SIMULATOR_H
#define RAGNAR_SIMULATOR_H

#include <vector>
#include <string>

namespace ragnar_simulator
{

class RagnarSimulator
{
public:
  RagnarSimulator();

  const std::vector<std::string>& getJointNames() const { return joint_names_; }

  const std::vector<double>& getJointPositions() const { return joint_positions_; }

  void setJointPositions(const std::vector<double>& new_state);

private:
  std::vector<std::string> joint_names_;
  std::vector<double> joint_positions_;
};

}

#endif
