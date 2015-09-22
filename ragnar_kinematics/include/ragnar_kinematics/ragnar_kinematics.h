#ifndef RAGNAR_KINEMATICS_H
#define RAGNAR_KINEMATICS_H

#include <eigen3/Eigen/Dense>

namespace ragnar_kinematics
{

bool inverse_kinematics(const double cartesian_mm[], double actuator_mm[]);
bool forward_kinematics(const double actuator_mm[], double cartesian_mm[]);

struct IntermediatePoints
{
  const static int N_ARMS = 4;
  const static int N_DIMS = 3;
  typedef Eigen::Matrix<double, N_DIMS, N_ARMS> ArmMatrixd;

  ArmMatrixd A; // position of motors
  ArmMatrixd B; // position of elbow
  ArmMatrixd C; // position of mount to eff plate

  // Alignment support
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

bool calcIntermediatePoints(const double actuator_mm[], const double cartesian_mm[],
                            IntermediatePoints& pts);

}

#endif // RAGNAR_KINEMATICS_H

