#include "ragnar_state_publisher/ragnar_state_publisher.h"
#include "ragnar_kinematics/ragnar_kinematics.h"
#include "tf/transform_listener.h"
#include "ragnar_kinematics/ragnar_kinematic_defs.h"

static tf::Vector3 toTF(const Eigen::Vector3d& v)
{
  return tf::Vector3(v(1), v(0), v(2));
}

static void calculateDirectedTransform(const Eigen::Vector3d& start,
                                       const Eigen::Vector3d& stop,
                                       const tf::Vector3 z_axis,
                                       tf::Transform& transform)
{
  tf::Vector3 origin;
  tf::Quaternion pose;

  origin.setX(start(1));
  origin.setY(start(0));
  origin.setZ(start(2) - 0.05);

  transform.setOrigin(origin);

  tf::Vector3 n = toTF(stop) - toTF(start);
  n.normalize();
  tf::Vector3 z = z_axis;
  tf::Vector3 y;
  y = n.cross(z);
  y.normalize();
  z = n.cross(y);

  tf::Matrix3x3 rotation(
       z.getX(), n.getX(), y.getX(),
       z.getY(), n.getY(), y.getY(),
       z.getZ(), n.getZ(), y.getZ());

  transform.setBasis(rotation);
}

static
void calculateLinkTransforms(const Eigen::Vector3d& a,
                             const Eigen::Vector3d& b,
                             const Eigen::Vector3d& c,
                             const tf::Vector3 z_axis,
                             tf::Transform& upper_tf,
                             tf::Transform& lower_tf)
{
  calculateDirectedTransform(a, b, z_axis, upper_tf);
  calculateDirectedTransform(b, c, z_axis, lower_tf);
}

void calculateEELinkTransform(const ragnar_kinematics::IntermediatePoints::ArmMatrixd& c, tf::Transform& ee_tf)
{
  double x,y,z;
  x=y=z=0;
  tf::Vector3 center;
  for(int i = 0; i < 4; ++i)
  {
    x+=c(1,i);
    y+=c(0,i);
    z+=(c(2,i) -0.05);
  }
  center.setX(x/4.0);
  center.setY(y/4.0);
  center.setZ(z/4.0);
  ee_tf.setOrigin(center);
}

namespace rsp = ragnar_state_publisher;



rsp::RagnarStatePublisher::RagnarStatePublisher(const std::string& joints_topic,
                                                const std::vector<std::string>& joint_names,
                                                const std::string& prefix)
  : joint_names_(joint_names)
  , prefix_(prefix)
{
  ROS_ASSERT(joint_names.size() == 4);

  joint_sub_ = nh_.subscribe<sensor_msgs::JointState>(joints_topic, 1,
                      boost::bind(&rsp::RagnarStatePublisher::updateJointPosition,
                                  this,
                                  _1));
  tf::Matrix3x3 mat;
  mat.setEulerYPR(-RAGNAR_JOINT1_BASE_PAN, 0, -RAGNAR_JOINT1_BASE_TILT);
  zi_.push_back(mat.getColumn(2));
  mat.setEulerYPR(-RAGNAR_JOINT2_BASE_PAN, 0, -RAGNAR_JOINT2_BASE_TILT);
  zi_.push_back(mat.getColumn(2));
  mat.setEulerYPR(-RAGNAR_JOINT3_BASE_PAN, 0, -RAGNAR_JOINT3_BASE_TILT);
  zi_.push_back(mat.getColumn(2));
  mat.setEulerYPR(-RAGNAR_JOINT4_BASE_PAN, 0, -RAGNAR_JOINT4_BASE_TILT);
  zi_.push_back(mat.getColumn(2));
}

bool rsp::RagnarStatePublisher::extractJoints(const sensor_msgs::JointState& msg, double* actuators) const
{
  int indexes[4] = {-1, -1, -1, -1};
  for (int i = 0; i < (int)msg.name.size(); ++i)
  {
    if (msg.name[i] == joint_names_[0]) indexes[0] = i;
    else if (msg.name[i] == joint_names_[1]) indexes[1] = i;
    else if (msg.name[i] == joint_names_[2]) indexes[2] = i;
    else if (msg.name[i] == joint_names_[3]) indexes[3] = i;
  }

  // Check for failure
  for (int i = 0; i < 4; ++i)
  {
    if (indexes[i] < 0) return false;
  }

  // Otherwise copy values, and continue on
  for (int i = 0; i < 4; i++)
  {
    actuators[i] = msg.position[indexes[i]];
  }

  return true;
}

void rsp::RagnarStatePublisher::updateJointPosition(const sensor_msgs::JointStateConstPtr& joints)
{
  double actuators[4];

  if (!extractJoints(*joints, actuators))
  {
    //  JointState did not contain all of the robot joint names
    return;
  }

  double pose[4];

  // using joint states, calculate forward kinematics of ragnar
  if (!ragnar_kinematics::forward_kinematics(actuators, pose))
  {
    ROS_WARN("Could not calculate FK for given pose");
    return;
  }

  // using joints and FK, calculate intermediate points on the robot
  // pts: A, B, B1, B2, C, C1, C2
  ragnar_kinematics::IntermediatePoints pts;
  ragnar_kinematics::calcIntermediatePoints(actuators, pose, pts);

  // using intermediate points, calculate transform to each link in model

  tf::Transform upper_link, lower_link, ee_link, base_link;
  // Joint 1
  calculateDirectedTransform(pts.A.col(0), pts.B.col(0),zi_[0],upper_link);
  tf_broadcaster_.sendTransform(tf::StampedTransform(upper_link,
                                                     joints->header.stamp,
                                                     prefix_ + "base_link",
                                                     prefix_ + "upper_arm_4"));
  calculateDirectedTransform(pts.B1.col(0), pts.C1.col(0),zi_[0],lower_link);
  tf_broadcaster_.sendTransform(tf::StampedTransform(lower_link,
                                                     joints->header.stamp,
                                                     prefix_ + "base_link",
                                                     prefix_ + "lower_arm_4a"));
  calculateDirectedTransform(pts.B2.col(0), pts.C2.col(0),zi_[0],lower_link);
  tf_broadcaster_.sendTransform(tf::StampedTransform(lower_link,
                                                     joints->header.stamp,
                                                     prefix_ + "base_link",
                                                     prefix_ + "lower_arm_4b"));
  // Joint 2
  calculateDirectedTransform(pts.A.col(1), pts.B.col(1),zi_[1],upper_link);
  tf_broadcaster_.sendTransform(tf::StampedTransform(upper_link,
                                                     joints->header.stamp,
                                                     prefix_ + "base_link",
                                                     prefix_ + "upper_arm_3"));
  calculateDirectedTransform(pts.B1.col(1), pts.C1.col(1),zi_[1],lower_link);
  tf_broadcaster_.sendTransform(tf::StampedTransform(lower_link,
                                                     joints->header.stamp,
                                                     prefix_ + "base_link",
                                                     prefix_ + "lower_arm_3a"));
  calculateDirectedTransform(pts.B2.col(1), pts.C2.col(1),zi_[1],lower_link);
  tf_broadcaster_.sendTransform(tf::StampedTransform(lower_link,
                                                     joints->header.stamp,
                                                     prefix_ + "base_link",
                                                     prefix_ + "lower_arm_3b"));
  // Joint 3
  calculateDirectedTransform(pts.A.col(2), pts.B.col(2),zi_[2],upper_link);
  tf_broadcaster_.sendTransform(tf::StampedTransform(upper_link,
                                                     joints->header.stamp,
                                                     prefix_ + "base_link",
                                                     prefix_ + "upper_arm_2"));
  calculateDirectedTransform(pts.B1.col(2), pts.C1.col(2),zi_[2],lower_link);
  tf_broadcaster_.sendTransform(tf::StampedTransform(lower_link,
                                                     joints->header.stamp,
                                                     prefix_ + "base_link",
                                                     prefix_ + "lower_arm_2a"));
  calculateDirectedTransform(pts.B2.col(2), pts.C2.col(2),zi_[2],lower_link);
  tf_broadcaster_.sendTransform(tf::StampedTransform(lower_link,
                                                     joints->header.stamp,
                                                     prefix_ + "base_link",
                                                     prefix_ + "lower_arm_2b"));
  // Joint 4
  calculateDirectedTransform(pts.A.col(3), pts.B.col(3),zi_[3],upper_link);
  tf_broadcaster_.sendTransform(tf::StampedTransform(upper_link,
                                                     joints->header.stamp,
                                                     prefix_ + "base_link",
                                                     prefix_ + "upper_arm_1"));
  calculateDirectedTransform(pts.B1.col(3), pts.C1.col(3),zi_[3],lower_link);
  tf_broadcaster_.sendTransform(tf::StampedTransform(lower_link,
                                                     joints->header.stamp,
                                                     prefix_ + "base_link",
                                                     prefix_ + "lower_arm_1a"));
  calculateDirectedTransform(pts.B2.col(3), pts.C2.col(3),zi_[3],lower_link);
  tf_broadcaster_.sendTransform(tf::StampedTransform(lower_link,
                                                     joints->header.stamp,
                                                     prefix_ + "base_link",
                                                     prefix_ + "lower_arm_1b"));
  // EE link
  ee_link.setIdentity();
  calculateEELinkTransform(pts.C,ee_link);
  tf_broadcaster_.sendTransform(tf::StampedTransform(ee_link,
                                                     joints->header.stamp,
                                                     prefix_ + "base_link",
                                                     prefix_ + "ee_link"));
}
