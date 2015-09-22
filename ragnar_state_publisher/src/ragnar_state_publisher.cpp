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


namespace rsp = ragnar_state_publisher;



rsp::RagnarStatePublisher::RagnarStatePublisher(const std::string& joints_topic)
{
  joint_sub_ = nh_.subscribe<sensor_msgs::JointState>(
                      joints_topic, 1,
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

  /*
  tf::TransformListener listener;
  ros::Duration(0.5).sleep();
  if(listener.waitForTransform("base_link2","base_link", ros::Time::now(), ros::Duration(3.0)))
  {
    listener.lookupTransform("base_link2","base_link",ros::Time::now(),base_transform_);
  }
  else
  {
    ROS_WARN("Could not find transform from base_link to base_link2");
  }*/
}

void rsp::RagnarStatePublisher::updateJointPosition(const sensor_msgs::JointStateConstPtr& joints)
{
  ROS_INFO("Handling new joint state, calculating tf");
  // using joint states, calculate forward kinematics of ragnar
  double actuators[4] = {joints->position[0], joints->position[1],
                        joints->position[2], joints->position[3]};

  double pose[4];

  if (!ragnar_kinematics::forward_kinematics(actuators, pose))
  {
    ROS_WARN("Could not calculate FK for given pose");
    return;
  }

  // using joints and FK, calculate intermediate points on the robot
  ragnar_kinematics::IntermediatePoints pts;
  ragnar_kinematics::calcIntermediatePoints(actuators, pose, pts);

  // using intermediate points, calculate transform to each link in model

  tf::Transform upper_link, lower_link;
  // Joint 1
  calculateLinkTransforms(pts.A.col(0), pts.B.col(0), pts.C.col(0), zi_[0], upper_link, lower_link);
  tf_broadcaster_.sendTransform(tf::StampedTransform(upper_link,
                                                     joints->header.stamp,
                                                     "base_link",
                                                     "upper_arm_1"));
  tf_broadcaster_.sendTransform(tf::StampedTransform(lower_link,
                                                     joints->header.stamp,
                                                     "base_link",
                                                     "lower_arm_1_yaw"));
  // Joint 2
  calculateLinkTransforms(pts.A.col(1), pts.B.col(1), pts.C.col(1), zi_[1], upper_link, lower_link);
  tf_broadcaster_.sendTransform(tf::StampedTransform(upper_link,
                                                     joints->header.stamp,
                                                     "base_link",
                                                     "upper_arm_2"));
  tf_broadcaster_.sendTransform(tf::StampedTransform(lower_link,
                                                     joints->header.stamp,
                                                     "base_link",
                                                     "lower_arm_2_yaw"));
  // Joint 3
  calculateLinkTransforms(pts.A.col(2), pts.B.col(2), pts.C.col(2), zi_[2], upper_link, lower_link);
  tf_broadcaster_.sendTransform(tf::StampedTransform(upper_link,
                                                     joints->header.stamp,
                                                     "base_link",
                                                     "upper_arm_3"));
  tf_broadcaster_.sendTransform(tf::StampedTransform(lower_link,
                                                     joints->header.stamp,
                                                     "base_link",
                                                     "lower_arm_3_yaw"));
  // Joint 4
  calculateLinkTransforms(pts.A.col(3), pts.B.col(3), pts.C.col(3), zi_[3], upper_link, lower_link);
  tf_broadcaster_.sendTransform(tf::StampedTransform(upper_link,
                                                     joints->header.stamp,
                                                     "base_link",
                                                     "upper_arm_4"));
  tf_broadcaster_.sendTransform(tf::StampedTransform(lower_link,
                                                     joints->header.stamp,
                                                     "base_link",
                                                     "lower_arm_4_yaw"));
  // world -> base_link
  tf::Transform world_tf = tf::Transform::getIdentity();
  tf_broadcaster_.sendTransform(tf::StampedTransform(world_tf,
                                                     joints->header.stamp,
                                                     "world",
                                                     "base_link"));
}
