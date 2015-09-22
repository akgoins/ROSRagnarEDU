#include <ragnar_kinematics/ragnar_kinematics.h>
#include "ragnar_kinematics/ragnar_kinematic_defs.h"
#include <ros/ros.h>
#include <geometry_msgs/PoseStamped.h>
#include <visualization_msgs/MarkerArray.h>
#include <sensor_msgs/JointState.h>

geometry_msgs::Point fromVec(const Eigen::Vector3d& v)
{
  geometry_msgs::Point pt;
  pt.x = v(1); pt.y = v(0); pt.z = v(2) - 0.05;
  return pt;
}

visualization_msgs::Marker
makeMarker(const Eigen::Vector3d& a,
           const Eigen::Vector3d& b,
           const Eigen::Vector3d& c,
           const std::string& ns)
{
  visualization_msgs::Marker marker;
  marker.header.frame_id = "world";
  marker.header.stamp = ros::Time::now();
  marker.ns = ns;
  marker.id = 0;
  marker.type = visualization_msgs::Marker::LINE_STRIP;
  marker.action = visualization_msgs::Marker::ADD;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = 0.05;
  marker.color.a = 1.0;
  if (ns == "1")
  {
    marker.color.r = 1.0;
  }
  else if (ns == "2")
  {
    marker.color.g = 1.0;
  }
  else if (ns == "3")
  {
    marker.color.b = 1.0;
  }
  else if (ns == "0")
  {
    marker.color.g = 1.0;
    marker.color.r = 1.0;
  }
  else
  {
    marker.color.g = marker.color.b = marker.color.r = 1.0;
  }

  marker.lifetime = ros::Duration(0);
  marker.frame_locked = false;

  marker.points.push_back(fromVec(Eigen::Vector3d(0, 0, 0)));
  marker.points.push_back(fromVec(a));
  marker.points.push_back(fromVec(b));
  marker.points.push_back(fromVec(c));



  return marker;
}

visualization_msgs::MarkerArray
makeArray(const ragnar_kinematics::IntermediatePoints& pts)
{
  visualization_msgs::MarkerArray array;
  array.markers.push_back(makeMarker(pts.A.col(0),
                                     pts.B.col(0),
                                     pts.C.col(0),
                                     "0"));
  array.markers.push_back(makeMarker(pts.A.col(1),
                                     pts.B.col(1),
                                     pts.C.col(1),
                                     "1"));
  array.markers.push_back(makeMarker(pts.A.col(2),
                                     pts.B.col(2),
                                     pts.C.col(2),
                                     "2"));
  array.markers.push_back(makeMarker(pts.A.col(3),
                                     pts.B.col(3),
                                     pts.C.col(3),
                                     "3"));
  return array;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "ragnar_kinematics_test");
  ros::NodeHandle nh;

  ros::Publisher pose_pub = nh.advertise<visualization_msgs::MarkerArray>("ragnar_pose", 1);
  ros::Publisher joint_pub = nh.advertise<sensor_msgs::JointState>("joint_states", 1);

  // Debug print
  debugParams();

  // Initial version of test code reads from stdin
  while (ros::ok())
  {
    std::string line;
    // Read line
    if (!std::getline(std::cin, line))
    {
      ROS_WARN("Could not read from stdin");
      break;
    }
    // Parse out line
    std::string action;
    double args[4];

    std::istringstream ss (line);

    ss >> action >> args[0] >> args[1] >> args[2] >> args[3];

    if (!ss)
    {
      ROS_WARN_STREAM("Error parsing string: " << line);
      continue;
    }

    // Dispatch call
    if (action == std::string("fk"))
    {
      double joints[4];
      double pose[4];

      joints[0] = args[0];
      joints[1] = args[1];
      joints[2] = args[2];
      joints[3] = args[3];

      if (!ragnar_kinematics::forward_kinematics(joints, pose))
      {
        ROS_WARN("FK call failed");
        continue;
      }

      // print out results
      ROS_INFO("Calculated FK:\n%f %f %f %f", pose[0], pose[1], pose[2], pose[3]);
    }
    else if (action == std::string("ik"))
    {
      double pose[4];
      double joints[4];

      pose[0] = args[0];
      pose[1] = args[1];
      pose[2] = args[2];
      pose[3] = args[3];

      if (!ragnar_kinematics::inverse_kinematics(pose, joints))
      {
        ROS_WARN("IK call failed");
        continue;
      }

      // print out results
      ROS_INFO("Calculated IK:\n%f %f %f %f", joints[0], joints[1], joints[2], joints[3]);
    }
    else if (action == std::string("ikp"))
    {
      double pose[4];
      double joints[4];

      pose[0] = args[0];
      pose[1] = args[1];
      pose[2] = args[2];
      pose[3] = args[3];

      if (!ragnar_kinematics::inverse_kinematics(pose, joints))
      {
        ROS_WARN("IK call failed");
        continue;
      }

      ragnar_kinematics::IntermediatePoints pts;
      ragnar_kinematics::calcIntermediatePoints(joints, pose, pts);

      visualization_msgs::MarkerArray array = makeArray(pts);
      pose_pub.publish(array);


      sensor_msgs::JointState js;
      js.header.frame_id= "base_link";
      js.header.stamp = ros::Time::now();
      js.position.assign(joints, joints+4);

      joint_pub.publish(js);

      ROS_INFO_STREAM("A:\n" << pts.A << "\nB:\n" << pts.B << "\nC:\n" << pts.C << "\n");
    }
    else if (action == std::string("circle"))
    {
      double r = 0.2f;
      for (int i = 0; i < 360; ++i)
      {
        double theta = static_cast<double>(i) * M_PI / 180.0;
        double x = std::cos(theta) * r;
        double y = std::sin(theta) * r;
        double z = -0.3;

        double joints[4];
        double pose[4];
        pose[0] = x;
        pose[1] = y;
        pose[2] = z;
        pose[3] = 0.0;
        if (!ragnar_kinematics::inverse_kinematics(pose, joints))
        {
          ROS_WARN("Failed IK");
          break;
        }
        ragnar_kinematics::IntermediatePoints pts;
        ragnar_kinematics::calcIntermediatePoints(joints, pose, pts);

        pose_pub.publish(makeArray(pts));
        ros::Duration(0.1).sleep();
      }
    }
    else
    {
      ROS_WARN_STREAM("Unrecognized action: " << action);
      continue;
    }

  } // end main loop
}
