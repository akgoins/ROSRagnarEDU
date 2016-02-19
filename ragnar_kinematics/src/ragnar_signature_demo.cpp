
#include <ros/ros.h>
#include <ragnar_kinematics/ragnar_kinematics.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <ragnar_kinematics/spline_creator.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <std_srvs/Empty.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

class RagnarSigner
{

public:
    RagnarSigner(){}
    ~RagnarSigner(){}

    void init(ros::NodeHandle nh);

    bool createSignature(std_srvs::Empty::Request& req, std_srvs::Empty::Response& resp);
    bool executeSignature(std_srvs::Empty::Request& req, std_srvs::Empty::Response& resp);

private:

    std::vector<std::vector<cv::Point2f> > trajectories_;
    cv::Mat signature_image_;

    SplineCreator creator_;
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;
    ros::Publisher traj_pub_;
    ros::Publisher image_pub_;
    ros::ServiceServer create_signature_;
    ros::ServiceServer execute_signature_;
};

void RagnarSigner::init(ros::NodeHandle nh)
{
    nh_ = nh;
    ros::NodeHandle pnh("~");
    pnh_ = pnh;
    traj_pub_ = nh_.advertise<trajectory_msgs::JointTrajectory>("joint_path_command", 1);
    image_pub_ = nh_.advertise<sensor_msgs::Image>("signature_path", 1);
    create_signature_ = pnh_.advertiseService("create_signature", &RagnarSigner::createSignature, this);
    execute_signature_ = pnh_.advertiseService("execute_signature", &RagnarSigner::executeSignature, this);
}

static void populateHeader(std_msgs::Header& header)
{
  header.frame_id = "base_link";
  header.stamp = ros::Time::now();
}

static trajectory_msgs::JointTrajectory makeCircleTrajectory()
{
  using namespace trajectory_msgs;
  // Header
  JointTrajectory traj;
  populateHeader(traj.header);

  // Create circle points
  const double r = 0.15;
  const double dt = 0.05;

  double pose[4];
  double joints[4];

  double total_t = dt;

  for (int i = 0; i < 360; ++i)
  {
    pose[0] = r * std::cos(i * M_PI / 180.0);
    pose[1] = r * std::sin(i * M_PI / 180.0);
    pose[2] = -0.35;
    pose[3] = 0.0;

    JointTrajectoryPoint pt;
    if (!ragnar_kinematics::inverse_kinematics(pose, joints))
    {
      ROS_WARN_STREAM("Could not solve for: " << pose[0] << " " << pose[1] << " " << pose[2] << " " << pose[3]);
    }
    else
    {
      pt.positions.assign(joints, joints+4);
      pt.time_from_start = ros::Duration(total_t);
      total_t += dt;
      traj.points.push_back(pt);
    }
  }
  return traj;
}

// Linear trajectory helpers
struct RagnarPoint {
  double joints[4];
};

struct RagnarPose {
  RagnarPose() {}

  RagnarPose(double x, double y, double z)
  {
    pose[0] = x;
    pose[1] = y;
    pose[2] = z;
    pose[3] = 0.0;
  }

  double pose[4];
};

RagnarPose interpPose(const RagnarPose& start, const RagnarPose& stop, double ratio)
{
  RagnarPose result;
  result.pose[0] = start.pose[0] + ratio * (stop.pose[0] - start.pose[0]);
  result.pose[1] = start.pose[1] + ratio * (stop.pose[1] - start.pose[1]);
  result.pose[2] = start.pose[2] + ratio * (stop.pose[2] - start.pose[2]);
  result.pose[3] = start.pose[3] + ratio * (stop.pose[3] - start.pose[3]);
  return result;
}

bool linearMove(const RagnarPose& start, const RagnarPose& stop, double ds, std::vector<RagnarPoint>& out)
{
  std::vector<RagnarPoint> pts;
  double delta_x = stop.pose[0] - start.pose[0];
  double delta_y = stop.pose[1] - start.pose[1];
  double delta_z = stop.pose[2] - start.pose[2];
  double delta_s = std::sqrt(delta_x * delta_x + delta_y * delta_y + delta_z * delta_z);

  unsigned steps = static_cast<unsigned>(delta_s / ds) + 1;

  for (unsigned i = 0; i <= steps; i++)
  {
    double ratio = static_cast<double>(i) / steps;
    RagnarPose pose = interpPose(start, stop, ratio);
    RagnarPoint pt;
    if (!ragnar_kinematics::inverse_kinematics(pose.pose, pt.joints))
    {
      return false;
    }
    pts.push_back(pt);
  }

  out = pts;
  return true;
}

std::vector<RagnarPoint> linearMove(const RagnarPose& start, const RagnarPose& stop, double ds)
{
  std::vector<RagnarPoint> pts;
  if (!linearMove(start, stop, ds, pts))
  {
    throw std::runtime_error("Couldn't plan for linear move");
  }
  return pts;
}

typedef std::vector<trajectory_msgs::JointTrajectoryPoint> TrajPointVec;

TrajPointVec toTrajPoints(const std::vector<RagnarPoint>& points, double time)
{
  TrajPointVec vec;
  double dt = time / points.size();
  double total_t = dt;

  for (size_t i = 0; i < points.size(); ++i)
  {
    trajectory_msgs::JointTrajectoryPoint pt;
    pt.positions.assign(points[i].joints, points[i].joints + 4);
    pt.time_from_start = ros::Duration(total_t);
    total_t += dt;

    vec.push_back(pt);
  }

  return vec;
}

// append b to a
TrajPointVec append(const TrajPointVec& a, const TrajPointVec& b)
{
  TrajPointVec result;
  result.reserve(a.size() + b.size());
  // insert a
  result.insert(result.end(), a.begin(), a.end());

  ros::Duration time_end_a = a.back().time_from_start;

  for (size_t i = 0; i < b.size(); ++i)
  {
    trajectory_msgs::JointTrajectoryPoint pt = b[i];
    pt.time_from_start += time_end_a;
    result.push_back(pt);
  }

  return result;
}



static trajectory_msgs::JointTrajectory makeLineTrajectory()
{
  using namespace trajectory_msgs;
  // Header
  JointTrajectory traj;
  populateHeader(traj.header);

  std::vector<RagnarPoint> points;
  RagnarPose start (-0.1, -0.1, -0.3);
  RagnarPose stop (0.1, 0.1, -0.4);

  if (!linearMove(start, stop, 0.01, points))
  {
    throw std::runtime_error("Linear movement planning failed");
  }

  const double dt = 0.1;
  double total_t = dt;

  for (unsigned i = 0; i < points.size(); ++i)
  {
    JointTrajectoryPoint pt;
    pt.positions.assign(points[i].joints, points[i].joints+4);
    pt.time_from_start = ros::Duration(total_t);
    total_t += dt;
    traj.points.push_back(pt);
  }
  return traj;
}

TrajPointVec singlePoint(const RagnarPose& pose, double dt)
{
  RagnarPoint joints;
  if (!ragnar_kinematics::inverse_kinematics(pose.pose, joints.joints))
  {
    throw std::runtime_error("Couldn't plan to point");
  }

  TrajPointVec v;
  trajectory_msgs::JointTrajectoryPoint pt;
  pt.positions.assign(joints.joints, joints.joints + 4);
  pt.time_from_start = ros::Duration(dt);
  v.push_back(pt);
  return v;
}


static trajectory_msgs::JointTrajectory convertToTrajectory(const std::vector<std::vector<cv::Point2f> >& in_traj, const float z, const u_int stride=1)
{
  trajectory_msgs::JointTrajectory traj;
  populateHeader(traj.header);

  const double LINEAR_MOVE_TIME = 1.0;
  const double VERTICAL_MOVE_TIME = 1.0;
  const double WAIT_PERIOD = 0.5;

  double velocity = 5.0;
  // Home position
  RagnarPose home_pt (0.0, 0.0, -0.2);
  TrajPointVec vec = singlePoint(home_pt, 5.0);

  RagnarPose first_pt = home_pt;
  for(int j = 0; j < in_traj.size(); ++j)
  {
    // First point
    ROS_INFO("trajectory size: %d", in_traj[j].size());
    ROS_INFO("starting to make Ragnar trajectory with pt %.3f, %.3f", in_traj[j][0].x, in_traj[j][0].y);
    RagnarPose last_pt (in_traj[j][0].x, in_traj[j][0].y, z + 0.075);
    vec = append(vec, toTrajPoints(linearMove(first_pt, last_pt, 0.01), LINEAR_MOVE_TIME ));

    // remaining points in trajectory
    first_pt = last_pt;
    last_pt.pose[2] = z;
    vec = append(vec, toTrajPoints(linearMove(first_pt, last_pt, 0.01), velocity*0.075));

    ROS_INFO_STREAM( "traj " << j << " size " << in_traj[j].size() );

    for(int i = 1; i < in_traj[j].size(); i=i+stride)
    {

      RagnarPose next_pt (in_traj[j][i].x, in_traj[j][i].y, z);
      double dist = sqrt(pow((last_pt.pose[0] - next_pt.pose[0]),2.0) + pow((last_pt.pose[1] - next_pt.pose[1]),2.0));
      vec = append(vec, toTrajPoints(linearMove(last_pt, next_pt, 0.01), velocity*dist));
      last_pt = next_pt;
    }

    // UP
    first_pt = last_pt;
    first_pt.pose[2] += 0.075;
    vec = append(vec, toTrajPoints(linearMove(last_pt, first_pt, 0.01), velocity*0.075));
  }

  // HOME
  vec = append(vec, toTrajPoints(linearMove(first_pt, home_pt, 0.01), LINEAR_MOVE_TIME));

  traj.points = vec;
  return traj;
}

bool RagnarSigner::createSignature(std_srvs::Empty::Request& req, std_srvs::Empty::Response& resp)
{
  sensor_msgs::ImageConstPtr recent_image = ros::topic::waitForMessage<sensor_msgs::Image>("/usb_camera/image_raw");
  cv_bridge::CvImagePtr bridge = cv_bridge::toCvCopy(recent_image, sensor_msgs::image_encodings::BGR8);

  //cv_bridge bridge;
  //bridge = cv_bridge::toCvCopy(recent_image, "mono8");

  cv::Mat src = bridge->image;

  float range_x = -0.45;
  cv::Point2i origin;
  origin.x = src.cols/2.0;
  origin.y = src.rows/2.0;
  SplineCreator creator(origin, range_x);
  ROS_INFO_STREAM("image size: " << src.rows << " " << src.cols);
  ROS_INFO_STREAM("image origin: " << origin.x << " " << origin.y);
  //std::vector<std::vector<cv::Point2f> > trajectories;
  creator.createTrajectories(src, trajectories_);

  bridge->image = src;
  sensor_msgs::ImagePtr pub = bridge->toImageMsg();
  pub->header.stamp = ros::Time::now();
  image_pub_.publish(pub);

  return true;
}

bool RagnarSigner::executeSignature(std_srvs::Empty::Request& req, std_srvs::Empty::Response& resp)
{
  ROS_INFO("execute signature callback");

    if(trajectories_.size() > 0)
    {
      // trajectory_msgs::JointTrajectory traj = makeLineTrajectory();
      // trajectory_msgs::JointTrajectory traj = makeCircleTrajectory();
      // trajectory_msgs::JointTrajectory traj = makePickPlaceTrajectory();
      trajectory_msgs::JointTrajectory traj = convertToTrajectory(trajectories_, -0.415, 8 );

      std::vector<std::string> names;
      names.push_back("joint_1");
      names.push_back("joint_2");
      names.push_back("joint_3");
      names.push_back("joint_4");

      traj.joint_names = names;
      ros::Duration(0.5).sleep();

      traj_pub_.publish(traj);
    }
    return true;
}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "ragnar_demo_motions");

  RagnarSigner my_signer;

  ros::NodeHandle nh;

  my_signer.init(nh);

  //ros::Publisher traj_pub = nh.advertise<trajectory_msgs::JointTrajectory>("joint_path_command", 1);
  //ros::ServiceServer server = nh.advertiseService("get_image", getImageSevice);

  // Create a trajectory from an image
  cv::Mat src = cv::imread("/home/ros/test_signature.png");
  cv::Point2i origin;


  ros::spin();
}
