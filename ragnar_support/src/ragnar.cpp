#include <ros/ros.h>
#include <tf/tf.h>
#include <geometry_msgs/PoseStamped.h>


class RagnarTF
{

public:
  RagnarTF::RagnarTF()
  {
    tf::Vector3 t;
    for(int i = 0; i < 4; ++i)
    {
      ai_.push_back(t);
      bi_.push_back(t);
      ci_.push_back(t);
    }
  }

private:
  std::vector<tf::Vector3> ai_;
  std::vector<tf::Vector3> bi_;
  std::vector<tf::Vector3> ci_;
  std::vector<tf::Transform> frames_;

  void publishTF();

};

RagnarTF::publishTF()
{
  frames_.clear();
  // calculate a->b pose for upper link frames
  for(int i = 0; i < 4; ++i)
  {
    tf::Vector3 origin;
    tf::Quaternion pose;
    tf::Transform frame;

    origin.setX(ai_[i].x());
    origin.setY(ai_[i].y());
    origin.setZ(ai_[i].z());

    frame.setOrigin(origin);

    // create pose from ai and bi
    tf::Vector3 n = bi_[i] - ai_[i];
    n.normalize();
    tf::Vector3 z;
    z.setZ(1.0);
    tf::Vector3 y;
    y = n.cross(z);
    y.normalize();
    z = n.cross(y);

    tf::Matrix3x3 rotation(
      n.getX(), y.getX(), z.getX(),
      n.getY(), y.getY(), z.getY(),
      n.getZ(), y.getZ(), z.getZ());

    frame.setBasis(rotation);
    frames_.push_back(frame);
  }

  // calculate b->c pose for lower link frames
  for(int i = 0; i < 4; ++i)
  {
    tf::Vector3 origin;
    tf::Quaternion pose;
    tf::Transform frame;

    origin.setX(bi_[i].x());
    origin.setY(bi_[i].y());
    origin.setZ(bi_[i].z());

    frame.setOrigin(origin);

    // create pose from ai and bi
    tf::Vector3 n = ci_[i] - bi_[i];
    n.normalize();
    tf::Vector3 z;
    z.setZ(1.0);
    tf::Vector3 y;
    y = n.cross(z);
    y.normalize();
    z = n.cross(y);

    tf::Matrix3x3 rotation(
      n.getX(), y.getX(), z.getX(),
      n.getY(), y.getY(), z.getY(),
      n.getZ(), y.getZ(), z.getZ());

    frame.setBasis(rotation);
    frames_.push_back(frame);
  }

}


int main(int argc, char **argv)
{
  // Start up ROS
  ros::init(argc, argv, "ragnar_tf");
  ros::NodeHandle nh;
  ros::NodeHandle priv_nh("~");


  ros::spin();


  return 0;
} //main
