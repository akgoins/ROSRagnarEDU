/*
  Copyright 2016 Southwest Research Institute

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/

#include <ros/ros.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <ragnar_kinematics/spline_creator.h>

void SplineCreator::createTrajectories(cv::Mat &image, std::vector<std::vector<cv::Point2f> > &trajectories)
{
  link_distance_ = 15;
  link_slope_tolerance_ = 0.7;
  trajectories.clear();
  if(range_x_ == 0)
  {
    ROS_ERROR("range_x parameter set to zero or not set, please set to a positive, non-zero value");
    return;
  }

  ROS_INFO("image size: %d, %d", image.cols, image.rows);
  image_size_.height = image.rows;
  image_size_.width = image.cols;

  cv::Mat bw, clr;
  cv::cvtColor(image, bw, CV_BGR2GRAY);
  cv::threshold(bw, bw, 100, 255, cv::THRESH_BINARY_INV);

  //cv::imshow("binary inverse thresholded image", bw);
  //cv::waitKey();

  thinningGuoHall(bw);

  cv::cvtColor(bw, clr, CV_GRAY2BGR);

  ROS_INFO("start of converting image");
  convertImage(clr, trajectories);
  image = clr;
  //cv::imshow("src", image);
  //cv::imshow("dst", bw);
  //cv::imshow("color", clr);

  //cv::waitKey();
}

/**
 * Perform one thinning iteration.
 * Normally you wouldn't call this function directly from your code.
 *
 * @param  im    Binary image with range = 0-1
 * @param  iter  0=even, 1=odd
 */
void SplineCreator::thinningGuoHallIteration(cv::Mat& im, int iter)
{
    cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1);

    for (int i = 1; i < im.rows; i++)
    {
        for (int j = 1; j < im.cols; j++)
        {
            uchar p2 = im.at<uchar>(i-1, j);
            uchar p3 = im.at<uchar>(i-1, j+1);
            uchar p4 = im.at<uchar>(i, j+1);
            uchar p5 = im.at<uchar>(i+1, j+1);
            uchar p6 = im.at<uchar>(i+1, j);
            uchar p7 = im.at<uchar>(i+1, j-1);
            uchar p8 = im.at<uchar>(i, j-1);
            uchar p9 = im.at<uchar>(i-1, j-1);

            int C  = (!p2 & (p3 | p4)) + (!p4 & (p5 | p6)) +
                     (!p6 & (p7 | p8)) + (!p8 & (p9 | p2));
            int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
            int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
            int N  = N1 < N2 ? N1 : N2;
            int m  = iter == 0 ? ((p6 | p7 | !p9) & p8) : ((p2 | p3 | !p5) & p4);

            if (C == 1 && (N >= 2 && N <= 3) & m == 0)
                marker.at<uchar>(i,j) = 1;
        }
    }

    im &= ~marker;
}

/**
 * Function for thinning the given binary image
 *
 * @param  im  Binary image with range = 0-255
 */
void SplineCreator::thinningGuoHall(cv::Mat& im)
{
    im /= 255;

    cv::Mat prev = cv::Mat::zeros(im.size(), CV_8UC1);
    cv::Mat diff;

    do {
        thinningGuoHallIteration(im, 0);
        thinningGuoHallIteration(im, 1);
        cv::absdiff(im, prev, diff);
        im.copyTo(prev);
    }
    while (cv::countNonZero(diff) > 0);

    im *= 255;
}


int SplineCreator::getNextPoint(cv::Mat& im, cv::Point2i& point, cv::Vec3b color)
{

  // Search all surrounding pixels to find any white pixels
  int count = 0;
  cv::Point2i pt = point;
  cv::Point2i lastPt;
  int lasti = 0;
  int lastj = 0;
  for(int ii = -1; ii < 2; ++ii)
  {
    for(int jj = -1; jj < 2; ++jj)
    {
      if(ii == 0 && jj == 0) continue;

      // if pixel is white, increase the pixel found count
      cv::Vec3b p = im.at<cv::Vec3b>(point.x + ii, point.y + jj);
      if(p.val[0] >= 200 && p.val[1] >= 200 && p.val[2] >= 200)
      {
        pt.x = point.x + ii;
        pt.y = point.y + jj;
        ++count;
      }
      else if(p == color) // if the pixel color is the same color as the line, save it since it is the previous pixel found
      {
        lastPt.x = point.x + ii;
        lastPt.y = point.y + jj;
        lasti = ii;
        lastj = jj;
        ++count;
      }
    }
  }


  // Initial attempt to make lines track in the same direction in the event that two lines cross
  // Return the white pixel that is on the opposite side as the previously colored pixel
  /*
  if(count > 1 && lasti != 0 && lastj != 0)
  {
    //ROS_INFO("count is greater than 1, last i,j: %d, %d", lasti, lastj);
    cv::Vec3b p1 = im.at<cv::Vec3b>(point.x - lasti, point.y - lastj);
    if(p1.val[0] >= 200 && p1.val[1] >= 200 && p1.val[2] >= 200)
    {
      pt.x = point.x - lasti;
      pt.y = point.y - lastj;
    }
    else
    {
      cv::Point2i testPt;
      if(lasti == 0 || lastj == 0 )
      {
        testPt = cv::Point2i(point.x - lasti + lastj, point.y - lastj - lasti);
      }
      else
      {
        testPt = cv::Point2i(point.x , point.y - lastj);
      }

      cv::Vec3b p = im.at<cv::Vec3b>(testPt);
      if(p.val[0] >= 200 && p.val[1] >= 200 && p.val[2] >= 200)
      {
        pt = testPt;
      }
      else
      {

        if(lasti == 0 || lastj == 0 )
        {
          testPt = cv::Point2i(point.x - lasti - lastj, point.y - lastj + lasti);
        }
        else
        {
          testPt = cv::Point2i(point.x - lasti, point.y );
        }

        p = im.at<cv::Vec3b>(testPt);
        if(p.val[0] >= 200 && p.val[1] >= 200 && p.val[2] >= 200)
        {
          pt = testPt;
        }
      }
    }

  }*/

  // Return the next continuguous point in the line to color
  point = pt;
  return count;
}

 void SplineCreator::createLines(cv::Mat& im, cv::Point2i start, cv::Vec3b color, std::vector<cv::Point2i>& line)
{

  cv::Point2i pt = start;
  int count = getNextPoint(im, pt, color);

  if(count > 1)
  {
    return;
  }

  im.at<cv::Vec3b>(start.x, start.y) = color;

  int error = 0;
  while((count < 3 && count > 0) && error < 2000)
  {
    cv::Point2i PtIn = pt;
    im.at<cv::Vec3b>(pt.x, pt.y) = color;
    count = getNextPoint(im, pt, color);
    if(PtIn == pt) break;
    line.push_back(pt);
    ++error;
  }


  // Repeat to make sure we get all of line segment;
  pt = start;
  error = 0;
  while((count < 3 && count > 0) && error < 2000)
  {
    cv::Point2i PtIn = pt;
    im.at<cv::Vec3b>(pt.x, pt.y) = color;
    count = getNextPoint(im, pt, color);
    if(PtIn == pt) break;
    line.insert(line.begin(), pt);
    ++error;
  }

  if(line.size() < 10)
  {
    ROS_WARN("line of size less than 10, deleting line");
      line.clear();
  }

}



bool SplineCreator::isNear(cv::Point2i pt1_a, cv::Point2i pt1_b, cv::Point2i pt2_a, cv::Point2i pt2_b)
{
  double dist = sqrt(pow(double(pt1_a.x) - double(pt2_a.x), 2.0) + pow(double(pt1_a.y) - double(pt2_a.y), 2.0));
  double slope1 = atan((double(pt1_a.y) - double(pt1_b.y))/(double(pt1_a.x) - double(pt1_b.x)));
  double slope2 = atan((double(pt2_a.y) - double(pt2_b.y))/(double(pt2_a.x) - double(pt2_b.x)));

  ROS_INFO("pt1 %d, %d; pt2 %d, %d", pt1_a.x, pt1_a.y, pt2_a.x, pt2_a.y);
  //ROS_INFO("pt1b %d, %d; pt2b %d, %d", pt1_b.x, pt1_b.y, pt2_b.x, pt2_b.y);
  ROS_INFO("pt distance: %.3f, slope1 %.3f, slope 2 %.3f", dist, slope1, slope2);
  return (dist <= link_distance_ && fabs(slope1 - slope2) <= link_slope_tolerance_);
}

bool SplineCreator::isNear(std::vector<cv::Point2i>  line1, std::vector<cv::Point2i> line2, std::vector<cv::Point2i>& out_line )
{
  int incr = 8;
  if(line1.size() < incr +1 || line2.size() < incr +1)
  {
    return false;
  }

  // Might be able to use iterators here to make the code easier

  // back of first line, front of second line
  if(isNear(line1.back(), line1[line1.size() - incr], line2[0], line2[incr]))
  {
    out_line = line1;
    out_line.insert(out_line.end(), line2.begin(), line2.end());
    return true;
  }

  // front of both lines
  if(isNear(line1[0], line1[incr], line2[0], line2[incr]))
  {
    std::vector<cv::Point2i> temp_line = line1;
    std::reverse(temp_line.begin(), temp_line.end());
    out_line = temp_line;
    out_line.insert(out_line.end(), line2.begin(), line2.end());
    return true;
  }

  // back of second line, front of first line
  if(isNear(line1[0], line1[incr], line2.back(), line2[line2.size() - incr]))
  {
    out_line = line2;
    out_line.insert(out_line.end(), line1.begin(), line1.end());
    return true;
  }

  // back of both lines
  if(isNear(line1.back(), line1[line1.size() - incr], line2.back(), line2[line2.size() - incr]))
  {
    std::vector<cv::Point2i> temp_line = line2;
    std::reverse(temp_line.begin(), temp_line.end());
    out_line = line1;
    out_line.insert(out_line.end(), temp_line.begin(), temp_line.end());
    return true;
  }

  return false;

}

void SplineCreator::linkLines( std::vector<std::vector<cv::Point2i> >& lines)
{
  ROS_INFO("number of lines in: %d", lines.size());
  std::vector<std::vector<cv::Point2i> > linked_lines = lines;
  // Check the end point of each line, if the slope of a line matches the slope of the endpoint of
  // another line, connect the two line ends together

  for(int i = 0; i < lines.size(); ++i)
  {

    // for all other lines, try to find a nearest line between line i and line j
    for(int j = i+1; j < lines.size(); ++j )
    {
      std::vector<cv::Point2i> temp_line;
      if(isNear(lines[i], lines[j], temp_line))
      {
        ROS_WARN("linked line found for lines %d and %d", i, j);
        // list of new lines consists of the new linked line, plus all of the remaining lines (without lines i and j)
        linked_lines.clear();
        linked_lines.push_back(temp_line);
        for(int k = 0; k < lines.size(); ++k) // add remaining lines
        {
          if(k == i || k == j)
          {
            continue;
          }
          linked_lines.push_back(lines[k]);
          ROS_INFO("adding line %d", k);
        }

        lines = linked_lines; // update lines list
        i = 0;  // reset i so that linking begins at the start again
        break;

      }



    }

  }



  //lines = linked_lines;

  ROS_INFO("number of lines in: %d", lines.size());
}


void SplineCreator::convertImage(cv::Mat& inImage, std::vector<std::vector<cv::Point2f> >& trajectories)
{
  cv::Vec3b colors[] = {
      cv::Vec3b(0x00, 0x00, 0xff),
      cv::Vec3b(0x00, 0x80, 0xff),
      cv::Vec3b(0x00, 0xff, 0xff),
      cv::Vec3b(0x00, 0xff, 0x88),
      cv::Vec3b(0x00, 0xff, 0x00),
      cv::Vec3b(0x80, 0xff, 0x00),
      cv::Vec3b(0xff, 0xff, 0x00),
      cv::Vec3b(0xff, 0x80, 0x00),
      cv::Vec3b(0xff, 0x00, 0x00),
      cv::Vec3b(0xff, 0x00, 0x80)
    };

  std::vector<std::vector<cv::Point2i> > lines;
  //uchar colors[] = {125, 250, 75, 200, 50, 250, 175, 100, 225, 50};

  int c = 0;
  int count = 1;
  for (int i = 1; i < inImage.rows; i++)
  {
      for (int j = 1; j < inImage.cols; j++)
      {

          cv::Vec3b p2 = inImage.at<cv::Vec3b>(i, j);

          if(p2.val[0] >= 200 && p2.val[1] >= 200 && p2.val[2] >= 200)
          {
            //ROS_INFO_STREAM("linking line number: " << count << " with color " << colors[c]);
            cv::Point2i pt(i, j);
            std::vector<cv::Point2i> line;
            createLines(inImage, pt, colors[c], line);
            if(line.size() > 10)
            {
              lines.push_back(line);
            }
            ++c;
            ++count;
            if(c > 8) c = 0;
          }
      }
  }

  /*
  linkLines(lines);

  // /*
  inImage.setTo(cv::Scalar(0));

  c=0;
  for(int i = 0; i < lines.size(); ++i)
  {
    for(int j = 0; j < lines[i].size(); ++j)
    {
      inImage.at<cv::Vec3b>(lines[i][j].x, lines[i][j].y) = colors[c];
    }
    ++c;
    if(c > 8) c = 0;
  }//*/

  trajectories = convertLineToTrajectory(lines);
  ROS_INFO("returning %u lines", lines.size());
}

std::vector<std::vector<cv::Point2f> > SplineCreator::convertLineToTrajectory(std::vector<std::vector<cv::Point2i> > lines)
{
  double scale_y;
  double scale_x = range_x_ / image_size_.width; // x-axis range, units in m/pixel
  if(range_y_ > 0.01)  // if y-axis range is given, use it; else, use x-axis scale for uniform scaling
  {
    scale_y = range_y_ / image_size_.height;
  }
  else
  {
    scale_y = scale_x;
  }

  ROS_WARN("scale x/y: %.5f, %.5f", scale_x, scale_y);
  std::vector<std::vector<cv::Point2f> > trajectories;

  for(int i = 0; i < lines.size(); ++i)
  {
    // convert all points in the given line to a trajectory that is appropriately scaled
    std::vector<cv::Point2f> traj;
    for(int j = 0; j < lines[i].size(); ++j)
    {
      cv::Point2f pt;
      pt.x = (float(lines[i][j].x) - float(origin_.y)) * scale_x; // convert the point to the correct origin, then scale to meters
      pt.y = (float(lines[i][j].y) - float(origin_.x)) * scale_y;
      traj.push_back(pt);
    }
    ROS_INFO("converted traj of size: %u", traj.size());
    if(traj.size() > 10)
    {
      trajectories.push_back(traj);
    }
  }

  return trajectories;
}

/*
using namespace cv;

int main(int argc, char** argv)
{
  ros::init(argc, argv, "spline_creator");
  ros::NodeHandle nh;

  float range_x = 1.0;

  cv::Point2i origin;
  origin.x = 320;
  origin.y = 240;
  cv::Mat src = cv::imread("/home/alex/test_signature.png");

  ROS_INFO("image size: %d, %d", src.cols, src.rows);

  SplineCreator creator(origin, range_x);
  std::vector<std::vector<cv::Point2f> > trajectories;

  creator.createTrajectories(src, trajectories);


  return 0;
}*/
