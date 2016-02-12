#ifndef SPLINE_CREATOR_H
#define SPLINE_CREATOR_H

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

class SplineCreator
{

  // to creat skeleton of image

public:
  SplineCreator(){}
  SplineCreator(cv::Point2i origin, float range_x){origin_ = origin; range_x_=range_x; }
  SplineCreator(cv::Point2i origin, float range_x, float range_y){origin_ = origin; range_x_=range_x; range_y_=range_y; }
  ~SplineCreator(){}

  void createTrajectories(cv::Mat& image, std::vector<std::vector<cv::Point2f> >& trajectories);

private:

  cv::Point2i origin_;
  cv::Size2i image_size_;
  float range_x_;
  float range_y_;

  // Taken from http://opencv-code.com/quick-tips/implementation-of-guo-hall-thinning-algorithm/
  void thinningGuoHallIteration(cv::Mat& im, int iter);
  void thinningGuoHall(cv::Mat& im);

  void convertImage(cv::Mat& inImage, std::vector<std::vector<cv::Point2f> >& trajectories);

  int getNextPoint(cv::Mat& im, cv::Point2i& point, cv::Vec3b color);

  void linkLines(cv::Mat& im, cv::Point2i start, cv::Vec3b color, std::vector<cv::Point2i>& line);

  std::vector<std::vector<cv::Point2f> > convertLineToTrajectory(std::vector<std::vector<cv::Point2i> > lines);

};



#endif // SPLINE_CREATOR_H
