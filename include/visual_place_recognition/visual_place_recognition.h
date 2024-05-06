/**
 * @file visual_place_recognition.h
 * @author Toshiki Nakamura
 * @brief C++ implementation of visual place recognition
 * @copyright Copyright (c) 2024
 */

#ifndef VISUAL_PLACE_RECOGNITION_VISUAL_PLACE_RECOGNITION_H
#define VISUAL_PLACE_RECOGNITION_VISUAL_PLACE_RECOGNITION_H

#include "DBoW3.h"

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

class VPR
{
public:
  VPR(void);
  void process(void);

private:
  // void image_callback(const sensor_msgs::ImageConstPtr &msg);
  DBoW3::Vocabulary load_vocabulary(const std::string &voc_file_path);
  DBoW3::Database create_database(const DBoW3::Vocabulary &voc);
  std::vector<cv::Mat> load_features(const std::vector<std::string> &image_file_paths);
  std::vector<std::string>
  load_image_file_paths(const std::string &image_dir_path, const std::string image_extension, const int num_images);
  void add_db(const std::vector<cv::Mat> &features, DBoW3::Database &db);
  void query(const DBoW3::Database &db, const std::vector<cv::Mat> &features);

  std::string voc_file_path_;
  std::string image_dir_path_;

  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;
  // ros::Subscriber image_sub_;
};

#endif  // VISUAL_PLACE_RECOGNITION_VISUAL_PLACE_RECOGNITION_H
