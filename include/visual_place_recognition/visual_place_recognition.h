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
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_datatypes.h>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

struct VPRData
{
  DBoW3::EntryId id;
  float x;
  float y;
  float theta;
};

class VPR
{
public:
  VPR(void);
  void process(void);

private:
  void image_callback(const sensor_msgs::ImageConstPtr &msg);
  DBoW3::Vocabulary load_vocabulary(const std::string &voc_file_path);
  DBoW3::Database create_database(const DBoW3::Vocabulary &voc);
  std::vector<cv::Mat> load_features(const std::vector<std::string> &image_file_paths);
  std::vector<std::string>
  load_image_file_paths(const std::string &image_dir_path, const std::string image_extension, const int num_images);
  std::vector<std::string>
  load_image_file_paths(const std::string &image_dir_path);
  void load_poses(const std::string &image_dir_path);
  void add_db(const std::vector<cv::Mat> &features, DBoW3::Database &db);
  void query(const DBoW3::Database &db, const std::vector<cv::Mat> &features);
  void scale_to_resolution(cv::Mat &image, const int resolution);
  cv::Mat calc_features(const cv::Mat &image);
  void query2(const cv::Mat &features);

  std::string voc_file_path_;
  std::string image_dir_path_;
  std::vector<VPRData> vpr_db_;

  float match_threshold_;
  int resolution_;

  DBoW3::Database db_;

  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;
  ros::Subscriber image_sub_;
  ros::Publisher vpr_pose_pub_;
  ros::Publisher image_pub_;
};

#endif  // VISUAL_PLACE_RECOGNITION_VISUAL_PLACE_RECOGNITION_H
