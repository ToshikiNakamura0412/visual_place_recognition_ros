/**
 * @file realtime_visual_place_recognition.h
 * @author Toshiki Nakamura
 * @brief C++ implementation of visual place recognition
 * @copyright Copyright (c) 2024
 */

#ifndef VISUAL_PLACE_RECOGNITION_REALTIME_VISUAL_PLACE_RECOGNITION_H
#define VISUAL_PLACE_RECOGNITION_REALTIME_VISUAL_PLACE_RECOGNITION_H

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <tf/transform_datatypes.h>
#include <tf2/utils.h>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <visual_place_recognition_ros/Feature.h>

#include "DBoW3.h"

struct VPRParams
{
  bool pose_subscribed_ = false;
  int num_of_voc_create_trigger_ = 10;
  int resolution_ = 240;
  float dist_threshold_ = 1.0;
};

struct DBoW3Params
{
  int k = 9;
  int L = 3;
  DBoW3::WeightingType weight = DBoW3::TF_IDF;
  DBoW3::ScoringType score = DBoW3::L1_NORM;
};

class VPRData
{
public:
  VPRData(void) {}
  VPRData(const float x, const float y, const float theta) : x(x), y(y), theta(theta) {}

  DBoW3::EntryId id = -1;
  float x = 0.0;
  float y = 0.0;
  float theta = 0.0;
};

class RealtimeVPR
{
public:
  RealtimeVPR(void);

private:
  void image_callback(const sensor_msgs::ImageConstPtr &msg);
  void image_callback2(const visual_place_recognition_ros::Feature::ConstPtr &msg);
  void pose_callback(const geometry_msgs::PoseWithCovarianceStampedConstPtr &msg);
  void scale_to_resolution(cv::Mat &image, const int resolution);
  cv::Mat calc_features(const cv::Mat &image);
  void query_pose(const cv::Mat &features);
  void add_db(const std::vector<cv::Mat> &features, DBoW3::Database &db, std::vector<VPRData> &vpr_db);

  VPRParams vpr_params_;
  DBoW3Params dbow3_params_;
  std::vector<cv::Mat> features_;
  DBoW3::Database db_;
  std::vector<VPRData> vpr_db_;

  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;
  ros::Subscriber pose_sub_;
  ros::Subscriber image_sub_;
  ros::Publisher vpr_pose_pub_;
  ros::Publisher image_pub_;

  geometry_msgs::PoseWithCovarianceStamped pose_;
  geometry_msgs::PoseWithCovarianceStamped base_pose_;

  DBoW3::Vocabulary *voc_;
};

#endif  // VISUAL_PLACE_RECOGNITION_REALTIME_VISUAL_PLACE_RECOGNITION_H
