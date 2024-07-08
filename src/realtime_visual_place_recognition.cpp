/**
 * @file realtime_visual_place_recognition.cpp
 * @author Toshiki Nakamura
 * @brief C++ implementation of visual place recognition
 * @copyright Copyright (c) 2024
 */

#include <vector>

#include "visual_place_recognition/realtime_visual_place_recognition.h"

RealtimeVPR::RealtimeVPR(void) : private_nh_("~")
{
  private_nh_.param<int>("num_of_voc_create_trigger", vpr_params_.num_of_voc_create_trigger_, 10);
  private_nh_.param<int>("resolution", vpr_params_.resolution_, 240);
  private_nh_.param<float>("dist_threshold", vpr_params_.dist_threshold_, 1.0);

  pose_sub_ = nh_.subscribe("pose", 1, &RealtimeVPR::pose_callback, this);
  image_sub_ = nh_.subscribe("image", 1, &RealtimeVPR::image_callback, this);
  image_pub_ = nh_.advertise<sensor_msgs::Image>("custom_image", 1);

  voc_ = new DBoW3::Vocabulary(dbow3_params_.k, dbow3_params_.L, dbow3_params_.weight, dbow3_params_.score);

  ROS_INFO_STREAM(ros::this_node::getName() << " node has started..");
  ROS_INFO_STREAM("VPR parameters");
  ROS_INFO_STREAM("  num_of_voc_create_trigger: " << vpr_params_.num_of_voc_create_trigger_);
  ROS_INFO_STREAM("  resolution: " << vpr_params_.resolution_);
  ROS_INFO_STREAM("  dist_threshold: " << vpr_params_.dist_threshold_);
  ROS_INFO_STREAM(*voc_);
}

void RealtimeVPR::pose_callback(const geometry_msgs::PoseWithCovarianceStampedConstPtr &msg)
{
  pose_ = *msg;
  if (!vpr_params_.pose_subscribed_)
  {
    base_pose_ = pose_;
    vpr_params_.pose_subscribed_ = true;
  }
}

void RealtimeVPR::image_callback(const sensor_msgs::ImageConstPtr &msg)
{
  cv_bridge::CvImagePtr cv_ptr;
  try
  {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
  }
  catch (cv_bridge::Exception &ex)
  {
    ROS_ERROR("Could not convert to color image");
    return;
  }
  scale_to_resolution(cv_ptr->image, 240);
  image_pub_.publish(cv_ptr->toImageMsg());

  if (vpr_params_.pose_subscribed_)
  {
    const float dist = hypot(
        pose_.pose.pose.position.x - base_pose_.pose.pose.position.x,
        pose_.pose.pose.position.y - base_pose_.pose.pose.position.y);

    if (dist > vpr_params_.dist_threshold_)
    {
      vpr_params_.pose_subscribed_ = false;
      features_.push_back(calc_features(cv_ptr->image));

      if (features_.size() != 0 && features_.size() % vpr_params_.num_of_voc_create_trigger_ == 0)
      {
        ROS_INFO_STREAM("feature size: " << features_.size());
        ROS_WARN_STREAM("create vocabulary");
        ros::Time start = ros::Time::now();
        voc_->create(features_);
        ROS_INFO_STREAM("vocabulary created in " << (ros::Time::now() - start).toSec() << " sec");
        ROS_WARN_STREAM("database created");
        start = ros::Time::now();
        db_ = DBoW3::Database(*voc_, false, 0);
        add_db(features_, db_);
        ROS_INFO_STREAM("database created in " << (ros::Time::now() - start).toSec() << " sec");
      }
    }
  }
}

void RealtimeVPR::scale_to_resolution(cv::Mat &image, const int resolution)
{
  const int w = image.cols;
  const int h = image.rows;
  cv::resize(image, image, cv::Size(w * resolution / h, resolution));
}

cv::Mat RealtimeVPR::calc_features(const cv::Mat &image)
{
  cv::Ptr<cv::Feature2D> fdetector = cv::ORB::create();
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
  return descriptors;
}

void RealtimeVPR::add_db(const std::vector<cv::Mat> &features, DBoW3::Database &db)
{
  for (int i = 0; i < features.size(); i++)
  {
    DBoW3::EntryId id = db.add(features[i]);
    if (vpr_db_.size() > i)
      vpr_db_[i].id = id;
  }
}

int main(int argc, char *argv[])
{
  ros::init(argc, argv, "realtime_visual_place_recognition");
  RealtimeVPR realtime_vpr;
  ros::spin();

  return 0;
}
