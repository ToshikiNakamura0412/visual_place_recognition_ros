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
  vpr_pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("vpr_pose", 1);
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
  scale_to_resolution(cv_ptr->image, vpr_params_.resolution_);
  image_pub_.publish(cv_ptr->toImageMsg());

  if (features_.size() > 130)
    query_pose(calc_features(cv_ptr->image));

  if (vpr_params_.pose_subscribed_)
  {
    const float dist = hypot(
        pose_.pose.pose.position.x - base_pose_.pose.pose.position.x,
        pose_.pose.pose.position.y - base_pose_.pose.pose.position.y);

    if (dist > vpr_params_.dist_threshold_)
    {
      vpr_params_.pose_subscribed_ = false;
      features_.push_back(calc_features(cv_ptr->image));
      vpr_db_.push_back(
          VPRData(pose_.pose.pose.position.x, pose_.pose.pose.position.y, tf2::getYaw(pose_.pose.pose.orientation)));

      if (features_.size() != 0 && features_.size() % vpr_params_.num_of_voc_create_trigger_ == 0)
      {
        voc_->create(features_);
        db_ = DBoW3::Database(*voc_, false, 0);
        add_db(features_, db_, vpr_db_);

        ROS_WARN_STREAM("database created (" << features_.size() << " features)");
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

void RealtimeVPR::query_pose(const cv::Mat &features)
{
  DBoW3::QueryResults ret;
  db_.query(features, ret, 1);
  for (const auto &data : vpr_db_)
  {
    if (data.id == ret.front().Id)
    {
      geometry_msgs::PoseStamped pose;
      pose.header.frame_id = "map";
      pose.header.stamp = ros::Time::now();
      pose.pose.position.x = data.x;
      pose.pose.position.y = data.y;
      pose.pose.orientation = tf::createQuaternionMsgFromYaw(data.theta);
      vpr_pose_pub_.publish(pose);
    }
  }
}

void RealtimeVPR::add_db(const std::vector<cv::Mat> &features, DBoW3::Database &db, std::vector<VPRData> &vpr_db)
{
  for (int i = 0; i < features.size(); i++)
  {
    DBoW3::EntryId id = db.add(features[i]);
    if (vpr_db.size() > i)
      vpr_db[i].id = id;
  }
}

int main(int argc, char *argv[])
{
  ros::init(argc, argv, "realtime_visual_place_recognition");
  RealtimeVPR realtime_vpr;
  ros::spin();

  return 0;
}
