/**
 * @file visual_place_recognition.cpp
 * @author Toshiki Nakamura
 * @brief C++ implementation of visual place recognition
 * @copyright Copyright (c) 2024
 */

#include <string>
#include <vector>

#include "visual_place_recognition/visual_place_recognition.h"

VPR::VPR(void) : private_nh_("~")
{
  private_nh_.param<std::string>("voc_file_path", voc_file_path_, std::string(""));
  private_nh_.param<std::string>("image_dir_path", image_dir_path_, std::string(""));
  private_nh_.param<float>("match_threshold", match_threshold_, 0.9);
  private_nh_.param<int>("resolution", resolution_, 240);

  image_sub_ = nh_.subscribe("image", 1, &VPR::image_callback, this);
  vpr_pose_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("vpr_pose", 1);
  image_pub_ = nh_.advertise<sensor_msgs::Image>("custom_image", 1);

  ROS_INFO_STREAM(ros::this_node::getName() << " node has started..");
  ROS_INFO_STREAM("voc_file_path: " << voc_file_path_);
  ROS_INFO_STREAM("image_dir_path: " << image_dir_path_);
  ROS_INFO_STREAM("match_threshold: " << match_threshold_);
  ROS_INFO_STREAM("resolution: " << resolution_);
  ROS_INFO_STREAM("");
}

void VPR::image_callback(const sensor_msgs::ImageConstPtr &msg)
{
  cv_bridge::CvImagePtr cv_ptr;
  try{
      cv_ptr = cv_bridge::toCvCopy(msg,sensor_msgs::image_encodings::MONO8);
  }
  catch(cv_bridge::Exception& ex){
      ROS_ERROR("Could not convert to color image");
      return;
  }
  image_pub_.publish(cv_ptr->toImageMsg());
  scale_to_resolution(cv_ptr->image, resolution_);
  query2(calc_features(cv_ptr->image));
}

void VPR::scale_to_resolution(cv::Mat &image, const int resolution)
{
  const int w = image.cols;
  const int h = image.rows;
  cv::resize(image, image, cv::Size(w * resolution / h, resolution));
}

cv::Mat VPR::calc_features(const cv::Mat &image)
{
  cv::Ptr<cv::Feature2D> fdetector = cv::ORB::create();
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
  return descriptors;
}

void VPR::query2(const cv::Mat &features)
{
  DBoW3::QueryResults ret;
  db_.query(features, ret, 4);
  for (const auto &r : ret)
  {
    if (r.Score > match_threshold_)
    {
      ROS_INFO_STREAM("Match to Image " << r.Id << " is " << r.Score);
      for (const auto &data : vpr_db_)
      {
        if (data.id == r.Id)
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
  }
}

void VPR::process(void)
{
  const DBoW3::Vocabulary voc = load_vocabulary(voc_file_path_);
  db_ = create_database(voc);
  // ROS_INFO_STREAM("Database info: " << db);

  std::vector<std::string> image_file_paths = load_image_file_paths(image_dir_path_);
  const std::vector<cv::Mat> features = load_features(image_file_paths);

  load_poses(image_dir_path_);
  add_db(features, db_);

  // query(db, features);
}

void VPR::add_db(const std::vector<cv::Mat> &features, DBoW3::Database &db)
{
  ROS_INFO_STREAM("Add images to database..");
  for (int i = 0; i < features.size(); i++)
  {
    DBoW3::EntryId id = db.add(features[i]);
    vpr_db_[i].id = id;
  }
  ROS_INFO_STREAM("Done");
}

DBoW3::Vocabulary VPR::load_vocabulary(const std::string &voc_file_path)
{
  ROS_INFO_STREAM("Load vocabulary..");
  DBoW3::Vocabulary voc(voc_file_path_);
  ROS_INFO_STREAM("Done");
  return voc;
}

DBoW3::Database VPR::create_database(const DBoW3::Vocabulary &voc)
{
  ROS_INFO_STREAM("Create database..");
  DBoW3::Database db(voc, false, 0);
  ROS_INFO_STREAM("Done");
  return db;
}

std::vector<std::string>
VPR::load_image_file_paths(const std::string &image_dir_path, const std::string image_extension, const int num_images)
{
  std::vector<std::string> image_file_paths;
  image_file_paths.reserve(num_images);
  for (int i = 0; i < num_images; i++)
  {
    std::stringstream ss;
    ss << image_dir_path << "/image" << i << image_extension;
    image_file_paths.push_back(ss.str());
  }
  return image_file_paths;
}

std::vector<std::string>
VPR::load_image_file_paths(const std::string &image_dir_path)
{
  std::ifstream ifs(image_dir_path + "/data.csv");
  std::vector<std::string> image_file_paths;
  std::string line;
  while (getline(ifs, line))
  {
    std::istringstream iss(line);
    std::string buffer;
    getline(iss, buffer, ',');
    image_file_paths.push_back(image_dir_path + "/" + buffer);
  }
  return image_file_paths;
}

void VPR::load_poses(const std::string &image_dir_path)
{
  std::ifstream ifs(image_dir_path + "/data.csv");
  std::vector<std::string> image_file_paths;
  std::string line;
  while (getline(ifs, line))
  {
    std::istringstream iss(line);
    std::string comma_buf;
    std::vector<std::string> line_bufs;
    while (getline(iss, comma_buf, ','))
      line_bufs.push_back(comma_buf);

    VPRData vpr_data;
    vpr_data.id = -1;
    vpr_data.x = std::stof(line_bufs[1]);
    vpr_data.y = std::stof(line_bufs[2]);
    vpr_data.theta = std::stof(line_bufs[3]);
    vpr_db_.push_back(vpr_data);
  }
}

std::vector<cv::Mat> VPR::load_features(const std::vector<std::string> &image_file_paths)
{
  cv::Ptr<cv::Feature2D> fdetector = cv::ORB::create();
  std::vector<cv::KeyPoint> keypoints;
  std::vector<cv::Mat> features;

  for (const auto &image_file_path : image_file_paths)
  {
    // ROS_INFO_STREAM("Read image: " << image_file_path);
    cv::Mat image = cv::imread(image_file_path, 0);

    if (image.empty())
      throw std::runtime_error("Could not open image " + image_file_path);

    // ROS_INFO_STREAM("Extract features: " << image_file_path);
    cv::Mat descriptors;
    fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
    features.push_back(descriptors);
  }

  return features;
}

void VPR::query(const DBoW3::Database &db, const std::vector<cv::Mat> &features)
{
  ROS_INFO_STREAM("Querying..");
  ROS_INFO("----------------------------");
  DBoW3::QueryResults ret;
  for (int i = 0; i < features.size(); i++)
  {
    ROS_INFO_STREAM("Querying image " << i);
    db.query(features[i], ret, 4);
    for (const auto &r : ret)
      if (r.Score > match_threshold_)
        ROS_INFO_STREAM("Image " << r.Id << " is " << r.Score);
    ROS_INFO("----------------------------");
  }
  ROS_INFO_STREAM("Done");
}

int main(int argc, char *argv[])
{
  ros::init(argc, argv, "visual_place_recognition");
  VPR visual_place_recognition;
  visual_place_recognition.process();
  ros::spin();

  return 0;
}
