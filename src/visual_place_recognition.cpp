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

  // image_sub_ = nh_.subscribe("image", 1, &VPR::image_callback, this);

  ROS_INFO_STREAM(ros::this_node::getName() << " node has started..");
  ROS_INFO_STREAM("voc_file_path: " << voc_file_path_);
  ROS_INFO_STREAM("image_dir_path: " << image_dir_path_);
}

// void VPR::image_callback(const sensor_msgs::ImageConstPtr &msg)
// {
//   ROS_INFO_STREAM("Received image");
// }

void VPR::process(void)
{
  const DBoW3::Vocabulary voc = load_vocabulary(voc_file_path_);
  DBoW3::Database db = create_database(voc);
  ROS_INFO_STREAM("Database info: " << db);

  std::vector<std::string> image_file_paths = load_image_file_paths(image_dir_path_);
  // const std::vector<cv::Mat> features = load_features(image_file_paths);
  // add_db(features, db);
  // query(db, features);
}

void VPR::add_db(const std::vector<cv::Mat> &features, DBoW3::Database &db)
{
  ROS_INFO_STREAM("Add images to database..");
  for (const auto &feature : features)
    db.add(feature);
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

  for (const auto &image_file_path : image_file_paths)
    ROS_INFO_STREAM("Image file path: " << image_file_path);

  return image_file_paths;
}

std::vector<cv::Mat> VPR::load_features(const std::vector<std::string> &image_file_paths)
{
  cv::Ptr<cv::Feature2D> fdetector = cv::ORB::create();
  std::vector<cv::KeyPoint> keypoints;
  std::vector<cv::Mat> features;

  for (const auto &image_file_path : image_file_paths)
  {
    ROS_INFO_STREAM("Read image: " << image_file_path);
    cv::Mat image = cv::imread(image_file_path, 0);

    if (image.empty())
      throw std::runtime_error("Could not open image " + image_file_path);

    ROS_INFO_STREAM("Extract features: " << image_file_path);
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

  return 0;
}
