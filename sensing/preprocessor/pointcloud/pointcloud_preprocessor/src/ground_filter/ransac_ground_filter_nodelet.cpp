/*
 * Copyright 2020 Tier IV, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/*
 * Copyright 2017-2019 Autoware Foundation. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 ********************
 *  v1.0: amc-nu (abrahammonrroy@yahoo.com)
 */
/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2014, JSK Lab
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/o2r other materials provided
 *     with the distribution.
 *   * Neither the name of the JSK Lab nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

#include "pointcloud_preprocessor/ground_filter/ransac_ground_filter_nodelet.hpp"

#include "pcl/common/centroid.h"
#include "pcl/visualization/common/common.h"
#include "pcl_ros/transforms.hpp"

#include "tf2_eigen/tf2_eigen.h"
#include <random>

namespace
{
Eigen::Vector3d getArbitralyOrthogonalVector(const Eigen::Vector3d & input)
{
  const double x = input.x();
  const double y = input.y();
  const double z = input.z();
  const double x2 = std::pow(x, 2);
  const double y2 = std::pow(y, 2);
  const double z2 = std::pow(z, 2);

  Eigen::Vector3d unit_vec{0, 0, 0};
  if (x2 <= y2 && x2 <= z2) {
    unit_vec.x() = 0;
    unit_vec.y() = z;
    unit_vec.z() = -y;
    unit_vec = unit_vec / std::sqrt(y2 + z2);
  } else if (y2 <= x2 && y2 <= z2) {
    unit_vec.x() = -z;
    unit_vec.y() = 0;
    unit_vec.z() = x;
    unit_vec = unit_vec / std::sqrt(z2 + x2);
  } else if (z2 <= x2 && z2 <= y2) {
    unit_vec.x() = y;
    unit_vec.y() = -x;
    unit_vec.z() = 0;
    unit_vec = unit_vec / std::sqrt(x2 + y2);
  }
  return unit_vec;
}

pointcloud_preprocessor::PlaneBasis getPlaneBasis(const Eigen::Vector3d & plane_normal)
{
  pointcloud_preprocessor::PlaneBasis basis;
  basis.e_z = plane_normal;
  basis.e_x = getArbitralyOrthogonalVector(plane_normal);
  basis.e_y = basis.e_x.cross(basis.e_z);
  return basis;
}

pointcloud_preprocessor::RGB getRandColor()
{
  pointcloud_preprocessor::RGB color;
  std::random_device rnd;
  std::mt19937 mt(rnd());
  std::uniform_int_distribution<> rrand(0, 255);
  color.r = rrand(mt);
  std::uniform_int_distribution<> grand(0, 255);
  color.g = grand(mt);
  std::uniform_int_distribution<> brand(0, 255);
  color.b = brand(mt);
  return color;
}

geometry_msgs::msg::Pose getDebugPose(const Eigen::Affine3d & plane_affine)
{
  geometry_msgs::msg::Pose debug_pose;
  const Eigen::Quaterniond quat(plane_affine.rotation());
  debug_pose.position.x = plane_affine.translation().x();
  debug_pose.position.y = plane_affine.translation().y();
  debug_pose.position.z = plane_affine.translation().z();
  debug_pose.orientation.x = quat.x();
  debug_pose.orientation.y = quat.y();
  debug_pose.orientation.z = quat.z();
  debug_pose.orientation.w = quat.w();
  return debug_pose;
}
}  // namespace

namespace pointcloud_preprocessor
{
RANSACGroundFilterComponent::RANSACGroundFilterComponent(const rclcpp::NodeOptions & options)
: Filter("RANSACGroundFilter", options)
{
  base_frame_ = declare_parameter("base_frame", "base_link");
  unit_axis_ = declare_parameter("unit_axis", "z");
  min_trial_ = declare_parameter("min_trial", 0);
  max_iterations_ = declare_parameter("max_iterations", 1000);
  min_inliers_ = declare_parameter("min_trial", 5000);
  min_points_ = declare_parameter("min_points", 1000);
  outlier_threshold_ = declare_parameter("outlier_threshold", 0.01);
  plane_slope_threshold_ = declare_parameter("plane_slope_threshold", 10.0);
  height_threshold_ = declare_parameter("height_threshold", 0.01);
  debug_ = declare_parameter("debug", false);

  if (unit_axis_ == "x") {
    unit_vec_ = Eigen::Vector3d::UnitX();
  } else if (unit_axis_ == "y") {
    unit_vec_ = Eigen::Vector3d::UnitY();
  } else if (unit_axis_ == "z") {
    unit_vec_ = Eigen::Vector3d::UnitZ();
  } else {
    unit_vec_ = Eigen::Vector3d::UnitZ();
  }

  if (debug_) {
    setDebugPublisher();
  }

  using std::placeholders::_1;
  set_param_res_ = this->add_on_set_parameters_callback(
    std::bind(&RANSACGroundFilterComponent::paramCallback, this, _1));
}

void RANSACGroundFilterComponent::setDebugPublisher()
{
  if (is_initilized_debug_message_) return;
  debug_pose_array_pub_ = create_publisher<geometry_msgs::msg::PoseArray>("debug/plane_pose_array", max_queue_size_);
  debug_ground_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>("debug/ground/pointcloud", max_queue_size_);
  is_initilized_debug_message_ = true;
}

void RANSACGroundFilterComponent::publishDebugMessage(
  const geometry_msgs::msg::PoseArray & debug_pose_array,
  const std::vector<pcl::PointCloud<PointType>> & ground_clouds, const std_msgs::msg::Header & header)
{
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_ground_ptr(new pcl::PointCloud<pcl::PointXYZRGB>);
  for (int i = 0; i < ground_clouds.size(); ++i) {
    RGB color = getRandColor();
    if (i < color_map_.size()) {
      color = color_map_.at(i);
    }
    for (int j = 0; j < ground_clouds[i].points.size(); ++j) {
      pcl::PointXYZRGB p;
      p.x = ground_clouds[i].points[j].x;
      p.y = ground_clouds[i].points[j].y;
      p.z = ground_clouds[i].points[j].z;
      p.r = color.r;
      p.g = color.g;
      p.b = color.b;
      colored_ground_ptr->points.push_back(p);
    }
  }
  sensor_msgs::msg::PointCloud2::SharedPtr ground_cloud_msg_ptr(new sensor_msgs::msg::PointCloud2);
  ground_cloud_msg_ptr->header = header;
  pcl::toROSMsg(*colored_ground_ptr, *ground_cloud_msg_ptr);
  ground_cloud_msg_ptr->header.frame_id = base_frame_;
  debug_pose_array_pub_->publish(debug_pose_array);
  debug_ground_cloud_pub_->publish(*ground_cloud_msg_ptr);
}

bool RANSACGroundFilterComponent::transformPointCloud(
  const std::string & in_target_frame, const PointCloud2ConstPtr & in_cloud_ptr,
  const PointCloud2::SharedPtr & out_cloud_ptr)
{
  if (in_target_frame == in_cloud_ptr->header.frame_id) {
    *out_cloud_ptr = *in_cloud_ptr;
    return true;
  }

  geometry_msgs::msg::TransformStamped transform_stamped;
  try {
    transform_stamped = tf_buffer_->lookupTransform(
      in_target_frame, in_cloud_ptr->header.frame_id, in_cloud_ptr->header.stamp,
      rclcpp::Duration::from_seconds(1.0));
  } catch (tf2::TransformException & ex) {
    RCLCPP_WARN(this->get_logger(), "%s", ex.what());
    return false;
  }
  // tf2::doTransform(*in_cloud_ptr, *out_cloud_ptr, transform_stamped);
  Eigen::Matrix4f mat = tf2::transformToEigen(transform_stamped.transform).matrix().cast<float>();
  pcl_ros::transformPointCloud(mat, *in_cloud_ptr, *out_cloud_ptr);
  out_cloud_ptr->header.frame_id = in_target_frame;
  return true;
}

void RANSACGroundFilterComponent::extractPointsIndices(
  const pcl::PointCloud<PointType>::Ptr in_cloud_ptr, const pcl::PointIndices & in_indices,
  pcl::PointCloud<PointType>::Ptr out_only_indices_cloud_ptr,
  pcl::PointCloud<PointType>::Ptr out_removed_indices_cloud_ptr)
{
  pcl::ExtractIndices<PointType> extract_ground;
  extract_ground.setInputCloud(in_cloud_ptr);
  extract_ground.setIndices(boost::make_shared<pcl::PointIndices>(in_indices));

  extract_ground.setNegative(false);  // true removes the indices, false leaves only the indices
  extract_ground.filter(*out_only_indices_cloud_ptr);

  extract_ground.setNegative(true);  // true removes the indices, false leaves only the indices
  extract_ground.filter(*out_removed_indices_cloud_ptr);
}

Eigen::Affine3d RANSACGroundFilterComponent::getPlaneAffine(
  const pcl::PointCloud<PointType> segment_ground_cloud, const Eigen::Vector3d & plane_normal)
{
  pcl::CentroidPoint<pcl::PointXYZ> centroid;
  for (const auto p : segment_ground_cloud.points) {
    centroid.add(p);
  }
  pcl::PointXYZ centroid_point;
  centroid.get(centroid_point);
  Eigen::Translation<double, 3> trans(centroid_point.x, centroid_point.y, centroid_point.z);
  const pointcloud_preprocessor::PlaneBasis basis = getPlaneBasis(plane_normal);
  Eigen::Matrix3d rot;
  rot << basis.e_x.x(), basis.e_y.x(), basis.e_z.x(), basis.e_x.y(), basis.e_y.y(), basis.e_z.y(),
    basis.e_x.z(), basis.e_y.z(), basis.e_z.z();
  return trans * rot;
}

void RANSACGroundFilterComponent::applyRecursiveRANSAC(
  const pcl::PointCloud<PointType>::Ptr & input,
  std::vector<pcl::PointIndices::Ptr> & output_inliers,
  std::vector<pcl::ModelCoefficients::Ptr> & output_coefficients,
  pcl::PointCloud<PointType>::Ptr & rest_cloud)
{
  *rest_cloud = *input;
  int counter = 0;
  while (true) {
    ++counter;
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

    pcl::SACSegmentation<PointType> seg;
    seg.setOptimizeCoefficients(true);
    seg.setRadiusLimits(0.3, std::numeric_limits<double>::max());
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setDistanceThreshold(outlier_threshold_);
    seg.setInputCloud(rest_cloud);
    seg.setMaxIterations(max_iterations_);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.segment(*inliers, *coefficients);

    if (inliers->indices.size() >= min_inliers_) {
      output_inliers.push_back(inliers);
      output_coefficients.push_back(coefficients);
    } else {
      if (min_trial_ <= counter) {
        return;
      }
    }

    // prepare for next loop
    pcl::PointCloud<PointType>::Ptr next_rest_cloud(new pcl::PointCloud<PointType>);
    pcl::ExtractIndices<PointType> ex;
    ex.setInputCloud(rest_cloud);
    ex.setIndices(inliers);
    ex.setNegative(true);
    ex.setKeepOrganized(true);
    ex.filter(*next_rest_cloud);
    if (next_rest_cloud->points.size() < min_points_) {
      RCLCPP_DEBUG(this->get_logger(), "no more enough points are left");
      return;
    }
    rest_cloud = next_rest_cloud;
  }
}

void RANSACGroundFilterComponent::filter(
  const PointCloud2::ConstSharedPtr & input, const IndicesPtr & indices, PointCloud2 & output)
{
  boost::mutex::scoped_lock lock(mutex_);
  sensor_msgs::msg::PointCloud2::SharedPtr input_transed_ptr(new sensor_msgs::msg::PointCloud2);
  if (!transformPointCloud(base_frame_, input, input_transed_ptr)) {
    RCLCPP_ERROR_STREAM_THROTTLE(
      this->get_logger(), *this->get_clock(), std::chrono::milliseconds(1000).count(),
      "Failed transform from " << base_frame_ << " to " << input->header.frame_id);
    return;
  }
  pcl::PointCloud<PointType>::Ptr current_sensor_cloud_ptr(new pcl::PointCloud<PointType>);
  pcl::fromROSMsg(*input_transed_ptr, *current_sensor_cloud_ptr);

  std::vector<pcl::PointIndices::Ptr> inliers_vector;
  std::vector<pcl::ModelCoefficients::Ptr> coefficients;
  pcl::PointCloud<PointType>::Ptr rest_cloud_ptr(new pcl::PointCloud<PointType>);
  applyRecursiveRANSAC(current_sensor_cloud_ptr, inliers_vector, coefficients, rest_cloud_ptr);

  geometry_msgs::msg::PoseArray debug_pose_array;
  debug_pose_array.header.frame_id = base_frame_;
  std::vector<pcl::PointCloud<PointType>> debug_ground_clouds;
  pcl::PointCloud<PointType>::Ptr no_ground_cloud_ptr(new pcl::PointCloud<PointType>);
  for (int i = 0; i < inliers_vector.size(); ++i) {
    Eigen::Vector3d plane_normal(
      coefficients.at(i)->values[0], coefficients.at(i)->values[1], coefficients.at(i)->values[2]);
    {
      const auto plane_slope = std::abs(
        std::acos(plane_normal.dot(unit_vec_) / (plane_normal.norm() * unit_vec_.norm())) * 180 /
        M_PI);
      if (plane_slope > plane_slope_threshold_) {
        continue;
      }
    }

    pcl::PointCloud<PointType>::Ptr segment_ground_cloud_ptr(new pcl::PointCloud<PointType>);
    pcl::PointCloud<PointType>::Ptr segment_no_ground_cloud_ptr(new pcl::PointCloud<PointType>);
    extractPointsIndices(
      current_sensor_cloud_ptr, *inliers_vector[i], segment_ground_cloud_ptr,
      segment_no_ground_cloud_ptr);
    const Eigen::Affine3d plane_affine = getPlaneAffine(*segment_ground_cloud_ptr, plane_normal);
    const geometry_msgs::msg::Pose debug_pose = getDebugPose(plane_affine);
    debug_pose_array.poses.push_back(debug_pose);
    debug_ground_clouds.push_back(*segment_ground_cloud_ptr);

    for (const auto & p : rest_cloud_ptr->points) {
      const Eigen::Vector3d transformed_point =
        plane_affine.inverse() * Eigen::Vector3d(p.x, p.y, p.z);
      if (std::abs(transformed_point.z()) > height_threshold_) {
        no_ground_cloud_ptr->points.push_back(p);
      }
    }
  }

  sensor_msgs::msg::PointCloud2::SharedPtr no_ground_cloud_msg_ptr(new sensor_msgs::msg::PointCloud2);
  pcl::toROSMsg(*no_ground_cloud_ptr, *no_ground_cloud_msg_ptr);
  no_ground_cloud_msg_ptr->header = input->header;
  sensor_msgs::msg::PointCloud2::SharedPtr no_ground_cloud_transed_msg_ptr(new sensor_msgs::msg::PointCloud2);
  if (!transformPointCloud(base_frame_, no_ground_cloud_msg_ptr, no_ground_cloud_transed_msg_ptr)) {
    RCLCPP_ERROR_STREAM_THROTTLE(
      this->get_logger(), *this->get_clock(), std::chrono::milliseconds(1000).count(),
      "Failed transform from " << base_frame_ << " to " <<
        no_ground_cloud_msg_ptr->header.frame_id);
    return;
  }
  output = *no_ground_cloud_transed_msg_ptr;

  if (debug_) {
    publishDebugMessage(debug_pose_array, debug_ground_clouds, input->header);
  }
}

rcl_interfaces::msg::SetParametersResult RANSACGroundFilterComponent::paramCallback(
  const std::vector<rclcpp::Parameter> & p)
{
  boost::mutex::scoped_lock lock(mutex_);

  if (get_param(p, "base_frame", base_frame_)) {
    RCLCPP_DEBUG(get_logger(), "Setting base_frame to: %s.", base_frame_.c_str());
  }
  if (get_param(p, "min_trial", min_trial_)) {
    RCLCPP_DEBUG(get_logger(), "Setting min_trial to: %d.", min_trial_);
  }
  if (get_param(p, "max_iterations", max_iterations_)) {
    RCLCPP_DEBUG(get_logger(), "Setting max_iterations to: %d.", max_iterations_);
  }
  if (get_param(p, "min_inliers", min_inliers_)) {
    RCLCPP_DEBUG(get_logger(), "Setting min_inliers to: %d.", min_inliers_);
  }
  if (get_param(p, "min_points", min_points_)) {
    RCLCPP_DEBUG(get_logger(), "Setting min_points to: %d.", min_points_);
  }
  if (get_param(p, "outlier_threshold", outlier_threshold_)) {
    RCLCPP_DEBUG(
      get_logger(), "Setting outlier_threshold to: %lf.", outlier_threshold_);
  }
  if (get_param(p, "height_threshold", height_threshold_)) {
    RCLCPP_DEBUG(get_logger(), "Setting height_threshold_ to: %lf.", height_threshold_);
  }
  if (get_param(p, "plane_slope_threshold", plane_slope_threshold_)) {
    RCLCPP_DEBUG(
      get_logger(), "Setting plane_slope_threshold to: %lf.", plane_slope_threshold_);
  }
  if (get_param(p, "debug", debug_)) {
    RCLCPP_DEBUG(
      get_logger(), "Setting debug to: %d.", debug_);
  }

  if (debug_) {
    setDebugPublisher();
  }

  if (get_param(p, "unit_axis", unit_axis_)) {
    if (unit_axis_ == "x") {
      unit_vec_ = Eigen::Vector3d::UnitX();
    } else if (unit_axis_ == "y") {
      unit_vec_ = Eigen::Vector3d::UnitY();
    } else if (unit_axis_ == "z") {
      unit_vec_ = Eigen::Vector3d::UnitZ();
    } else {
      unit_vec_ = Eigen::Vector3d::UnitZ();
    }
    RCLCPP_DEBUG(
      get_logger(), "Setting unit_axis to: %s.", unit_axis_.c_str());
  }

  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  result.reason = "success";

  return result;
}

}  // namespace pointcloud_preprocessor

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(pointcloud_preprocessor::RANSACGroundFilterComponent)
