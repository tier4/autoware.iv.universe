/*
 * Copyright 2015-2019 Autoware Foundation. All rights reserved.
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

#ifndef GYRO_ODOMETER_GYRO_ODOMETER_CORE_H_

#include <geometry_msgs/msg/twist_stamped.hpp>
#include <geometry_msgs/msg/twist_with_covariance_stamped.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/float32.hpp>

#include <tf2/transform_datatypes.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/transform_listener.h>

#include <rclcpp/rclcpp.hpp>


class GyroOdometer : public rclcpp::Node
{
public:
  GyroOdometer();
  ~GyroOdometer();

private:
  void callbackTwist(const geometry_msgs::msg::TwistStamped::ConstSharedPtr twist_msg_ptr);
  void callbackImu(const sensor_msgs::msg::Imu::ConstSharedPtr imu_msg_ptr);
  bool getTransform(
    const std::string & target_frame, const std::string & source_frame,
    const geometry_msgs::msg::TransformStamped::SharedPtr transform_stamped_ptr);

  // ros::NodeHandle nh_;
  // ros::NodeHandle private_nh_;

  // ros::Subscriber vehicle_twist_sub_;
  // ros::Subscriber imu_sub_;

  // ros::Publisher twist_pub_;
  // ros::Publisher twist_with_covariance_pub_;
  // ros::Publisher linear_x_pub_;
  // ros::Publisher angular_z_pub_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  std::string output_frame_;
  geometry_msgs::msg::TwistStamped::ConstSharedPtr twist_msg_ptr_;
};

#endif
