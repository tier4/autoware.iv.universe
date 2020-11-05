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
#pragma once

#include <autoware_perception_msgs/msg/lamp_state.hpp>
#include <autoware_perception_msgs/msg/traffic_light_roi_array.hpp>
#include <autoware_perception_msgs/msg/traffic_light_state.hpp>
#include <autoware_perception_msgs/msg/traffic_light_state_array.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <message_filters/time_synchronizer.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Header.h>

#include <traffic_light_classifier/classifier_interface.hpp>

#if ENABLE_GPU
#include <traffic_light_classifier/cnn_classifier.hpp>
#endif

#include <traffic_light_classifier/color_classifier.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <memory>
#include <mutex>

namespace traffic_light
{
class TrafficLightClassifierNodelet : public nodelet::Nodelet
{
public:
  virtual void onInit();
  void connectCb();
  void imageRoiCallback(
    const sensor_msgs::ImageConstPtr & input_image_msg,
    const autoware_perception_msgs::TrafficLightRoiArrayConstPtr & input_rois_msg);

private:
  enum ClassifierType {
    HSVFilter = 0,
    CNN = 1,
  };

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  std::shared_ptr<image_transport::ImageTransport> image_transport_;
  image_transport::SubscriberFilter image_sub_;
  message_filters::Subscriber<autoware_perception_msgs::TrafficLightRoiArray> roi_sub_;
  typedef message_filters::sync_policies::ExactTime<
    sensor_msgs::Image, autoware_perception_msgs::TrafficLightRoiArray>
    SyncPolicy;
  typedef message_filters::Synchronizer<SyncPolicy> Sync;
  std::shared_ptr<Sync> sync_;
  typedef message_filters::sync_policies::ApproximateTime<
    sensor_msgs::Image, autoware_perception_msgs::TrafficLightRoiArray>
    ApproximateSyncPolicy;
  typedef message_filters::Synchronizer<ApproximateSyncPolicy> ApproximateSync;
  std::shared_ptr<ApproximateSync> approximate_sync_;
  bool is_approximate_sync_;
  std::mutex connect_mutex_;
  ros::Publisher tl_states_pub_;
  std::shared_ptr<ClassifierInterface> classifier_ptr_;
};

}  // namespace traffic_light
