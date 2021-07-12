/*
 * Copyright 2021 Tier IV, inc. All rights reserved.
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
 *
 * This file includes works by PCL.
 *
 * ======== ORIGINAL LICENSE AND COPYRIGHTS BELOW ========
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
 *  Copyright (c) 2012-, Open Perception, Inc.
 *
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
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
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
 *
 * $Id$
 *
 */

#include "ndt_ocl.h"
#include <ros/package.h>

#ifndef PCL_REGISTRATION_NDT_OCL_IMPL_H_
#define PCL_REGISTRATION_NDT_OCL_IMPL_H_

#define KERNEL_FILE "ocl/compute_derivatives.cl"
#define KERNEL_NAME "computeDerivativesCL"

#define OCL_CREATE_BUFFER_CHECK(ret_mem, context, flags, size, host_ptr, errcode_ret) \
  ret_mem = clCreateBuffer(context, flags, size, host_ptr, &errcode_ret); \
  if (errcode_ret != CL_SUCCESS) {                                      \
    return -1;                                                          \
  }
#define OCL_WRITE_BUFFER_CHECK(command_queue, buffer, blocking_write, offset, cb, ptr, num_events_in_wait_list, event_wait_list, event) \
  if (clEnqueueWriteBuffer(command_queue, buffer, blocking_write, offset, cb, ptr, num_events_in_wait_list, event_wait_list, event) != CL_SUCCESS) { \
    return -1;                                                          \
  }
#define OCL_SET_KERNEL_ARG_CHECK(kernel, arg_index, arg_size, arg_value) \
  if (clSetKernelArg(kernel, arg_index, arg_size, arg_value) != CL_SUCCESS) { \
    return -1;                                                          \
  }
#define OCL_SET_KERNEL_ARG_SVM_CHECK(kernel, arg_index, arg_value)      \
  if (clSetKernelArgSVMPointer(kernel, arg_index, arg_value) != CL_SUCCESS) { \
    return -1;                                                          \
  }
#define OCL_READ_BUFFER_CHECK(command_queue, buffer, blocking_read, offset, cb, ptr, num_events_in_wait_list, event_wait_list, event) \
  if (clEnqueueReadBuffer(command_queue, buffer, blocking_read, offset, cb, ptr, num_events_in_wait_list, event_wait_list, event) != CL_SUCCESS) { \
    return -1;                                                          \
  }
#define OCL_RELEASE_MEMORY_CHECK(mem)           \
  if (mem != NULL) {                            \
    clReleaseMemObject(mem);                    \
    mem = NULL;                                 \
  }
#define OCL_RELEASE_KERNEL_CHECK(kernel)        \
  if (kernel != NULL) {                         \
    clReleaseKernel(kernel);                    \
    kernel = NULL;                              \
  }
#define OCL_RELEASE_QUEUE_CHECK(queue)          \
  if (queue != NULL) {                          \
    clReleaseCommandQueue(queue);               \
    queue = NULL;                               \
  }
#define OCL_RELEASE_PROGRAM_CHECK(program)      \
  if (program != NULL) {                        \
    clReleaseProgram(program);                  \
    program = NULL;                             \
  }
#define OCL_RELEASE_CONTEXT_CHECK(context)      \
  if (context != NULL) {                        \
    clReleaseContext(context);                  \
    context = NULL;                             \
  }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointSource, typename PointTarget>
ndt_ocl::NormalDistributionsTransform<PointSource, PointTarget>::NormalDistributionsTransform()
  : target_cells_(),
    resolution_(1.0f),
    step_size_(0.1),
    outlier_ratio_(0.55),
    gauss_d1_(),
    gauss_d2_(),
    gauss_d3_(),
    trans_probability_(),
    j_ang_a_(),
    j_ang_b_(),
    j_ang_c_(),
    j_ang_d_(),
    j_ang_e_(),
    j_ang_f_(),
    j_ang_g_(),
    j_ang_h_(),
    h_ang_a2_(),
    h_ang_a3_(),
    h_ang_b2_(),
    h_ang_b3_(),
    h_ang_c2_(),
    h_ang_c3_(),
    h_ang_d1_(),
    h_ang_d2_(),
    h_ang_d3_(),
    h_ang_e1_(),
    h_ang_e2_(),
    h_ang_e3_(),
    h_ang_f1_(),
    h_ang_f2_(),
    h_ang_f3_(),
    limit_(LIMIT_NUM)
{
  reg_name_ = "NormalDistributionsTransform";

  double gauss_c1, gauss_c2;

  // Initializes the guassian fitting parameters (eq. 6.8) [Magnusson 2009]
  gauss_c1 = 10.0 * (1 - outlier_ratio_);
  gauss_c2 = outlier_ratio_ / pow(resolution_, 3);
  gauss_d3_ = -log(gauss_c2);
  gauss_d1_ = -log(gauss_c1 + gauss_c2) - gauss_d3_;
  gauss_d2_ = -2 * log((-log(gauss_c1 * exp(-0.5) + gauss_c2) - gauss_d3_) / gauss_d1_);

  transformation_epsilon_ = 0.1;
  max_iterations_ = 35;

  Queue_ = NULL;
  k_compute_derivatives_ = NULL;
  context_ = NULL;
  program_ = NULL;
  d_query_x_ = NULL;
  d_query_y_ = NULL;
  d_query_z_ = NULL;
  d_inputs_x_ = NULL;
  d_inputs_y_ = NULL;
  d_inputs_z_ = NULL;
  d_j_ang_ = NULL;
  d_h_ang_ = NULL;
  d_neighbor_candidates_ = NULL;
  d_neighbor_candidate_dists_ = NULL;
  d_scores_ = NULL;
  d_score_gradients_ = NULL;
  d_hessians_ = NULL;

  // initialize OpenCL
  cl_uint platform = 0;
  cl_uint device = 0;
  int ret = initializeOCL(platform, device);
  if (ret != 0) {
    std::cerr << "error : initializing OpenCL failed." << std::endl;
    finalizeOCL();
  }
  target_cells_.setContext(context_);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointSource, typename PointTarget>
void ndt_ocl::NormalDistributionsTransform<PointSource, PointTarget>::computeTransformation(
  PointCloudSource & output, const Eigen::Matrix4f & guess)
{
  nr_iterations_ = 0;
  converged_ = false;

  double gauss_c1, gauss_c2;

  // Initializes the guassian fitting parameters (eq. 6.8) [Magnusson 2009]
  gauss_c1 = 10 * (1 - outlier_ratio_);
  gauss_c2 = outlier_ratio_ / pow(resolution_, 3);
  gauss_d3_ = -log(gauss_c2);
  gauss_d1_ = -log(gauss_c1 + gauss_c2) - gauss_d3_;
  gauss_d2_ = -2 * log((-log(gauss_c1 * exp(-0.5) + gauss_c2) - gauss_d3_) / gauss_d1_);

  if (guess != Eigen::Matrix4f::Identity()) {
    // Initialise final transformation to the guessed one
    final_transformation_ = guess;
    // Apply guessed transformation prior to search for neighbours
    transformPointCloud(output, output, guess);
  }

  Eigen::Transform<float, 3, Eigen::Affine, Eigen::ColMajor> eig_transformation;
  eig_transformation.matrix() = final_transformation_;
  transformation_array_.clear();
  transformation_array_.push_back(final_transformation_);

  // Convert initial guess matrix to 6 element transformation vector
  Eigen::Matrix<double, 6, 1> p, delta_p, score_gradient;
  Eigen::Vector3f init_translation = eig_transformation.translation();
  Eigen::Vector3f init_rotation = eig_transformation.rotation().eulerAngles(0, 1, 2);
  p << init_translation(0), init_translation(1), init_translation(2), init_rotation(0),
    init_rotation(1), init_rotation(2);

  Eigen::Matrix<double, 6, 6> hessian;

  double score = 0;
  double delta_p_norm;

  // Calculate derivates of initial transform vector, subsequent derivative calculations are done in the step length
  // determination.
  score = computeDerivatives(score_gradient, hessian, output, p);
  bool converged_rotation = false;
  while (!converged_) {
    // Store previous transformation
    previous_transformation_ = transformation_;

    // Solve for decent direction using newton method, line 23 in Algorithm 2 [Magnusson 2009]
    Eigen::JacobiSVD<Eigen::Matrix<double, 6, 6>> sv(
                                                     hessian, Eigen::ComputeFullU | Eigen::ComputeFullV);
    // Negative for maximization as opposed to minimization
    delta_p = sv.solve(-score_gradient);

    // Calculate step length with guarnteed sufficient decrease [More, Thuente 1994]
    delta_p_norm = delta_p.norm();

    if (delta_p_norm == 0 || delta_p_norm != delta_p_norm) {
      trans_probability_ = score / static_cast<double>(input_->points.size());
      converged_ = delta_p_norm == delta_p_norm;
      return;
    }

    // Eigen::Matrix<double, 6, 1> delta_p_rotation = delta_p;
    // delta_p_rotation(0) = delta_p_rotation(1) = delta_p_rotation(2) = 0;
    // double delta_p_rotation_norm = delta_p_rotation.norm ();
    //
    // Eigen::Matrix<double, 6, 1> score_gradient_rotation = score_gradient;
    // score_gradient_rotation(0) = score_gradient_rotation(1) = score_gradient_rotation(2) = 0;

    delta_p.normalize();
    // delta_p_rotation.normalize();

    // if(!converged_rotation && delta_p_rotation_norm > 0.001 && nr_iterations_ < 10)
    // {
    //   delta_p = delta_p_rotation;
    //   delta_p_norm = delta_p_rotation_norm;
    //   score_gradient = score_gradient_rotation;
    //   step_size_ = 0.01;
    //   transformation_epsilon_ = 0.001;
    //   delta_p_norm = computeStepLengthMT (p, delta_p, delta_p_norm, step_size_, transformation_epsilon_ / 2, score,
    //   score_gradient, hessian, output);
    // }
    // else
    // {
    //   converged_rotation = true;
    //   transformation_epsilon_ = 0.01;
    //   step_size_ = 0.1;
    delta_p_norm = computeStepLengthMT(
                                       p, delta_p, delta_p_norm, step_size_, transformation_epsilon_ / 2, score, score_gradient,
                                       hessian, output);
    // }

    delta_p *= delta_p_norm;

    transformation_ =
      (Eigen::Translation<float, 3>(
                                    static_cast<float>(delta_p(0)), static_cast<float>(delta_p(1)),
                                    static_cast<float>(delta_p(2))) *
       Eigen::AngleAxis<float>(static_cast<float>(delta_p(3)), Eigen::Vector3f::UnitX()) *
       Eigen::AngleAxis<float>(static_cast<float>(delta_p(4)), Eigen::Vector3f::UnitY()) *
       Eigen::AngleAxis<float>(static_cast<float>(delta_p(5)), Eigen::Vector3f::UnitZ()))
      .matrix();

    transformation_array_.push_back(final_transformation_);

    p = p + delta_p;

    // Update Visualizer (untested)
    if (update_visualizer_ != 0)
      update_visualizer_(output, std::vector<int>(), *target_, std::vector<int>());

    if (
        nr_iterations_ > max_iterations_ ||
        (nr_iterations_ && (std::fabs(delta_p_norm) < transformation_epsilon_))) {
      converged_ = true;
    }

    nr_iterations_++;
  }

  // Store transformation probability.  The realtive differences within each scan registration are accurate
  // but the normalization constants need to be modified for it to be globally accurate
  trans_probability_ = score / static_cast<double>(input_->points.size());

  hessian_ = hessian;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointSource, typename PointTarget>
double ndt_ocl::NormalDistributionsTransform<PointSource, PointTarget>::computeDerivatives(
  Eigen::Matrix<double, 6, 1> & score_gradient, Eigen::Matrix<double, 6, 6> & hessian,
  PointCloudSource & trans_cloud, Eigen::Matrix<double, 6, 1> & p, bool compute_hessian)
{
  score_gradient.setZero();
  hessian.setZero();
  double score = 0;

  n_query_ = trans_cloud.size();
  if (n_query_ > MAX_PCL_INPUT_NUM) {
    std::cerr << "error : query points is too large." << std::endl;
    return -1.0;
  }
  for (int i = 0; i < n_query_; i++) {
    scores_[i] = 0.0f;
    for (int j = 0; j < 6; j++) {
      for (int k = 0; k < 1; k++) {
        score_gradients_[i][j][k] = 0.0f;
      }
    }
    for (int j = 0; j < 6; j++) {
      for (int k = 0; k < 6; k++) {
        hessians_[i][j][k] = 0.0f;
      }
    }
  }

  for (int i = 0; i < n_query_; i++) {
    query_points_x_[i] = trans_cloud.points[i].x;
    query_points_y_[i] = trans_cloud.points[i].y;
    query_points_z_[i] = trans_cloud.points[i].z;
  }

  if (input_->points.size() > MAX_PCL_INPUT_NUM) {
    std::cerr << "error : input_ points is too large." << std::endl;
    return -1.0;
  }
  for (int i = 0; i < input_->points.size(); i++) {
    input_points_x_[i] = input_->points[i].x;
    input_points_y_[i] = input_->points[i].y;
    input_points_z_[i] = input_->points[i].z;
  }

  gauss_d1_f_ = gauss_d1_;
  gauss_d2_f_ = gauss_d2_;
  gauss_d3_f_ = gauss_d3_;

  // Precompute Angular Derivatives (eq. 6.19 and 6.21)[Magnusson 2009]
  computeAngleDerivatives(p);

  // Update gradient and hessian for each point, line 17 in Algorithm 2 [Magnusson 2009]

  // parallel in OpenCL
  int ret;
  ret = createOCLMemoryObjects();
  if (ret != 0) {
    std::cerr << "error : failed to create OpenCL memory objects" << std::endl;
  }
  else {
    ret =copyToOCLMemoryObjects();
  }
  if (ret != 0) {
    std::cerr << "error : failed to copy to OpenCL memory objects" << std::endl;
  }
  else {
    ret = computeDerivativesCL();
  }
  if (ret != 0) {
    std::cerr << "error : failed to call computeDerivativesCL" << std::endl;
  }
  else {
    ret = readOCLMemoryObjects();
  }
  if (ret != 0) {
    std::cerr << "error : failed to read OpenCL memory objects" << std::endl;
  }
  releaseOCLMemoryObjects();

  for (int i = 0; i < n_query_; i++) {
    score += scores_[i];
    for (int j = 0; j < 6; j++) {
      for (int k = 0; k < 1; k++) {
        score_gradient(j,k) += score_gradients_[i][j][k];
      }
    }
    for (int j = 0; j < 6; j++) {
      for (int k = 0; k < 6; k++) {
        hessian(j,k) += hessians_[i][j][k];
      }
    }
  }

  return (score);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointSource, typename PointTarget>
int ndt_ocl::NormalDistributionsTransform<PointSource, PointTarget>::initializeOCL(cl_uint platform, cl_uint device)
{
  size_t source_size, ret_size;
  cl_uint num_platforms, num_devices;
  cl_int ret;

  // platform
  ret = clGetPlatformIDs(MAX_PLATFORMS, platform_id_, &num_platforms);
  if (ret != CL_SUCCESS) {
    std::cerr << "error : clGetPlatformIDs() error " << ret << std::endl;
    return 1;
  }
  if (platform >= num_platforms) {
    std::cerr << "error : platform = " << platform << "(limit = " << num_platforms - 1 << ")" << std::endl;
    return 1;
  }

  // device
  ret = clGetDeviceIDs(platform_id_[platform], CL_DEVICE_TYPE_ALL, MAX_DEVICES, device_id_, &num_devices);
  if (ret != CL_SUCCESS) {
    std::cerr << "error : clGetDeviceIDs() error " << ret << std::endl;
    return 1;
  }
  if (device >= num_devices) {
    std::cerr << "error : device = " << device << "(limit = " << num_devices - 1 << ")" << std::endl;
    return 1;
  }

  // device name (optional information)
  {
    char str[BUFSIZ];
    ret = clGetDeviceInfo(device_id_[device], CL_DEVICE_NAME, sizeof(str), str, &ret_size);
    if (ret != CL_SUCCESS) {
      std::cerr << "error : clGetDeviceInfo() error " << ret << std::endl;
      return 1;
    }
    std::cout << "info : " << str << " (platform = " << platform << ", device = " << device << ")" << std::endl;
  }

  // context
  context_ = clCreateContext(NULL, 1, &device_id_[device], NULL, NULL, &ret);
  if (ret != CL_SUCCESS) {
    std::cerr << "error : clCreateContext() error " << ret << std::endl;
    return 1;
  }

  // command queue
  Queue_ = clCreateCommandQueue(context_, device_id_[device], 0, &ret);
  if (ret != CL_SUCCESS) {
    std::cerr << "error : clCreateCommandQueue() error " << ret << std::endl;
    return 1;
  }

  // source
  {
    FILE * fp;
    std::string kernel_file_path_ = ros::package::getPath("ndt_scan_matcher") + "/" + KERNEL_FILE;
    if ((fp = fopen(kernel_file_path_.c_str(), "r")) == NULL) {
      std::cerr << "error : could not open " << kernel_file_path_ << std::endl;
      return 1;
    }
    // alloc
    char source_str[MAX_SOURCE_SIZE];
    char * source_ptr = source_str;
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    // create program
    program_ = clCreateProgramWithSource(context_, 1, (const char **)&source_ptr, (const size_t *)&source_size, &ret);
    fclose(fp);
    if (ret != CL_SUCCESS) {
      std::cerr << "error : clCreateProgramWithSource() error " << ret << std::endl;
      return 1;
    }
  }

  // build program
  if (clBuildProgram(program_, 1, &device_id_[device], NULL, NULL, NULL) != CL_SUCCESS) {
    std::cerr << "error : clBuildProgram() error" << std::endl;
    size_t logSize;
    clGetProgramBuildInfo(program_, device_id_[device], CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
    std::unique_ptr<char[]> buildLog(new char[logSize + 1]);
    clGetProgramBuildInfo(program_, device_id_[device], CL_PROGRAM_BUILD_LOG, logSize, buildLog.get(), NULL);
    std::cout << buildLog.get() << std::endl;
    return 1;
  }

  // kernel
  k_compute_derivatives_ = clCreateKernel(program_, KERNEL_NAME, &ret);
  if (ret != CL_SUCCESS) {
    std::cerr << "error : clCreateKernel() error" << std::endl;
    return 1;
  }
  return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointSource, typename PointTarget>
int ndt_ocl::NormalDistributionsTransform<PointSource, PointTarget>::createOCLMemoryObjects(void)
{
  if (n_query_ > MAX_PCL_INPUT_NUM) {
    return -1;
  }

  size_t source_size, ret_size, points_size, map_points_size;
  const size_t j_ang_size = 8 * 4 * sizeof(float);
  const size_t h_ang_size = 16 * 4 * sizeof(float);
  cl_uint num_platforms, num_devices;
  cl_int ret;

  // memory object
  points_size = n_query_ * sizeof(float);
  neighbor_candidates_size_float_ = limit_ * n_query_ * sizeof(float);
  neighbor_candidates_size_int_ = limit_ * n_query_ * sizeof(int);
  score_gradients_size_ = n_query_ * 6 * 1 * sizeof(float);
  hessians_size_ = n_query_ * 6 * 6 * sizeof(float);
  score_size_ = points_size;

  OCL_CREATE_BUFFER_CHECK(d_query_x_, context_, CL_MEM_READ_WRITE, points_size, NULL, ret);
  OCL_CREATE_BUFFER_CHECK(d_query_y_, context_, CL_MEM_READ_WRITE, points_size, NULL, ret);
  OCL_CREATE_BUFFER_CHECK(d_query_z_, context_, CL_MEM_READ_WRITE, points_size, NULL, ret);
  OCL_CREATE_BUFFER_CHECK(d_inputs_x_, context_, CL_MEM_READ_WRITE, points_size, NULL, ret);
  OCL_CREATE_BUFFER_CHECK(d_inputs_y_, context_, CL_MEM_READ_WRITE, points_size, NULL, ret);
  OCL_CREATE_BUFFER_CHECK(d_inputs_z_, context_, CL_MEM_READ_WRITE, points_size, NULL, ret);
  OCL_CREATE_BUFFER_CHECK(d_neighbor_candidates_, context_, CL_MEM_READ_WRITE, neighbor_candidates_size_int_, NULL, ret);
  OCL_CREATE_BUFFER_CHECK(d_neighbor_candidate_dists_, context_, CL_MEM_READ_WRITE, neighbor_candidates_size_float_, NULL, ret);
  OCL_CREATE_BUFFER_CHECK(d_j_ang_, context_, CL_MEM_READ_WRITE, j_ang_size, NULL, ret);
  OCL_CREATE_BUFFER_CHECK(d_h_ang_, context_, CL_MEM_READ_WRITE, h_ang_size, NULL, ret);
  OCL_CREATE_BUFFER_CHECK(d_scores_, context_, CL_MEM_READ_WRITE, score_size_, NULL, ret);
  OCL_CREATE_BUFFER_CHECK(d_score_gradients_, context_, CL_MEM_READ_WRITE, score_gradients_size_, NULL, ret);
  OCL_CREATE_BUFFER_CHECK(d_hessians_, context_, CL_MEM_READ_WRITE, hessians_size_, NULL, ret);

  return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointSource, typename PointTarget>
int ndt_ocl::NormalDistributionsTransform<PointSource, PointTarget>::copyToOCLMemoryObjects()
{
  if (n_query_ > MAX_PCL_INPUT_NUM) {
    return -1;
  }

  size_t points_size, map_points_size, map_size, points_size_int, map_mean_size, map_inverse_cov_size;
  const size_t j_ang_size = 8 * 4 * sizeof(float);
  const size_t h_ang_size = 16 * 4 * sizeof(float);

  points_size = n_query_ * sizeof(float);
  map_size = target_cells_.num_centroids_ * sizeof(float);
  map_mean_size = target_cells_.num_centroids_ * 3 * sizeof(float);
  map_inverse_cov_size = target_cells_.num_centroids_ * 3 * 3 * sizeof(float);
  points_size_int = target_cells_.num_centroids_ * sizeof(int);
  score_gradients_size_ = n_query_ * 6 * 1 * sizeof(float);
  hessians_size_ = n_query_ * 6 * 6 * sizeof(float);
  for (int i=0; i < 8; i++) {
    for (int j=0; j < 4; j++) {
      j_ang_array_[i][j] = j_ang(i, j);
    }
  }
  for (int i=0; i < 16; i++) {
    for (int j=0; j < 4; j++) {
      h_ang_array_[i][j] = h_ang(i, j);
    }
  }

  OCL_WRITE_BUFFER_CHECK(Queue_, d_query_x_, CL_TRUE, 0, points_size, query_points_x_, 0, NULL, NULL);
  OCL_WRITE_BUFFER_CHECK(Queue_, d_query_y_, CL_TRUE, 0, points_size, query_points_y_, 0, NULL, NULL);
  OCL_WRITE_BUFFER_CHECK(Queue_, d_query_z_, CL_TRUE, 0, points_size, query_points_z_, 0, NULL, NULL);
  OCL_WRITE_BUFFER_CHECK(Queue_, target_cells_.d_map_points_x_, CL_TRUE, 0, map_size, target_cells_.map_points_x_, 0, NULL, NULL);
  OCL_WRITE_BUFFER_CHECK(Queue_, target_cells_.d_map_points_y_, CL_TRUE, 0, map_size, target_cells_.map_points_y_, 0, NULL, NULL);
  OCL_WRITE_BUFFER_CHECK(Queue_, target_cells_.d_map_points_z_, CL_TRUE, 0, map_size, target_cells_.map_points_z_, 0, NULL, NULL);
  OCL_WRITE_BUFFER_CHECK(Queue_, target_cells_.d_node_indexes_, CL_TRUE, 0, points_size_int, target_cells_.node_indexes_, 0, NULL, NULL);
  OCL_WRITE_BUFFER_CHECK(Queue_, target_cells_.d_map_mean_, CL_TRUE, 0, map_mean_size, target_cells_.map_mean_, 0, NULL, NULL);
  OCL_WRITE_BUFFER_CHECK(Queue_, target_cells_.d_map_inverse_cov_, CL_TRUE, 0, map_inverse_cov_size, target_cells_.map_inverse_cov_, 0, NULL, NULL);
  OCL_WRITE_BUFFER_CHECK(Queue_, d_inputs_x_, CL_TRUE, 0, points_size, input_points_x_, 0, NULL, NULL);
  OCL_WRITE_BUFFER_CHECK(Queue_, d_inputs_y_, CL_TRUE, 0, points_size, input_points_y_, 0, NULL, NULL);
  OCL_WRITE_BUFFER_CHECK(Queue_, d_inputs_z_, CL_TRUE, 0, points_size, input_points_z_, 0, NULL, NULL);
  OCL_WRITE_BUFFER_CHECK(Queue_, d_j_ang_, CL_TRUE, 0, j_ang_size, j_ang_array_, 0, NULL, NULL);
  OCL_WRITE_BUFFER_CHECK(Queue_, d_h_ang_, CL_TRUE, 0, h_ang_size, h_ang_array_, 0, NULL, NULL);
  OCL_WRITE_BUFFER_CHECK(Queue_, d_scores_, CL_TRUE, 0, score_size_, scores_, 0, NULL, NULL);
  OCL_WRITE_BUFFER_CHECK(Queue_, d_score_gradients_, CL_TRUE, 0, score_gradients_size_, score_gradients_, 0, NULL, NULL);
  OCL_WRITE_BUFFER_CHECK(Queue_, d_hessians_, CL_TRUE, 0, hessians_size_, hessians_, 0, NULL, NULL);

  return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointSource, typename PointTarget>
int ndt_ocl::NormalDistributionsTransform<PointSource, PointTarget>::computeDerivativesCL(void)
{
  const size_t local_item_size = 256;
  size_t global_item_size;
  cl_int ret;

  // set arguments
  OCL_SET_KERNEL_ARG_CHECK(k_compute_derivatives_, 0, sizeof(cl_mem), (void *)&d_query_x_);
  OCL_SET_KERNEL_ARG_CHECK(k_compute_derivatives_, 1, sizeof(cl_mem), (void *)&d_query_y_);
  OCL_SET_KERNEL_ARG_CHECK(k_compute_derivatives_, 2, sizeof(cl_mem), (void *)&d_query_z_);
  OCL_SET_KERNEL_ARG_CHECK(k_compute_derivatives_, 3, sizeof(cl_mem), (void *)&target_cells_.d_map_points_x_);
  OCL_SET_KERNEL_ARG_CHECK(k_compute_derivatives_, 4, sizeof(cl_mem), (void *)&target_cells_.d_map_points_y_);
  OCL_SET_KERNEL_ARG_CHECK(k_compute_derivatives_, 5, sizeof(cl_mem), (void *)&target_cells_.d_map_points_z_);
  OCL_SET_KERNEL_ARG_CHECK(k_compute_derivatives_, 6, sizeof(cl_mem), (void *)&target_cells_.d_node_indexes_);
  OCL_SET_KERNEL_ARG_SVM_CHECK(k_compute_derivatives_, 7, target_cells_.kdtree_root_);
  OCL_SET_KERNEL_ARG_CHECK(k_compute_derivatives_, 8, sizeof(int), (void *)&n_query_);
  OCL_SET_KERNEL_ARG_CHECK(k_compute_derivatives_, 9, sizeof(int), (void *)&limit_);
  OCL_SET_KERNEL_ARG_CHECK(k_compute_derivatives_, 10, sizeof(float), (void *)&resolution_);
  OCL_SET_KERNEL_ARG_CHECK(k_compute_derivatives_, 11, sizeof(cl_mem), (void *)&d_neighbor_candidates_);
  OCL_SET_KERNEL_ARG_CHECK(k_compute_derivatives_, 12, sizeof(cl_mem), (void *)&d_neighbor_candidate_dists_);
  OCL_SET_KERNEL_ARG_CHECK(k_compute_derivatives_, 13, sizeof(cl_mem), (void *)&target_cells_.d_map_mean_);
  OCL_SET_KERNEL_ARG_CHECK(k_compute_derivatives_, 14, sizeof(cl_mem), (void *)&target_cells_.d_map_inverse_cov_);
  OCL_SET_KERNEL_ARG_CHECK(k_compute_derivatives_, 15, sizeof(cl_mem), (void *)&d_inputs_x_);
  OCL_SET_KERNEL_ARG_CHECK(k_compute_derivatives_, 16, sizeof(cl_mem), (void *)&d_inputs_y_);
  OCL_SET_KERNEL_ARG_CHECK(k_compute_derivatives_, 17, sizeof(cl_mem), (void *)&d_inputs_z_);
  OCL_SET_KERNEL_ARG_CHECK(k_compute_derivatives_, 18, sizeof(cl_mem), (void *)&d_j_ang_);
  OCL_SET_KERNEL_ARG_CHECK(k_compute_derivatives_, 19, sizeof(cl_mem), (void *)&d_h_ang_);
  OCL_SET_KERNEL_ARG_CHECK(k_compute_derivatives_, 20, sizeof(cl_mem), (void *)&d_scores_);
  OCL_SET_KERNEL_ARG_CHECK(k_compute_derivatives_, 21, sizeof(cl_mem), (void *)&d_score_gradients_);
  OCL_SET_KERNEL_ARG_CHECK(k_compute_derivatives_, 22, sizeof(cl_mem), (void *)&d_hessians_);
  OCL_SET_KERNEL_ARG_CHECK(k_compute_derivatives_, 23, sizeof(float), (void*)&gauss_d1_f_);
  OCL_SET_KERNEL_ARG_CHECK(k_compute_derivatives_, 24, sizeof(float), (void*)&gauss_d2_f_);
  OCL_SET_KERNEL_ARG_CHECK(k_compute_derivatives_, 25, sizeof(float), (void*)&gauss_d3_f_);

  // work item
  global_item_size = ((target_cells_.num_centroids_ + local_item_size - 1) / local_item_size) * local_item_size;

  // kicking the kernel
  ret = clEnqueueNDRangeKernel(Queue_, k_compute_derivatives_, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
  if (CL_SUCCESS != ret) {
    std::cerr << "error : clEnqueueNDRangeKernel error code " << ret << std::endl;
    return -1;
  }
  return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointSource, typename PointTarget>
int ndt_ocl::NormalDistributionsTransform<PointSource, PointTarget>::readOCLMemoryObjects()
{
  OCL_READ_BUFFER_CHECK(Queue_, d_neighbor_candidates_, CL_TRUE, 0, neighbor_candidates_size_int_, neighbor_candidate_indexes_, 0, NULL, NULL);
  OCL_READ_BUFFER_CHECK(Queue_, d_neighbor_candidate_dists_, CL_TRUE, 0, neighbor_candidates_size_float_, neighbor_candidate_dists_, 0, NULL, NULL);
  OCL_READ_BUFFER_CHECK(Queue_, d_scores_, CL_TRUE, 0, score_size_, scores_, 0, NULL, NULL);
  OCL_READ_BUFFER_CHECK(Queue_, d_score_gradients_, CL_TRUE, 0, score_gradients_size_, score_gradients_, 0, NULL, NULL);
  OCL_READ_BUFFER_CHECK(Queue_, d_hessians_, CL_TRUE, 0, hessians_size_, hessians_, 0, NULL, NULL);
  return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointSource, typename PointTarget>
void ndt_ocl::NormalDistributionsTransform<PointSource, PointTarget>::releaseOCLMemoryObjects()
{
  OCL_RELEASE_MEMORY_CHECK(d_query_x_);
  OCL_RELEASE_MEMORY_CHECK(d_query_y_);
  OCL_RELEASE_MEMORY_CHECK(d_query_z_);
  OCL_RELEASE_MEMORY_CHECK(d_neighbor_candidates_);
  OCL_RELEASE_MEMORY_CHECK(d_neighbor_candidate_dists_);
  OCL_RELEASE_MEMORY_CHECK(d_inputs_x_);
  OCL_RELEASE_MEMORY_CHECK(d_inputs_y_);
  OCL_RELEASE_MEMORY_CHECK(d_inputs_z_);
  OCL_RELEASE_MEMORY_CHECK(d_j_ang_);
  OCL_RELEASE_MEMORY_CHECK(d_h_ang_);
  OCL_RELEASE_MEMORY_CHECK(d_scores_);
  OCL_RELEASE_MEMORY_CHECK(d_score_gradients_);
  OCL_RELEASE_MEMORY_CHECK(d_hessians_);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointSource, typename PointTarget>
void ndt_ocl::NormalDistributionsTransform<PointSource, PointTarget>::finalizeOCL()
{
  target_cells_.finalize();
  releaseOCLMemoryObjects();
  OCL_RELEASE_KERNEL_CHECK(k_compute_derivatives_);
  OCL_RELEASE_QUEUE_CHECK(Queue_);
  OCL_RELEASE_PROGRAM_CHECK(program_);
  OCL_RELEASE_CONTEXT_CHECK(context_);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointSource, typename PointTarget>
void ndt_ocl::NormalDistributionsTransform<PointSource, PointTarget>::computeAngleDerivatives(
                                                                                              Eigen::Matrix<double, 6, 1> & p, bool compute_hessian)
{
  // Simplified math for near 0 angles
  double cx, cy, cz, sx, sy, sz;
  if (fabs(p(3)) < 10e-5) {
    // p(3) = 0;
    cx = 1.0;
    sx = 0.0;
  } else {
    cx = cos(p(3));
    sx = sin(p(3));
  }
  if (fabs(p(4)) < 10e-5) {
    // p(4) = 0;
    cy = 1.0;
    sy = 0.0;
  } else {
    cy = cos(p(4));
    sy = sin(p(4));
  }

  if (fabs(p(5)) < 10e-5) {
    // p(5) = 0;
    cz = 1.0;
    sz = 0.0;
  } else {
    cz = cos(p(5));
    sz = sin(p(5));
  }

  // Precomputed angular gradiant components. Letters correspond to Equation 6.19 [Magnusson 2009]
  j_ang_a_ << (-sx * sz + cx * sy * cz), (-sx * cz - cx * sy * sz), (-cx * cy);
  j_ang_b_ << (cx * sz + sx * sy * cz), (cx * cz - sx * sy * sz), (-sx * cy);
  j_ang_c_ << (-sy * cz), sy * sz, cy;
  j_ang_d_ << sx * cy * cz, (-sx * cy * sz), sx * sy;
  j_ang_e_ << (-cx * cy * cz), cx * cy * sz, (-cx * sy);
  j_ang_f_ << (-cy * sz), (-cy * cz), 0;
  j_ang_g_ << (cx * cz - sx * sy * sz), (-cx * sz - sx * sy * cz), 0;
  j_ang_h_ << (sx * cz + cx * sy * sz), (cx * sy * cz - sx * sz), 0;

  j_ang.setZero();
  j_ang.row(0).noalias() =
    Eigen::Vector4f((-sx * sz + cx * sy * cz), (-sx * cz - cx * sy * sz), (-cx * cy), 0.0f);
  j_ang.row(1).noalias() =
    Eigen::Vector4f((cx * sz + sx * sy * cz), (cx * cz - sx * sy * sz), (-sx * cy), 0.0f);
  j_ang.row(2).noalias() = Eigen::Vector4f((-sy * cz), sy * sz, cy, 0.0f);
  j_ang.row(3).noalias() = Eigen::Vector4f(sx * cy * cz, (-sx * cy * sz), sx * sy, 0.0f);
  j_ang.row(4).noalias() = Eigen::Vector4f((-cx * cy * cz), cx * cy * sz, (-cx * sy), 0.0f);
  j_ang.row(5).noalias() = Eigen::Vector4f((-cy * sz), (-cy * cz), 0, 0.0f);
  j_ang.row(6).noalias() =
    Eigen::Vector4f((cx * cz - sx * sy * sz), (-cx * sz - sx * sy * cz), 0, 0.0f);
  j_ang.row(7).noalias() =
    Eigen::Vector4f((sx * cz + cx * sy * sz), (cx * sy * cz - sx * sz), 0, 0.0f);

  if (compute_hessian) {
    // Precomputed angular hessian components. Letters correspond to Equation 6.21 and numbers correspond to row index
    // [Magnusson 2009]
    h_ang_a2_ << (-cx * sz - sx * sy * cz), (-cx * cz + sx * sy * sz), sx * cy;
    h_ang_a3_ << (-sx * sz + cx * sy * cz), (-cx * sy * sz - sx * cz), (-cx * cy);

    h_ang_b2_ << (cx * cy * cz), (-cx * cy * sz), (cx * sy);
    h_ang_b3_ << (sx * cy * cz), (-sx * cy * sz), (sx * sy);

    h_ang_c2_ << (-sx * cz - cx * sy * sz), (sx * sz - cx * sy * cz), 0;
    h_ang_c3_ << (cx * cz - sx * sy * sz), (-sx * sy * cz - cx * sz), 0;

    h_ang_d1_ << (-cy * cz), (cy * sz), (sy);
    h_ang_d2_ << (-sx * sy * cz), (sx * sy * sz), (sx * cy);
    h_ang_d3_ << (cx * sy * cz), (-cx * sy * sz), (-cx * cy);

    h_ang_e1_ << (sy * sz), (sy * cz), 0;
    h_ang_e2_ << (-sx * cy * sz), (-sx * cy * cz), 0;
    h_ang_e3_ << (cx * cy * sz), (cx * cy * cz), 0;

    h_ang_f1_ << (-cy * cz), (cy * sz), 0;
    h_ang_f2_ << (-cx * sz - sx * sy * cz), (-cx * cz + sx * sy * sz), 0;
    h_ang_f3_ << (-sx * sz + cx * sy * cz), (-cx * sy * sz - sx * cz), 0;

    h_ang.setZero();
    h_ang.row(0).noalias() =
      Eigen::Vector4f((-cx * sz - sx * sy * cz), (-cx * cz + sx * sy * sz), sx * cy, 0.0f);  // a2
    h_ang.row(1).noalias() = Eigen::Vector4f(
                                             (-sx * sz + cx * sy * cz), (-cx * sy * sz - sx * cz), (-cx * cy), 0.0f);  // a3

    h_ang.row(2).noalias() =
      Eigen::Vector4f((cx * cy * cz), (-cx * cy * sz), (cx * sy), 0.0f);  // b2
    h_ang.row(3).noalias() =
      Eigen::Vector4f((sx * cy * cz), (-sx * cy * sz), (sx * sy), 0.0f);  // b3

    h_ang.row(4).noalias() =
      Eigen::Vector4f((-sx * cz - cx * sy * sz), (sx * sz - cx * sy * cz), 0, 0.0f);  // c2
    h_ang.row(5).noalias() =
      Eigen::Vector4f((cx * cz - sx * sy * sz), (-sx * sy * cz - cx * sz), 0, 0.0f);  // c3

    h_ang.row(6).noalias() = Eigen::Vector4f((-cy * cz), (cy * sz), (sy), 0.0f);  // d1
    h_ang.row(7).noalias() =
      Eigen::Vector4f((-sx * sy * cz), (sx * sy * sz), (sx * cy), 0.0f);  // d2
    h_ang.row(8).noalias() =
      Eigen::Vector4f((cx * sy * cz), (-cx * sy * sz), (-cx * cy), 0.0f);  // d3

    h_ang.row(9).noalias() = Eigen::Vector4f((sy * sz), (sy * cz), 0, 0.0f);               // e1
    h_ang.row(10).noalias() = Eigen::Vector4f((-sx * cy * sz), (-sx * cy * cz), 0, 0.0f);  // e2
    h_ang.row(11).noalias() = Eigen::Vector4f((cx * cy * sz), (cx * cy * cz), 0, 0.0f);    // e3

    h_ang.row(12).noalias() = Eigen::Vector4f((-cy * cz), (cy * sz), 0, 0.0f);  // f1
    h_ang.row(13).noalias() =
      Eigen::Vector4f((-cx * sz - sx * sy * cz), (-cx * cz + sx * sy * sz), 0, 0.0f);  // f2
    h_ang.row(14).noalias() =
      Eigen::Vector4f((-sx * sz + cx * sy * cz), (-cx * sy * sz - sx * cz), 0, 0.0f);  // f3
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointSource, typename PointTarget>
void ndt_ocl::NormalDistributionsTransform<PointSource, PointTarget>::computePointDerivatives(
  Eigen::Vector3d & x, Eigen::Matrix<float, 4, 6> & point_gradient_,
  Eigen::Matrix<float, 24, 6> & point_hessian_, bool compute_hessian) const
{
  Eigen::Vector4f x4(x[0], x[1], x[2], 0.0f);

  // Calculate first derivative of Transformation Equation 6.17 w.r.t. transform vector p.
  // Derivative w.r.t. ith element of transform vector corresponds to column i, Equation 6.18 and 6.19 [Magnusson 2009]
  Eigen::Matrix<float, 8, 1> x_j_ang = j_ang * x4;

  point_gradient_(1, 3) = x_j_ang[0];
  point_gradient_(2, 3) = x_j_ang[1];
  point_gradient_(0, 4) = x_j_ang[2];
  point_gradient_(1, 4) = x_j_ang[3];
  point_gradient_(2, 4) = x_j_ang[4];
  point_gradient_(0, 5) = x_j_ang[5];
  point_gradient_(1, 5) = x_j_ang[6];
  point_gradient_(2, 5) = x_j_ang[7];

  if (compute_hessian) {
    Eigen::Matrix<float, 16, 1> x_h_ang = h_ang * x4;

    // Vectors from Equation 6.21 [Magnusson 2009]
    Eigen::Vector4f a(0, x_h_ang[0], x_h_ang[1], 0.0f);
    Eigen::Vector4f b(0, x_h_ang[2], x_h_ang[3], 0.0f);
    Eigen::Vector4f c(0, x_h_ang[4], x_h_ang[5], 0.0f);
    Eigen::Vector4f d(x_h_ang[6], x_h_ang[7], x_h_ang[8], 0.0f);
    Eigen::Vector4f e(x_h_ang[9], x_h_ang[10], x_h_ang[11], 0.0f);
    Eigen::Vector4f f(x_h_ang[12], x_h_ang[13], x_h_ang[14], 0.0f);

    // Calculate second derivative of Transformation Equation 6.17 w.r.t. transform vector p.
    // Derivative w.r.t. ith and jth elements of transform vector corresponds to the 3x1 block matrix starting at
    // (3i,j), Equation 6.20 and 6.21 [Magnusson 2009]
    point_hessian_.block<4, 1>((9 / 3) * 4, 3) = a;
    point_hessian_.block<4, 1>((12 / 3) * 4, 3) = b;
    point_hessian_.block<4, 1>((15 / 3) * 4, 3) = c;
    point_hessian_.block<4, 1>((9 / 3) * 4, 4) = b;
    point_hessian_.block<4, 1>((12 / 3) * 4, 4) = d;
    point_hessian_.block<4, 1>((15 / 3) * 4, 4) = e;
    point_hessian_.block<4, 1>((9 / 3) * 4, 5) = c;
    point_hessian_.block<4, 1>((12 / 3) * 4, 5) = e;
    point_hessian_.block<4, 1>((15 / 3) * 4, 5) = f;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointSource, typename PointTarget>
void ndt_ocl::NormalDistributionsTransform<PointSource, PointTarget>::computePointDerivatives(
  Eigen::Vector3d & x, Eigen::Matrix<double, 3, 6> & point_gradient_,
  Eigen::Matrix<double, 18, 6> & point_hessian_, bool compute_hessian) const
{
  // Calculate first derivative of Transformation Equation 6.17 w.r.t. transform vector p.
  // Derivative w.r.t. ith element of transform vector corresponds to column i, Equation 6.18 and 6.19 [Magnusson 2009]
  point_gradient_(1, 3) = x.dot(j_ang_a_);
  point_gradient_(2, 3) = x.dot(j_ang_b_);
  point_gradient_(0, 4) = x.dot(j_ang_c_);
  point_gradient_(1, 4) = x.dot(j_ang_d_);
  point_gradient_(2, 4) = x.dot(j_ang_e_);
  point_gradient_(0, 5) = x.dot(j_ang_f_);
  point_gradient_(1, 5) = x.dot(j_ang_g_);
  point_gradient_(2, 5) = x.dot(j_ang_h_);

  if (compute_hessian) {
    // Vectors from Equation 6.21 [Magnusson 2009]
    Eigen::Vector3d a, b, c, d, e, f;

    a << 0, x.dot(h_ang_a2_), x.dot(h_ang_a3_);
    b << 0, x.dot(h_ang_b2_), x.dot(h_ang_b3_);
    c << 0, x.dot(h_ang_c2_), x.dot(h_ang_c3_);
    d << x.dot(h_ang_d1_), x.dot(h_ang_d2_), x.dot(h_ang_d3_);
    e << x.dot(h_ang_e1_), x.dot(h_ang_e2_), x.dot(h_ang_e3_);
    f << x.dot(h_ang_f1_), x.dot(h_ang_f2_), x.dot(h_ang_f3_);

    // Calculate second derivative of Transformation Equation 6.17 w.r.t. transform vector p.
    // Derivative w.r.t. ith and jth elements of transform vector corresponds to the 3x1 block matrix starting at
    // (3i,j), Equation 6.20 and 6.21 [Magnusson 2009]
    point_hessian_.block<3, 1>(9, 3) = a;
    point_hessian_.block<3, 1>(12, 3) = b;
    point_hessian_.block<3, 1>(15, 3) = c;
    point_hessian_.block<3, 1>(9, 4) = b;
    point_hessian_.block<3, 1>(12, 4) = d;
    point_hessian_.block<3, 1>(15, 4) = e;
    point_hessian_.block<3, 1>(9, 5) = c;
    point_hessian_.block<3, 1>(12, 5) = e;
    point_hessian_.block<3, 1>(15, 5) = f;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointSource, typename PointTarget>
double ndt_ocl::NormalDistributionsTransform<PointSource, PointTarget>::updateDerivatives(
  Eigen::Matrix<double, 6, 1> & score_gradient, Eigen::Matrix<double, 6, 6> & hessian,
  const Eigen::Matrix<float, 4, 6> & point_gradient4,
  const Eigen::Matrix<float, 24, 6> & point_hessian_, const Eigen::Vector3d & x_trans,
  const Eigen::Matrix3d & c_inv, bool compute_hessian) const
{
  Eigen::Matrix<float, 1, 4> x_trans4(x_trans[0], x_trans[1], x_trans[2], 0.0f);
  Eigen::Matrix4f c_inv4 = Eigen::Matrix4f::Zero();
  c_inv4.topLeftCorner(3, 3) = c_inv.cast<float>();

  float gauss_d2 = gauss_d2_;

  // e^(-d_2/2 * (x_k - mu_k)^T Sigma_k^-1 (x_k - mu_k)) Equation 6.9 [Magnusson 2009]
  float e_x_cov_x = exp(-gauss_d2 * x_trans4.dot(x_trans4 * c_inv4) * 0.5f);
  // Calculate probability of transtormed points existance, Equation 6.9 [Magnusson 2009]
  float score_inc = -gauss_d1_ * e_x_cov_x;

  e_x_cov_x = gauss_d2 * e_x_cov_x;

  // Error checking for invalid values.
  if (e_x_cov_x > 1 || e_x_cov_x < 0 || e_x_cov_x != e_x_cov_x) return (0);

  // Reusable portion of Equation 6.12 and 6.13 [Magnusson 2009]
  e_x_cov_x *= gauss_d1_;

  Eigen::Matrix<float, 4, 6> c_inv4_x_point_gradient4 = c_inv4 * point_gradient4;
  Eigen::Matrix<float, 6, 1> x_trans4_dot_c_inv4_x_point_gradient4 =
    x_trans4 * c_inv4_x_point_gradient4;

  score_gradient.noalias() += (e_x_cov_x * x_trans4_dot_c_inv4_x_point_gradient4).cast<double>();

  if (compute_hessian) {
    Eigen::Matrix<float, 1, 4> x_trans4_x_c_inv4 = x_trans4 * c_inv4;
    Eigen::Matrix<float, 6, 6> point_gradient4_colj_dot_c_inv4_x_point_gradient4_col_i =
      point_gradient4.transpose() * c_inv4_x_point_gradient4;
    Eigen::Matrix<float, 6, 1> x_trans4_dot_c_inv4_x_ext_point_hessian_4ij;

    for (int i = 0; i < 6; i++) {
      // Sigma_k^-1 d(T(x,p))/dpi, Reusable portion of Equation 6.12 and 6.13 [Magnusson 2009]
      // Update gradient, Equation 6.12 [Magnusson 2009]
      x_trans4_dot_c_inv4_x_ext_point_hessian_4ij.noalias() =
        x_trans4_x_c_inv4 * point_hessian_.block<4, 6>(i * 4, 0);

      for (int j = 0; j < hessian.cols(); j++) {
        // Update hessian, Equation 6.13 [Magnusson 2009]
        hessian(i, j) +=
          e_x_cov_x * (-gauss_d2 * x_trans4_dot_c_inv4_x_point_gradient4(i) *
                       x_trans4_dot_c_inv4_x_point_gradient4(j) +
                       x_trans4_dot_c_inv4_x_ext_point_hessian_4ij(j) +
                       point_gradient4_colj_dot_c_inv4_x_point_gradient4_col_i(j, i));
      }
    }
  }

  return (score_inc);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointSource, typename PointTarget>
void ndt_ocl::NormalDistributionsTransform<PointSource, PointTarget>::computeHessian(
  Eigen::Matrix<double, 6, 6> & hessian, PointCloudSource & trans_cloud,
  Eigen::Matrix<double, 6, 1> &)
{
  // Original Point and Transformed Point
  PointSource x_pt, x_trans_pt;
  // Original Point and Transformed Point (for math)
  Eigen::Vector3d x, x_trans;
  // Occupied Voxel
  TargetGridLeafConstPtr cell;
  // Inverse Covariance of Occupied Voxel
  Eigen::Matrix3d c_inv;

  // Initialize Point Gradient and Hessian
  Eigen::Matrix<double, 3, 6> point_gradient_;
  Eigen::Matrix<double, 18, 6> point_hessian_;
  point_gradient_.setZero();
  point_gradient_.block<3, 3>(0, 0).setIdentity();
  point_hessian_.setZero();

  hessian.setZero();

  // Precompute Angular Derivatives unessisary because only used after regular derivative calculation

  // Update hessian for each point, line 17 in Algorithm 2 [Magnusson 2009]
  for (size_t idx = 0; idx < input_->points.size(); idx++) {
    x_trans_pt = trans_cloud.points[idx];

    // Find nieghbors (Radius search has been experimentally faster than direct neighbor checking.
    std::vector<TargetGridLeafConstPtr> neighborhood;
    std::vector<float> distances;
    target_cells_.radiusSearch(x_trans_pt, resolution_, neighborhood, distances);

    for (typename std::vector<TargetGridLeafConstPtr>::iterator neighborhood_it =
           neighborhood.begin();
         neighborhood_it != neighborhood.end(); neighborhood_it++) {
      cell = *neighborhood_it;

      {
        x_pt = input_->points[idx];
        x = Eigen::Vector3d(x_pt.x, x_pt.y, x_pt.z);

        x_trans = Eigen::Vector3d(x_trans_pt.x, x_trans_pt.y, x_trans_pt.z);

        // Denorm point, x_k' in Equations 6.12 and 6.13 [Magnusson 2009]
        x_trans -= cell->getMean();
        // Uses precomputed covariance for speed.
        c_inv = cell->getInverseCov();

        // Compute derivative of transform function w.r.t. transform vector, J_E and H_E in Equations 6.18 and 6.20
        // [Magnusson 2009]
        computePointDerivatives(x, point_gradient_, point_hessian_);
        // Update hessian, lines 21 in Algorithm 2, according to Equations 6.10, 6.12 and 6.13, respectively [Magnusson
        // 2009]
        updateHessian(hessian, point_gradient_, point_hessian_, x_trans, c_inv);
      }
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointSource, typename PointTarget>
void ndt_ocl::NormalDistributionsTransform<PointSource, PointTarget>::updateHessian(
  Eigen::Matrix<double, 6, 6> & hessian, const Eigen::Matrix<double, 3, 6> & point_gradient_,
  const Eigen::Matrix<double, 18, 6> & point_hessian_, const Eigen::Vector3d & x_trans,
  const Eigen::Matrix3d & c_inv) const
{
  Eigen::Vector3d cov_dxd_pi;
  // e^(-d_2/2 * (x_k - mu_k)^T Sigma_k^-1 (x_k - mu_k)) Equation 6.9 [Magnusson 2009]
  double e_x_cov_x = gauss_d2_ * exp(-gauss_d2_ * x_trans.dot(c_inv * x_trans) / 2);

  // Error checking for invalid values.
  if (e_x_cov_x > 1 || e_x_cov_x < 0 || e_x_cov_x != e_x_cov_x) return;

  // Reusable portion of Equation 6.12 and 6.13 [Magnusson 2009]
  e_x_cov_x *= gauss_d1_;

  for (int i = 0; i < 6; i++) {
    // Sigma_k^-1 d(T(x,p))/dpi, Reusable portion of Equation 6.12 and 6.13 [Magnusson 2009]
    cov_dxd_pi = c_inv * point_gradient_.col(i);

    for (int j = 0; j < hessian.cols(); j++) {
      // Update hessian, Equation 6.13 [Magnusson 2009]
      hessian(i, j) += e_x_cov_x * (-gauss_d2_ * x_trans.dot(cov_dxd_pi) *
                                    x_trans.dot(c_inv * point_gradient_.col(j)) +
                                    x_trans.dot(c_inv * point_hessian_.block<3, 1>(3 * i, j)) +
                                    point_gradient_.col(j).dot(cov_dxd_pi));
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointSource, typename PointTarget>
bool ndt_ocl::NormalDistributionsTransform<PointSource, PointTarget>::updateIntervalMT(
  double & a_l, double & f_l, double & g_l, double & a_u, double & f_u, double & g_u, double a_t,
  double f_t, double g_t)
{
  // Case U1 in Update Algorithm and Case a in Modified Update Algorithm [More, Thuente 1994]
  if (f_t > f_l) {
    a_u = a_t;
    f_u = f_t;
    g_u = g_t;
    return (false);
  }
  // Case U2 in Update Algorithm and Case b in Modified Update Algorithm [More, Thuente 1994]
  else if (g_t * (a_l - a_t) > 0) {
    a_l = a_t;
    f_l = f_t;
    g_l = g_t;
    return (false);
  }
  // Case U3 in Update Algorithm and Case c in Modified Update Algorithm [More, Thuente 1994]
  else if (g_t * (a_l - a_t) < 0) {
    a_u = a_l;
    f_u = f_l;
    g_u = g_l;

    a_l = a_t;
    f_l = f_t;
    g_l = g_t;
    return (false);
  }
  // Interval Converged
  else
    return (true);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointSource, typename PointTarget>
double ndt_ocl::NormalDistributionsTransform<PointSource, PointTarget>::trialValueSelectionMT(
  double a_l, double f_l, double g_l, double a_u, double f_u, double g_u, double a_t, double f_t,
  double g_t)
{
  // Case 1 in Trial Value Selection [More, Thuente 1994]
  if (f_t > f_l) {
    // Calculate the minimizer of the cubic that interpolates f_l, f_t, g_l and g_t
    // Equation 2.4.52 [Sun, Yuan 2006]
    double z = 3 * (f_t - f_l) / (a_t - a_l) - g_t - g_l;
    double w = std::sqrt(z * z - g_t * g_l);
    // Equation 2.4.56 [Sun, Yuan 2006]
    double a_c = a_l + (a_t - a_l) * (w - g_l - z) / (g_t - g_l + 2 * w);

    // Calculate the minimizer of the quadratic that interpolates f_l, f_t and g_l
    // Equation 2.4.2 [Sun, Yuan 2006]
    double a_q = a_l - 0.5 * (a_l - a_t) * g_l / (g_l - (f_l - f_t) / (a_l - a_t));

    if (std::fabs(a_c - a_l) < std::fabs(a_q - a_l))
      return (a_c);
    else
      return (0.5 * (a_q + a_c));
  }
  // Case 2 in Trial Value Selection [More, Thuente 1994]
  else if (g_t * g_l < 0) {
    // Calculate the minimizer of the cubic that interpolates f_l, f_t, g_l and g_t
    // Equation 2.4.52 [Sun, Yuan 2006]
    double z = 3 * (f_t - f_l) / (a_t - a_l) - g_t - g_l;
    double w = std::sqrt(z * z - g_t * g_l);
    // Equation 2.4.56 [Sun, Yuan 2006]
    double a_c = a_l + (a_t - a_l) * (w - g_l - z) / (g_t - g_l + 2 * w);

    // Calculate the minimizer of the quadratic that interpolates f_l, g_l and g_t
    // Equation 2.4.5 [Sun, Yuan 2006]
    double a_s = a_l - (a_l - a_t) / (g_l - g_t) * g_l;

    if (std::fabs(a_c - a_t) >= std::fabs(a_s - a_t))
      return (a_c);
    else
      return (a_s);
  }
  // Case 3 in Trial Value Selection [More, Thuente 1994]
  else if (std::fabs(g_t) <= std::fabs(g_l)) {
    // Calculate the minimizer of the cubic that interpolates f_l, f_t, g_l and g_t
    // Equation 2.4.52 [Sun, Yuan 2006]
    double z = 3 * (f_t - f_l) / (a_t - a_l) - g_t - g_l;
    double w = std::sqrt(z * z - g_t * g_l);
    double a_c = a_l + (a_t - a_l) * (w - g_l - z) / (g_t - g_l + 2 * w);

    // Calculate the minimizer of the quadratic that interpolates g_l and g_t
    // Equation 2.4.5 [Sun, Yuan 2006]
    double a_s = a_l - (a_l - a_t) / (g_l - g_t) * g_l;

    double a_t_next;

    if (std::fabs(a_c - a_t) < std::fabs(a_s - a_t))
      a_t_next = a_c;
    else
      a_t_next = a_s;

    if (a_t > a_l)
      return (std::min(a_t + 0.66 * (a_u - a_t), a_t_next));
    else
      return (std::max(a_t + 0.66 * (a_u - a_t), a_t_next));
  }
  // Case 4 in Trial Value Selection [More, Thuente 1994]
  else {
    // Calculate the minimizer of the cubic that interpolates f_u, f_t, g_u and g_t
    // Equation 2.4.52 [Sun, Yuan 2006]
    double z = 3 * (f_t - f_u) / (a_t - a_u) - g_t - g_u;
    double w = std::sqrt(z * z - g_t * g_u);
    // Equation 2.4.56 [Sun, Yuan 2006]
    return (a_u + (a_t - a_u) * (w - g_u - z) / (g_t - g_u + 2 * w));
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointSource, typename PointTarget>
double ndt_ocl::NormalDistributionsTransform<PointSource, PointTarget>::computeStepLengthMT(
  const Eigen::Matrix<double, 6, 1> & x, Eigen::Matrix<double, 6, 1> & step_dir, double step_init,
  double step_max, double step_min, double & score, Eigen::Matrix<double, 6, 1> & score_gradient,
  Eigen::Matrix<double, 6, 6> & hessian, PointCloudSource & trans_cloud)
{
  // Set the value of phi(0), Equation 1.3 [More, Thuente 1994]
  double phi_0 = -score;
  // Set the value of phi'(0), Equation 1.3 [More, Thuente 1994]
  double d_phi_0 = -(score_gradient.dot(step_dir));

  Eigen::Matrix<double, 6, 1> x_t;

  if (d_phi_0 >= 0) {
    // Not a decent direction
    if (d_phi_0 == 0)
      return 0;
    else {
      // Reverse step direction and calculate optimal step.
      d_phi_0 *= -1;
      step_dir *= -1;
    }
  }

  // The Search Algorithm for T(mu) [More, Thuente 1994]

  int max_step_iterations = 10;
  int step_iterations = 0;

  // Sufficient decreace constant, Equation 1.1 [More, Thuete 1994]
  double mu = 1.e-4;
  // Curvature condition constant, Equation 1.2 [More, Thuete 1994]
  double nu = 0.9;

  // Initial endpoints of Interval I,
  double a_l = 0, a_u = 0;

  // Auxiliary function psi is used until I is determined ot be a closed interval, Equation 2.1 [More, Thuente 1994]
  double f_l = auxilaryFunction_PsiMT(a_l, phi_0, phi_0, d_phi_0, mu);
  double g_l = auxilaryFunction_dPsiMT(d_phi_0, d_phi_0, mu);

  double f_u = auxilaryFunction_PsiMT(a_u, phi_0, phi_0, d_phi_0, mu);
  double g_u = auxilaryFunction_dPsiMT(d_phi_0, d_phi_0, mu);

  // Check used to allow More-Thuente step length calculation to be skipped by making step_min == step_max
  bool interval_converged = (step_max - step_min) > 0, open_interval = true;

  double a_t = step_init;
  a_t = std::min(a_t, step_max);
  a_t = std::max(a_t, step_min);

  x_t = x + step_dir * a_t;

  final_transformation_ =
    (Eigen::Translation<float, 3>(
                                  static_cast<float>(x_t(0)), static_cast<float>(x_t(1)), static_cast<float>(x_t(2))) *
     Eigen::AngleAxis<float>(static_cast<float>(x_t(3)), Eigen::Vector3f::UnitX()) *
     Eigen::AngleAxis<float>(static_cast<float>(x_t(4)), Eigen::Vector3f::UnitY()) *
     Eigen::AngleAxis<float>(static_cast<float>(x_t(5)), Eigen::Vector3f::UnitZ()))
    .matrix();

  // New transformed point cloud
  transformPointCloud(*input_, trans_cloud, final_transformation_);

  // Updates score, gradient and hessian.  Hessian calculation is unessisary but testing showed that most step
  // calculations use the initial step suggestion and recalculation the reusable portions of the hessian would intail
  // more computation time.
  score = computeDerivatives(score_gradient, hessian, trans_cloud, x_t, true);

  // Calculate phi(alpha_t)
  double phi_t = -score;
  // Calculate phi'(alpha_t)
  double d_phi_t = -(score_gradient.dot(step_dir));

  // Calculate psi(alpha_t)
  double psi_t = auxilaryFunction_PsiMT(a_t, phi_t, phi_0, d_phi_0, mu);
  // Calculate psi'(alpha_t)
  double d_psi_t = auxilaryFunction_dPsiMT(d_phi_t, d_phi_0, mu);

  // Iterate until max number of iterations, interval convergance or a value satisfies the sufficient decrease,
  // Equation 1.1, and curvature condition, Equation 1.2 [More, Thuente 1994]
  while (
         !interval_converged && step_iterations < max_step_iterations &&
         !(psi_t <= 0 /*Sufficient Decrease*/ && d_phi_t <= -nu * d_phi_0 /*Curvature Condition*/)) {
    // Use auxilary function if interval I is not closed
    if (open_interval) {
      a_t = trialValueSelectionMT(a_l, f_l, g_l, a_u, f_u, g_u, a_t, psi_t, d_psi_t);
    } else {
      a_t = trialValueSelectionMT(a_l, f_l, g_l, a_u, f_u, g_u, a_t, phi_t, d_phi_t);
    }

    a_t = std::min(a_t, step_max);
    a_t = std::max(a_t, step_min);

    x_t = x + step_dir * a_t;

    final_transformation_ =
      (Eigen::Translation<float, 3>(
                                    static_cast<float>(x_t(0)), static_cast<float>(x_t(1)), static_cast<float>(x_t(2))) *
       Eigen::AngleAxis<float>(static_cast<float>(x_t(3)), Eigen::Vector3f::UnitX()) *
       Eigen::AngleAxis<float>(static_cast<float>(x_t(4)), Eigen::Vector3f::UnitY()) *
       Eigen::AngleAxis<float>(static_cast<float>(x_t(5)), Eigen::Vector3f::UnitZ()))
      .matrix();

    // New transformed point cloud
    // Done on final cloud to prevent wasted computation
    transformPointCloud(*input_, trans_cloud, final_transformation_);

    // Updates score, gradient. Values stored to prevent wasted computation.
    score = computeDerivatives(score_gradient, hessian, trans_cloud, x_t, false);

    // Calculate phi(alpha_t+)
    phi_t = -score;
    // Calculate phi'(alpha_t+)
    d_phi_t = -(score_gradient.dot(step_dir));

    // Calculate psi(alpha_t+)
    psi_t = auxilaryFunction_PsiMT(a_t, phi_t, phi_0, d_phi_0, mu);
    // Calculate psi'(alpha_t+)
    d_psi_t = auxilaryFunction_dPsiMT(d_phi_t, d_phi_0, mu);

    // Check if I is now a closed interval
    if (open_interval && (psi_t <= 0 && d_psi_t >= 0)) {
      open_interval = false;

      // Converts f_l and g_l from psi to phi
      f_l = f_l + phi_0 - mu * d_phi_0 * a_l;
      g_l = g_l + mu * d_phi_0;

      // Converts f_u and g_u from psi to phi
      f_u = f_u + phi_0 - mu * d_phi_0 * a_u;
      g_u = g_u + mu * d_phi_0;
    }

    if (open_interval) {
      // Update interval end points using Updating Algorithm [More, Thuente 1994]
      interval_converged = updateIntervalMT(a_l, f_l, g_l, a_u, f_u, g_u, a_t, psi_t, d_psi_t);
    } else {
      // Update interval end points using Modified Updating Algorithm [More, Thuente 1994]
      interval_converged = updateIntervalMT(a_l, f_l, g_l, a_u, f_u, g_u, a_t, phi_t, d_phi_t);
    }

    step_iterations++;
  }

  // If inner loop was run then hessian needs to be calculated.
  // Hessian is unnessisary for step length determination but gradients are required
  // so derivative and transform data is stored for the next iteration.
  if (step_iterations) computeHessian(hessian, trans_cloud, x_t);

  return (a_t);
}

template <typename PointSource, typename PointTarget>
double ndt_ocl::NormalDistributionsTransform<PointSource, PointTarget>::calculateScore(
  const PointCloudSource & trans_cloud) const
{
  double score = 0;

  for (int idx = 0; idx < trans_cloud.points.size(); idx++) {
    PointSource x_trans_pt = trans_cloud.points[idx];

    // Find nieghbors (Radius search has been experimentally faster than direct neighbor checking.
    std::vector<TargetGridLeafConstPtr> neighborhood;
    std::vector<float> distances;
    target_cells_.radiusSearch(x_trans_pt, resolution_, neighborhood, distances);

    for (typename std::vector<TargetGridLeafConstPtr>::iterator neighborhood_it =
           neighborhood.begin();
         neighborhood_it != neighborhood.end(); neighborhood_it++) {
      TargetGridLeafConstPtr cell = *neighborhood_it;

      Eigen::Vector3d x_trans = Eigen::Vector3d(x_trans_pt.x, x_trans_pt.y, x_trans_pt.z);

      // Denorm point, x_k' in Equations 6.12 and 6.13 [Magnusson 2009]
      x_trans -= cell->getMean();
      // Uses precomputed covariance for speed.
      Eigen::Matrix3d c_inv = cell->getInverseCov();

      // e^(-d_2/2 * (x_k - mu_k)^T Sigma_k^-1 (x_k - mu_k)) Equation 6.9 [Magnusson 2009]
      double e_x_cov_x = exp(-gauss_d2_ * x_trans.dot(c_inv * x_trans) / 2);
      // Calculate probability of transtormed points existance, Equation 6.9 [Magnusson 2009]
      double score_inc = -gauss_d1_ * e_x_cov_x - gauss_d3_;

      score += score_inc / neighborhood.size();
    }
  }
  return (score) / static_cast<double>(trans_cloud.size());
}

#endif  // PCL_REGISTRATION_NDT_OCL_IMPL_H_
