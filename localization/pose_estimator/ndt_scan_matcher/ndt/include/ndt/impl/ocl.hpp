/*
 * Copyright 2021 Tier IV inc. All rights reserved.
 *
 * The NormalDistributionsTransformOCL functions are based on
 * NormalDistributionsTransformOMP with naming modification.
 *
 * This class is also licensed under the Apache License, Version 2.0.
 *
 * === ORIGINAL LICENSE ===
 *
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
 *
 * ========================
 */

#ifndef NORMAL_DISTRIBUTIONS_TRANSFORM_OCL_HPP
#define NORMAL_DISTRIBUTIONS_TRANSFORM_OCL_HPP

template <class PointSource, class PointTarget>
NormalDistributionsTransformOCL<PointSource, PointTarget>::NormalDistributionsTransformOCL()
  : ndt_ptr_(new ndt_ocl::NormalDistributionsTransform<PointSource, PointTarget>)
{
}

template <class PointSource, class PointTarget>
void NormalDistributionsTransformOCL<PointSource, PointTarget>::align(
  pcl::PointCloud<PointSource> & output, const Eigen::Matrix4f & guess)
{
  ndt_ptr_->align(output, guess);
}

template <class PointSource, class PointTarget>
void NormalDistributionsTransformOCL<PointSource, PointTarget>::setInputTarget(
  const boost::shared_ptr<pcl::PointCloud<PointTarget>> & map_ptr)
{
  ndt_ptr_->setInputTarget(map_ptr);
}

template <class PointSource, class PointTarget>
void NormalDistributionsTransformOCL<PointSource, PointTarget>::setInputSource(
  const boost::shared_ptr<pcl::PointCloud<PointSource>> & scan_ptr)
{
  ndt_ptr_->setInputSource(scan_ptr);
}

template <class PointSource, class PointTarget>
void NormalDistributionsTransformOCL<PointSource, PointTarget>::setMaximumIterations(int max_iter)
{
  ndt_ptr_->setMaximumIterations(max_iter);
}

template <class PointSource, class PointTarget>
void NormalDistributionsTransformOCL<PointSource, PointTarget>::setResolution(float res)
{
  ndt_ptr_->setResolution(res);
}

template <class PointSource, class PointTarget>
void NormalDistributionsTransformOCL<PointSource, PointTarget>::setStepSize(double step_size)
{
  ndt_ptr_->setStepSize(step_size);
}

template <class PointSource, class PointTarget>
void NormalDistributionsTransformOCL<PointSource, PointTarget>::setTransformationEpsilon(double trans_eps)
{
  ndt_ptr_->setTransformationEpsilon(trans_eps);
}

template <class PointSource, class PointTarget>
int NormalDistributionsTransformOCL<PointSource, PointTarget>::getMaximumIterations()
{
  return ndt_ptr_->getMaximumIterations();
}

template <class PointSource, class PointTarget>
int NormalDistributionsTransformOCL<PointSource, PointTarget>::getFinalNumIteration() const
{
  return ndt_ptr_->getFinalNumIteration();
}

template <class PointSource, class PointTarget>
float NormalDistributionsTransformOCL<PointSource, PointTarget>::getResolution() const
{
  return ndt_ptr_->getResolution();
}

template <class PointSource, class PointTarget>
double NormalDistributionsTransformOCL<PointSource, PointTarget>::getStepSize() const
{
  return ndt_ptr_->getStepSize();
}

template <class PointSource, class PointTarget>
double NormalDistributionsTransformOCL<PointSource, PointTarget>::getTransformationEpsilon()
{
  return ndt_ptr_->getTransformationEpsilon();
}

template <class PointSource, class PointTarget>
double NormalDistributionsTransformOCL<PointSource, PointTarget>::getTransformationProbability()
  const
{
  return ndt_ptr_->getTransformationProbability();
}

template <class PointSource, class PointTarget>
double NormalDistributionsTransformOCL<PointSource, PointTarget>::getFitnessScore()
{
  return ndt_ptr_->getFitnessScore();
}

template <class PointSource, class PointTarget>
boost::shared_ptr<const pcl::PointCloud<PointTarget>>
NormalDistributionsTransformOCL<PointSource, PointTarget>::getInputTarget() const
{
  return ndt_ptr_->getInputTarget();
}

template <class PointSource, class PointTarget>
boost::shared_ptr<const pcl::PointCloud<PointSource>>
NormalDistributionsTransformOCL<PointSource, PointTarget>::getInputSource() const
{
  return ndt_ptr_->getInputSource();
}

template <class PointSource, class PointTarget>
Eigen::Matrix4f NormalDistributionsTransformOCL<PointSource, PointTarget>::getFinalTransformation() const
{
  return ndt_ptr_->getFinalTransformation();
}

template <class PointSource, class PointTarget>
std::vector<Eigen::Matrix4f>
NormalDistributionsTransformOCL<PointSource, PointTarget>::getFinalTransformationArray() const
{
  return ndt_ptr_->getFinalTransformationArray();
}

template <class PointSource, class PointTarget>
Eigen::Matrix<double, 6, 6> NormalDistributionsTransformOCL<PointSource, PointTarget>::getHessian() const
{
  // return ndt_ptr_->getHessian();
  return Eigen::Matrix<double, 6, 6>();
}

template <class PointSource, class PointTarget>
boost::shared_ptr<pcl::search::KdTree<PointTarget>>
NormalDistributionsTransformOCL<PointSource, PointTarget>::getSearchMethodTarget() const
{
  return ndt_ptr_->getSearchMethodTarget();
}

#endif
