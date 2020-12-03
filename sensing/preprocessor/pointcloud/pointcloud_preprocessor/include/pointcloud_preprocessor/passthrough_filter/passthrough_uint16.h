/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2012, Willow Garage, Inc.
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

#ifndef PCL_FILTERS_PASSTHROUGH_UINT16_H_
#define PCL_FILTERS_PASSTHROUGH_UINT16_H_

#include "pcl/filters/filter_indices.h"

namespace pcl
{
/** \brief @b PassThroughUInt16 passes points in a cloud based on constraints for one particular field of the point type.
 * \details Iterates through the entire input once, automatically filtering non-finite points and the points outside
 * the interval specified by setFilterLimits(), which applies only to the field specified by setFilterFieldName().
 * <br><br>
 * Usage example:
 * \code
 * pcl::PassThroughUInt16<PointType> ptfilter (true); // Initializing with true will allow us to extract the removed indices
 * ptfilter.setInputCloud (cloud_in);
 * ptfilter.setFilterFieldName ("x");
 * ptfilter.setFilterLimits (0.0, 1000.0);
 * ptfilter.filter (*indices_x);
 * // The indices_x array indexes all points of cloud_in that have x between 0.0 and 1000.0
 * indices_rem = ptfilter.getRemovedIndices ();
 * // The indices_rem array indexes all points of cloud_in that have x smaller than 0.0 or larger than 1000.0
 * // and also indexes all non-finite points of cloud_in
 * ptfilter.setIndices (indices_x);
 * ptfilter.setFilterFieldName ("z");
 * ptfilter.setFilterLimits (-10.0, 10.0);
 * ptfilter.setNegative (true);
 * ptfilter.filter (*indices_xz);
 * // The indices_xz array indexes all points of cloud_in that have x between 0.0 and 1000.0 and z larger than 10.0 or smaller than -10.0
 * ptfilter.setIndices (indices_xz);
 * ptfilter.setFilterFieldName ("intensity");
 * ptfilter.setFilterLimits (FLT_MIN, 0.5);
 * ptfilter.setNegative (false);
 * ptfilter.filter (*cloud_out);
 * // The resulting cloud_out contains all points of cloud_in that are finite and have:
 * // x between 0.0 and 1000.0, z larger than 10.0 or smaller than -10.0 and intensity smaller than 0.5.
 * \endcode
 * \author Radu Bogdan Rusu
 * \ingroup filters
 */
template <typename PointT>
class PassThroughUInt16 : public FilterIndices<PointT>
{
protected:
  typedef typename FilterIndices<PointT>::PointCloud PointCloud;
  typedef typename PointCloud::Ptr PointCloudPtr;
  typedef typename PointCloud::ConstPtr PointCloudConstPtr;
  typedef typename pcl::traits::fieldList<PointT>::type FieldList;

public:
  typedef boost::shared_ptr<PassThroughUInt16<PointT> > Ptr;
  typedef boost::shared_ptr<const PassThroughUInt16<PointT> > ConstPtr;

  /** \brief Constructor.
   * \param[in] extract_removed_indices Set to true if you want to be able to extract the indices of points being removed (default = false).
   */
  PassThroughUInt16(bool extract_removed_indices = false)
  : FilterIndices<PointT>::FilterIndices(extract_removed_indices),
    filter_field_name_(""),
    filter_limit_min_(0),
    filter_limit_max_(UINT16_MAX)
  {
    filter_name_ = "PassThroughUInt16";
  }

  /** \brief Provide the name of the field to be used for filtering data.
   * \details In conjunction with setFilterLimits(), points having values outside this interval for this field will be discarded.
   * \param[in] field_name The name of the field that will be used for filtering.
   */
  inline void setFilterFieldName(const std::string & field_name)
  {
    filter_field_name_ = field_name;
  }

  /** \brief Retrieve the name of the field to be used for filtering data.
   * \return The name of the field that will be used for filtering.
   */
  inline std::string const getFilterFieldName() { return (filter_field_name_); }

  /** \brief Set the numerical limits for the field for filtering data.
   * \details In conjunction with setFilterFieldName(), points having values outside this interval for this field will be discarded.
   * \param[in] limit_min The minimum allowed field value (default = FLT_MIN).
   * \param[in] limit_max The maximum allowed field value (default = FLT_MAX).
   */
  inline void setFilterLimits(const std::uint16_t & limit_min, const std::uint16_t & limit_max)
  {
    filter_limit_min_ = limit_min;
    filter_limit_max_ = limit_max;
  }

  /** \brief Get the numerical limits for the field for filtering data.
   * \param[out] limit_min The minimum allowed field value (default = FLT_MIN).
   * \param[out] limit_max The maximum allowed field value (default = FLT_MAX).
   */
  inline void getFilterLimits(std::uint16_t & limit_min, std::uint16_t & limit_max)
  {
    limit_min = filter_limit_min_;
    limit_max = filter_limit_max_;
  }

  /** \brief Set to true if we want to return the data outside the interval specified by setFilterLimits (min, max)
   * Default: false.
   * \warning This method will be removed in the future. Use setNegative() instead.
   * \param[in] limit_negative return data inside the interval (false) or outside (true)
   */
  inline void setFilterLimitsNegative(const bool limit_negative) { negative_ = limit_negative; }

  /** \brief Get whether the data outside the interval (min/max) is to be returned (true) or inside (false). 
   * \warning This method will be removed in the future. Use getNegative() instead.
   * \param[out] limit_negative true if data \b outside the interval [min; max] is to be returned, false otherwise
   */
  inline void getFilterLimitsNegative(bool & limit_negative) { limit_negative = negative_; }

  /** \brief Get whether the data outside the interval (min/max) is to be returned (true) or inside (false). 
   * \warning This method will be removed in the future. Use getNegative() instead.
   * \return true if data \b outside the interval [min; max] is to be returned, false otherwise
   */
  inline bool getFilterLimitsNegative() { return (negative_); }

protected:
  using PCLBase<PointT>::input_;
  using PCLBase<PointT>::indices_;
  using Filter<PointT>::filter_name_;
  using Filter<PointT>::getClassName;
  using FilterIndices<PointT>::negative_;
  using FilterIndices<PointT>::keep_organized_;
  using FilterIndices<PointT>::user_filter_value_;
  using FilterIndices<PointT>::extract_removed_indices_;
  using FilterIndices<PointT>::removed_indices_;

  /** \brief Filtered results are stored in a separate point cloud.
   * \param[out] output The resultant point cloud.
   */
  void applyFilter(PointCloud & output);

  /** \brief Filtered results are indexed by an indices array.
   * \param[out] indices The resultant indices.
   */
  void applyFilter(std::vector<int> & indices) { applyFilterIndices(indices); }

  /** \brief Filtered results are indexed by an indices array.
   * \param[out] indices The resultant indices.
   */
  void applyFilterIndices(std::vector<int> & indices);

private:
  /** \brief The name of the field that will be used for filtering. */
  std::string filter_field_name_;

  /** \brief The minimum allowed field value (default = FLT_MIN). */
  std::uint16_t filter_limit_min_;

  /** \brief The maximum allowed field value (default = FLT_MIN). */
  std::uint16_t filter_limit_max_;
};

////////////////////////////////////////////////////////////////////////////////////////////
/** \brief PassThroughUInt16 uses the base Filter class methods to pass through all data that satisfies the user given
 * constraints.
 * \author Radu B. Rusu
 * \ingroup filters
 */
template <>
class PCL_EXPORTS PassThroughUInt16<pcl::PCLPointCloud2> : public Filter<pcl::PCLPointCloud2>
{
  typedef pcl::PCLPointCloud2 PCLPointCloud2;
  typedef PCLPointCloud2::Ptr PCLPointCloud2Ptr;
  typedef PCLPointCloud2::ConstPtr PCLPointCloud2ConstPtr;

  using Filter<pcl::PCLPointCloud2>::removed_indices_;
  using Filter<pcl::PCLPointCloud2>::extract_removed_indices_;

public:
  /** \brief Constructor. */
  PassThroughUInt16(bool extract_removed_indices = false)
  : Filter<pcl::PCLPointCloud2>::Filter(extract_removed_indices),
    keep_organized_(false),
    user_filter_value_(std::numeric_limits<float>::quiet_NaN()),
    filter_field_name_(""),
    filter_limit_min_(0),
    filter_limit_max_(UINT16_MAX),
    filter_limit_negative_(false)
  {
    filter_name_ = "PassThroughUInt16";
  }

  /** \brief Set whether the filtered points should be kept and set to the
   * value given through \a setUserFilterValue (default: NaN), or removed
   * from the PointCloud, thus potentially breaking its organized
   * structure. By default, points are removed.
   *
   * \param[in] val set to true whether the filtered points should be kept and
   * set to a user given value (default: NaN)
   */
  inline void setKeepOrganized(bool val) { keep_organized_ = val; }

  /** \brief Obtain the value of the internal \a keep_organized_ parameter. */
  inline bool getKeepOrganized() { return (keep_organized_); }

  /** \brief Provide a value that the filtered points should be set to
   * instead of removing them.  Used in conjunction with \a
   * setKeepOrganized ().
   * \param[in] val the user given value that the filtered point dimensions should be set to
   */
  inline void setUserFilterValue(float val) { user_filter_value_ = val; }

  /** \brief Provide the name of the field to be used for filtering data. In conjunction with  \a setFilterLimits,
   * points having values outside this interval will be discarded.
   * \param[in] field_name the name of the field that contains values used for filtering
   */
  inline void setFilterFieldName(const std::string & field_name)
  {
    filter_field_name_ = field_name;
  }

  /** \brief Get the name of the field used for filtering. */
  inline std::string const getFilterFieldName() { return (filter_field_name_); }

  /** \brief Set the field filter limits. All points having field values outside this interval will be discarded.
   * \param[in] limit_min the minimum allowed field value
   * \param[in] limit_max the maximum allowed field value
   */
  inline void setFilterLimits(const std::uint16_t & limit_min, const std::uint16_t & limit_max)
  {
    filter_limit_min_ = limit_min;
    filter_limit_max_ = limit_max;
  }

  /** \brief Get the field filter limits (min/max) set by the user. The default values are -FLT_MAX, FLT_MAX. 
   * \param[out] limit_min the minimum allowed field value
   * \param[out] limit_max the maximum allowed field value
   */
  inline void getFilterLimits(std::uint16_t & limit_min, std::uint16_t & limit_max)
  {
    limit_min = filter_limit_min_;
    limit_max = filter_limit_max_;
  }

  /** \brief Set to true if we want to return the data outside the interval specified by setFilterLimits (min, max).
   * Default: false.
   * \param[in] limit_negative return data inside the interval (false) or outside (true)
   */
  inline void setFilterLimitsNegative(const bool limit_negative)
  {
    filter_limit_negative_ = limit_negative;
  }

  /** \brief Get whether the data outside the interval (min/max) is to be returned (true) or inside (false). 
   * \param[out] limit_negative true if data \b outside the interval [min; max] is to be returned, false otherwise
   */
  inline void getFilterLimitsNegative(bool & limit_negative)
  {
    limit_negative = filter_limit_negative_;
  }

  /** \brief Get whether the data outside the interval (min/max) is to be returned (true) or inside (false). 
   * \return true if data \b outside the interval [min; max] is to be returned, false otherwise
   */
  inline bool getFilterLimitsNegative() { return (filter_limit_negative_); }

protected:
  void applyFilter(PCLPointCloud2 & output);

private:
  /** \brief Keep the structure of the data organized, by setting the
   * filtered points to a user given value (NaN by default). 
   */
  bool keep_organized_;

  /** \brief User given value to be set to any filtered point. Casted to
   * the correct field type. 
   */
  float user_filter_value_;

  /** \brief The desired user filter field name. */
  std::string filter_field_name_;

  /** \brief The minimum allowed filter value a point will be considered from. */
  std::uint16_t filter_limit_min_;

  /** \brief The maximum allowed filter value a point will be considered from. */
  std::uint16_t filter_limit_max_;

  /** \brief Set to true if we want to return the data outside (\a filter_limit_min_;\a filter_limit_max_). Default: false. */
  bool filter_limit_negative_;
};
}  // namespace pcl

#ifdef PCL_NO_PRECOMPILE
#include "pcl/filters/impl/passthrough.hpp"
#endif

#endif  // PCL_FILTERS_PASSTHROUGH_H_
