#ifndef FAST_GICP_FAST_GICP_HPP
#define FAST_GICP_FAST_GICP_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <pcl/registration/registration.h>
#include <fast_gicp/gicp/lsq_registration.hpp>
#include <fast_gicp/gicp/gicp_settings.hpp>

namespace fast_gicp {

/**
 * @brief Fast GICP algorithm optimized for multi threading with OpenMP
 */
template<typename PointSource, typename PointTarget>
class FastGICP : public LsqRegistration<PointSource, PointTarget> {
public:
  using Scalar = float;
  using Matrix4 = typename pcl::Registration<PointSource, PointTarget, Scalar>::Matrix4;

  using PointCloudSource = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudSource;
  using PointCloudSourcePtr = typename PointCloudSource::Ptr;
  using PointCloudSourceConstPtr = typename PointCloudSource::ConstPtr;

  using PointCloudTarget = typename pcl::Registration<PointSource, PointTarget, Scalar>::PointCloudTarget;
  using PointCloudTargetPtr = typename PointCloudTarget::Ptr;
  using PointCloudTargetConstPtr = typename PointCloudTarget::ConstPtr;

  void setSourceSortedCloud(typename pcl::PointCloud<PointSource>::Ptr source_sortedcloud);
  void setTargetsortedCloud(typename pcl::PointCloud<PointTarget>::Ptr target_sortedcloud);

  double compute_matching_error(Eigen::Isometry3d& trans);
  void setInit_Guess(Eigen::Matrix4d guess);
  Eigen::Vector2d get_matchingerror();
  std::vector<std::vector<Eigen::Vector3d>> source_neighbor_set_;
  std::vector<std::vector<Eigen::Vector3d>> target_neighbor_set_;
  Eigen::Matrix4d init_guess_ = Eigen::Matrix4d::Identity();
  double first_matching_error_;
  double last_matching_error_;
  int iter_times_;
  int index_;
  bool use_svd_;
  bool use_kernal_;
  bool use_fec_;
  bool use_linear_;
  bool use_sampling_;
  double ave_filter_no_;
  double ave_point_;
  double rejected_points_no_;
  double rejected_points;
  double truncted_thre_;
  std::vector<int> point_flag_;
  typename pcl::PointCloud<PointSource>::Ptr source_sortedcloud_;
  typename pcl::PointCloud<PointTarget>::Ptr target_sortedcloud_;

protected:
  using pcl::Registration<PointSource, PointTarget, Scalar>::reg_name_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::input_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::target_;
  using pcl::Registration<PointSource, PointTarget, Scalar>::corr_dist_threshold_;

public:
  FastGICP();
  virtual ~FastGICP() override;



  void setNumThreads(int n);
  void setCorrespondenceRandomness(int k);
  void setRegularizationMethod(RegularizationMethod method);

  virtual void swapSourceAndTarget() override;
  virtual void clearSource() override;
  virtual void clearTarget() override;

  virtual void setInputSource(const PointCloudSourceConstPtr& cloud) override;
  virtual void setSourceCovariances(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covs);
  virtual void setInputTarget(const PointCloudTargetConstPtr& cloud) override;
  virtual void setTargetCovariances(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covs);

  const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& getSourceCovariances() const {
    return source_covs_;
  }

  const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& getTargetCovariances() const {
    return target_covs_;
  }
  template<typename PointT>
  void local_covariances(const typename pcl::PointCloud<PointT>::ConstPtr& cloud, const typename pcl::PointCloud<PointT>::ConstPtr& cloud_normal, std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances);

protected:
  virtual void computeTransformation(PointCloudSource& output, const Matrix4& guess) override;

  virtual void update_correspondences(const Eigen::Isometry3d& trans);

  virtual double linearize(const Eigen::Isometry3d& trans, Eigen::Matrix<double, 6, 6>* H, Eigen::Matrix<double, 6, 1>* b) override;

  virtual double compute_error(const Eigen::Isometry3d& trans) override;

  template<typename PointT>
  bool calculate_covariances(const typename pcl::PointCloud<PointT>::ConstPtr& cloud, pcl::search::KdTree<PointT>& kdtree,
                             std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances,
                             typename pcl::PointCloud<PointSource>::Ptr sorted_cloud_);
  template<typename PointT>
  bool calculate_covariances(const typename pcl::PointCloud<PointT>::ConstPtr& cloud, pcl::search::KdTree<PointT>& kdtree,
                             std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances);



protected:
  int num_threads_;
  int k_correspondences_;

  pcl::PointCloud<pcl::PointXYZ>::Ptr source_normals_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr target_normals_;

  pcl::PointCloud<pcl::PointXYZ>::Ptr source_linear_cloud_;
  pcl::PointCloud<pcl::PointXYZ>::Ptr target_linear_cloud_;

  RegularizationMethod regularization_method_;

  std::shared_ptr<pcl::search::KdTree<PointSource>> source_kdtree_;
  std::shared_ptr<pcl::search::KdTree<PointTarget>> target_kdtree_;

  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> source_covs_;
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> target_covs_;

  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> mahalanobis_;

  std::vector<int> correspondences_;
  std::vector<float> sq_distances_;


  bool calculate_linear_covariances(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                                    pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_normal_,
                                    std::vector<Eigen::Matrix4d,Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances);

};
}  // namespace fast_gicp

#endif