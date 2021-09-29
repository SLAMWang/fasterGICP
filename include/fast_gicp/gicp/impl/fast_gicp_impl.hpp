#ifndef FAST_GICP_FAST_GICP_IMPL_HPP
#define FAST_GICP_FAST_GICP_IMPL_HPP
#include "ctime"
#include <fast_gicp/so3/so3.hpp>
#include "utility.h"
namespace fast_gicp {

template <typename PointSource, typename PointTarget>
FastGICP<PointSource, PointTarget>::FastGICP() {
#ifdef _OPENMP
  num_threads_ = omp_get_max_threads();
#else
  num_threads_ = 1;
#endif

  k_correspondences_ = 20;
  iter_times_ = 0;
  reg_name_ = "FastGICP";
  corr_dist_threshold_ = std::numeric_limits<float>::max();

  regularization_method_ = RegularizationMethod::NONE;
  source_kdtree_.reset(new pcl::search::KdTree<PointSource>);
  target_kdtree_.reset(new pcl::search::KdTree<PointTarget>);
}

template <typename PointSource, typename PointTarget>
FastGICP<PointSource, PointTarget>::~FastGICP() {}

template <typename PointSource, typename PointTarget>
void FastGICP<PointSource, PointTarget>::setNumThreads(int n) {
  num_threads_ = n;

#ifdef _OPENMP
  if (n == 0) {
    num_threads_ = omp_get_max_threads();
  }
#endif
}

template <typename PointSource, typename PointTarget>
void FastGICP<PointSource, PointTarget>::setCorrespondenceRandomness(int k) {
  k_correspondences_ = k;
}

template <typename PointSource, typename PointTarget>
void FastGICP<PointSource, PointTarget>::setRegularizationMethod(RegularizationMethod method) {
  regularization_method_ = method;
}

template <typename PointSource, typename PointTarget>
void FastGICP<PointSource, PointTarget>::swapSourceAndTarget() {
  input_.swap(target_);
  //pcl::copyPointCloud(target_sortedcloud_,target_);
  source_kdtree_.swap(target_kdtree_);
  source_covs_.swap(target_covs_);
  source_sortedcloud_.swap(target_sortedcloud_);
  correspondences_.clear();
  sq_distances_.clear();
}

template <typename PointSource, typename PointTarget>
void FastGICP<PointSource, PointTarget>::clearSource() {
  input_.reset();
  source_covs_.clear();
}

template <typename PointSource, typename PointTarget>
void FastGICP<PointSource, PointTarget>::clearTarget() {
  target_.reset();
  target_covs_.clear();
}

template <typename PointSource, typename PointTarget>
void FastGICP<PointSource, PointTarget>::setInputSource(const PointCloudSourceConstPtr& cloud) {
  if (input_ == cloud) {
    return;
  }

  pcl::Registration<PointSource, PointTarget, Scalar>::setInputSource(cloud);
  source_kdtree_->setInputCloud(cloud);
  source_covs_.clear();
}

template <typename PointSource, typename PointTarget>
void FastGICP<PointSource, PointTarget>::setInputTarget(const PointCloudTargetConstPtr& cloud) {
  if (target_ == cloud) {
    //cout<<"target_ == cloud"<<endl;
    return;
  }
  ////////cout<<"setInputTarget(cloud)"<<endl;
  pcl::Registration<PointSource, PointTarget, Scalar>::setInputTarget(cloud);
  target_kdtree_->setInputCloud(cloud);
  target_covs_.clear();
}
template <typename PointSource, typename PointTarget>
void FastGICP<PointSource, PointTarget>::setSourceSortedCloud(typename pcl::PointCloud<PointSource>::Ptr source_sortedcloud){
  source_sortedcloud_ = source_sortedcloud;
}
template <typename PointSource, typename PointTarget>
void FastGICP<PointSource, PointTarget>::setTargetsortedCloud(typename pcl::PointCloud<PointTarget>::Ptr target_sortedcloud){

  target_sortedcloud_ = target_sortedcloud;
}

template <typename PointSource, typename PointTarget>
void FastGICP<PointSource, PointTarget>::setSourceCovariances(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covs) {
  source_covs_ = covs;
}

template <typename PointSource, typename PointTarget>
void FastGICP<PointSource, PointTarget>::setTargetCovariances(const std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covs) {
  target_covs_ = covs;
}

template <typename PointSource, typename PointTarget>
void FastGICP<PointSource, PointTarget>::setInit_Guess(Eigen::Matrix4d guess)
{
  init_guess_ = guess;
}
template <typename PointSource, typename PointTarget>
Eigen::Vector2d FastGICP<PointSource, PointTarget>::get_matchingerror(){
  return Eigen::Vector2d(first_matching_error_,last_matching_error_);
}

template <typename PointSource, typename PointTarget>
void FastGICP<PointSource, PointTarget>::computeTransformation(PointCloudSource& output, const Matrix4& guess) {

  if (target_covs_.size() != target_->size()) {
    calculate_covariances(target_, *target_kdtree_, target_covs_,target_sortedcloud_);
  }
  if (source_covs_.size() != input_->size()) {
    calculate_covariances(input_, *source_kdtree_, source_covs_,source_sortedcloud_);
  }
  //Eigen::Isometry3d x0 = Eigen::Isometry3d(init_guess_);
  //cout<<"compute matching error "<<endl;
  //update_correspondences(x0);
  //first_matching_error_ = compute_error(x0);
  //cout<<"matching error before optimization: "<<first_matching_error_<<endl;
  Matrix4 M = init_guess_.cast<float>();
  LsqRegistration<PointSource, PointTarget>::computeTransformation(output, M);
  //Eigen::Isometry3d x1 = Eigen::Isometry3d(LsqRegistration<PointSource, PointTarget>::final_transformation_. template cast<double>());
  //update_correspondences(x1);
  //last_matching_error_ = compute_error(x1);

  iter_times_ = 0;
}

template <typename PointSource, typename PointTarget>
void FastGICP<PointSource, PointTarget>::update_correspondences(const Eigen::Isometry3d& trans) {
  assert(source_covs_.size() == input_->size());
  assert(target_covs_.size() == target_->size());
  //cout<<"trans"<<endl;
  //cout<<trans.matrix()<<endl;
  if(iter_times_<1)
    target_kdtree_->setInputCloud(target_);
  Eigen::Isometry3f trans_f = trans.cast<float>();
    if(iter_times_==0)
    {
      correspondences_.resize(input_->size());
      sq_distances_.resize(input_->size());
      mahalanobis_.resize(input_->size());
    }

  std::vector<int> k_indices(1);
  std::vector<float> k_sq_dists(1);


#pragma omp parallel for num_threads(num_threads_) firstprivate(k_indices, k_sq_dists) schedule(guided, 8)
  for (int i = 0; i < input_->size(); i++) {
    int step_fec;
    if(use_fec_)
      step_fec = 1;
    else
      step_fec = 10;
    PointTarget pt;
    pt.getVector4fMap() = trans_f * input_->at(i).getVector4fMap();

    target_kdtree_->nearestKSearch(pt, 1, k_indices, k_sq_dists);

    sq_distances_[i] = k_sq_dists[0];
    correspondences_[i] = k_sq_dists[0] < corr_dist_threshold_ * corr_dist_threshold_ ? k_indices[0] : -1;

    if (correspondences_[i] < 0) {
      continue;
    }
      if(iter_times_<step_fec or index_<50)
      {
        const int target_index = correspondences_[i];
        const auto& cov_A = source_covs_[i];
        const auto& cov_B = target_covs_[target_index];


        Eigen::Matrix4d RCR = cov_B + trans.matrix() * cov_A * trans.matrix().transpose();

        RCR(3, 3) = 1.0;
        mahalanobis_[i] = RCR.inverse();

        mahalanobis_[i](3, 3) = 0.0f;
      }
  }
}

template <typename PointSource, typename PointTarget>
double FastGICP<PointSource, PointTarget>::linearize(const Eigen::Isometry3d& trans, Eigen::Matrix<double, 6, 6>* H, Eigen::Matrix<double, 6, 1>* b) {

  update_correspondences(trans);;
  double sum_errors = 0.0,e(0.0);
  std::vector<Eigen::Matrix<double, 6, 6>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 6>>> Hs(num_threads_);
  std::vector<Eigen::Matrix<double, 6, 1>, Eigen::aligned_allocator<Eigen::Matrix<double, 6, 1>>> bs(num_threads_);
  for (int i = 0; i < num_threads_; i++) {
    Hs[i].setZero();
    bs[i].setZero();
  }
  rejected_points = 0;
  float rho;
  for (int i = 0; i < input_->size(); i++) {
    int N = 99;
    int target_index = correspondences_[i];
    if (target_index < 0) {
      continue;
    }

    const Eigen::Vector4d mean_A = input_->at(i).getVector4fMap().template cast<double>();
    const auto& cov_A = source_covs_[i];

    const Eigen::Vector4d mean_B = target_->at(target_index).getVector4fMap().template cast<double>();
    const auto& cov_B = target_covs_[target_index];

    const Eigen::Vector4d transed_mean_A = trans * mean_A;
    const Eigen::Vector4d error = mean_B - transed_mean_A;
    e = error.transpose() * mahalanobis_[i] * error;


    if(use_kernal_ and index_>50)
    {
      float random = rand()%(N+1)/(float)(N+1);

      float d = exp(-1*pow(e,2)/0.01);//0.01
      if(random < d and index_>50)
      {
        if(iter_times_<1)
          rejected_points += 1;
        continue;
      }
    }
    if(use_sampling_)
    {
      if(point_flag_[i]==-1)
        continue;
    }

    sum_errors += e;

    if (H == nullptr || b == nullptr) {
      continue;
    }

    Eigen::Matrix<double, 4, 6> dtdx0 = Eigen::Matrix<double, 4, 6>::Zero();
    dtdx0.block<3, 3>(0, 0) = skewd(transed_mean_A.head<3>());
    dtdx0.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();

    Eigen::Matrix<double, 4, 6> jlossexp = dtdx0;

    Eigen::Matrix<double, 6, 6> Hi = jlossexp.transpose() * mahalanobis_[i] * jlossexp;
    Eigen::Matrix<double, 6, 1> bi = jlossexp.transpose() * mahalanobis_[i] * error;

    Hs[omp_get_thread_num()] += Hi;
    bs[omp_get_thread_num()] += bi;
  }

  if (H && b) {
    H->setZero();
    b->setZero();
    for (int i = 0; i < num_threads_; i++) {
      (*H) += Hs[i];
      (*b) += bs[i];
    }
  }
  iter_times_++;

  return sum_errors;
}

template <typename PointSource, typename PointTarget>
double FastGICP<PointSource, PointTarget>::compute_error(const Eigen::Isometry3d& trans) {
  double sum_errors = 0.0;

#pragma omp parallel for num_threads(num_threads_) reduction(+ : sum_errors) schedule(guided, 8)
  for (int i = 0; i < input_->size(); i++) {

    int target_index = correspondences_[i];
    if (target_index < 0) {
      continue;
    }

    const Eigen::Vector4d mean_A = input_->at(i).getVector4fMap().template cast<double>();
    const auto& cov_A = source_covs_[i];

    const Eigen::Vector4d mean_B = target_->at(target_index).getVector4fMap().template cast<double>();
    const auto& cov_B = target_covs_[target_index];

    const Eigen::Vector4d transed_mean_A = trans * mean_A;
    const Eigen::Vector4d error = mean_B - transed_mean_A;

    sum_errors += error.transpose() * mahalanobis_[i] * error;

  }

  return sum_errors;
}

Eigen::Vector2i getRowCol(Eigen::Vector3d p)
{
  float verticalAngle, horizonAngle, range;
  size_t rowIdn, columnIdn, index, cloudSize;

  verticalAngle = atan2(p[2], sqrt(p[0] * p[0] + p[1] * p[1])) * 180 / M_PI;
  rowIdn = (verticalAngle + ang_bottom) / ang_res_y;
  horizonAngle = atan2(p[0], p[1]) * 180 / M_PI;
  columnIdn = -round((horizonAngle - 90.0) / ang_res_x) + Horizon_SCAN / 2;
  if (columnIdn >= Horizon_SCAN)
    columnIdn -= Horizon_SCAN;
  return Eigen::Vector2i(rowIdn,columnIdn);
}



template <typename PointSource, typename PointTarget>
template <typename PointT>
bool FastGICP<PointSource, PointTarget>::calculate_covariances(
  const typename pcl::PointCloud<PointT>::ConstPtr& cloud,
  pcl::search::KdTree<PointT>& kdtree,
  std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>& covariances,
  typename pcl::PointCloud<PointSource>::Ptr sorted_cloud_) {
  if (kdtree.getInputCloud() != sorted_cloud_) {
    kdtree.setInputCloud(sorted_cloud_);
  }
  covariances.resize(cloud->size());
  point_flag_.resize(cloud->size());
  bool use_kdtree = true;
  kdtree.setInputCloud(sorted_cloud_);
  int filter_no_(0);
  /*
  string path1,path2;
  path1 = "/home/wjk/Fast_gicp_odometry/fast_gicp-master/tmp/total_points.txt";
  path2 = "/home/wjk/Fast_gicp_odometry/fast_gicp-master/tmp/sampled_points.txt";
  std::ofstream ofs1(path1.c_str());
  std::ofstream ofs2(path2.c_str());*/
#pragma omp parallel for num_threads(num_threads_) schedule(guided, 8)
  for (int i = 0; i < cloud->size(); i++) {
    Eigen::Matrix4d cov;
    Eigen::Matrix3d U_matrix;
    if(use_kdtree)
    {
      std::vector<int> k_indices;
      std::vector<float> k_sq_distances;
      Eigen::Vector3d p_a,p_b,p_c,a,b,c;
      kdtree.nearestKSearch(cloud->at(i), k_correspondences_, k_indices, k_sq_distances);
      Eigen::Matrix<double, 4, -1> neighbors(4, k_correspondences_);

      if(use_svd_) {
      for (int j = 0; j < k_correspondences_; j++) {
            neighbors.col(j) = sorted_cloud_->at(k_indices[j]).getVector4fMap().template cast<double>();
        }
        neighbors.colwise() -= neighbors.rowwise().mean().eval();
        cov = neighbors * neighbors.transpose() / neighbors.cols();
      }else{
        p_a = sorted_cloud_->at(k_indices[0]).getVector3fMap().template cast<double>();
        p_b = sorted_cloud_->at(k_indices[k_correspondences_-2]).getVector3fMap().template cast<double>();
        p_c = sorted_cloud_->at(k_indices[k_correspondences_-1]).getVector3fMap().template cast<double>();
        a = p_c - p_a;
        b = p_b - p_a;
        p_c = a.cross(b);
        p_b = a.cross(p_c);
        p_b = p_b/p_b.norm();
        U_matrix.col(0) = a/a.norm();
        U_matrix.col(1) = p_b;//p_
        U_matrix.col(2) = p_c/p_c.norm();
      }
      }
    if (regularization_method_ == RegularizationMethod::NONE) {
      covariances[i] = cov;
    } else if (regularization_method_ == RegularizationMethod::FROBENIUS) {
      double lambda = 1e-3;
      Eigen::Matrix3d C = cov.block<3, 3>(0, 0).cast<double>() + lambda * Eigen::Matrix3d::Identity();
      Eigen::Matrix3d C_inv = C.inverse();
      covariances[i].setZero();
      covariances[i].template block<3, 3>(0, 0) = (C_inv / C_inv.norm()).inverse();
    } else {

      Eigen::Vector3d values;

      switch (regularization_method_) {
        default:
          std::cerr << "here must not be reached" << std::endl;
          abort();
        case RegularizationMethod::PLANE:
          // cout<<"Plane distribution"<<endl;
          if(use_linear_)
            values = Eigen::Vector3d(1, 0.1, 1e-3);
          else
            values = Eigen::Vector3d(1, 1, 1e-3);
          break;
        case RegularizationMethod::MIN_EIG:
          // cout<<"MIN_EIG"<<endl;
          //values = svd.singularValues().array().max(1e-3);
          break;
        case RegularizationMethod::NORMALIZED_MIN_EIG:
          // cout<<"Normalized_min_eig"<<endl;
          //values = svd.singularValues() / svd.singularValues().maxCoeff();
          values = values.array().max(1e-3);
          break;
      }

      covariances[i].setZero();
      if (use_svd_)
      {
        int N  = 99;
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov.block<3, 3>(0, 0), Eigen::ComputeFullU | Eigen::ComputeFullV);
        covariances[i].template block<3, 3>(0, 0) = svd.matrixU() * values.asDiagonal() * svd.matrixV().transpose();
        values = svd.singularValues() / svd.singularValues().maxCoeff();
        if(use_sampling_)
        {
          float random = rand()%(N+1)/(float)(N+1);
          float d = exp(-1*pow(values[2],2)/0.25);  //0.25 for kitti
          if(random > d and index_>50)
          {
            point_flag_[i] = -1;
            filter_no_++;
          }else
          {
            point_flag_[i] = 1;
          }
        }
      }
      else
      {
        covariances[i].template block<3, 3>(0, 0) = U_matrix * values.asDiagonal() * U_matrix.transpose();
      }
    }
  }
  if(index_>50)
  {
    ave_filter_no_ += filter_no_;
    ave_point_ += point_flag_.size();
  }
  return true;
}

}  // namespace fast_gicp

#endif
