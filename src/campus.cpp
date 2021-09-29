#include <chrono>
#include <iostream>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/circular_buffer.hpp>

#include <pcl/io/pcd_io.h>
#include<pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>

#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>

#include "imageProjection.h"
#include "featureAssociation.h"
namespace fs = boost::filesystem;

#ifdef USE_VGICP_CUDA
#include <fast_gicp/ndt/ndt_cuda.hpp>
#include <fast_gicp/gicp/fast_vgicp_cuda.hpp>
#endif
typedef pcl::PointXYZ GICP_PointType;
class KittiLoader {
public:
  KittiLoader(const std::string& dataset_path) : dataset_path(dataset_path) {
    ret.clear();
      struct dirent **namelist;

      int n = scandir(dataset_path.c_str(), &namelist, fileNameFilter, alphasort);

      for (int i = 0; i < n; ++i) {
        std::string filePath(namelist[i]->d_name);
        ret.push_back(filePath);
        free(namelist[i]);
      };
      free(namelist);
  }
  ~KittiLoader() {}

  size_t size() const { return ret.size(); }

  static int fileNameFilter(const struct dirent *cur) {
    std::string str(cur->d_name);
    if (str.find(".bin") != std::string::npos) {
      return 1;
    }
    return 0;
  }

  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(size_t i,string path) const {
    std::string filename = path + ret.at(i);
    FILE* file = fopen(filename.c_str(), "rb");
    if (!file) {
      std::cerr << "error: failed to load " << filename << std::endl;
      return nullptr;
    }

    std::vector<float> buffer(1000000);
    size_t num_points = fread(reinterpret_cast<char*>(buffer.data()), sizeof(float), buffer.size(), file) / 4;
    fclose(file);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    cloud->resize(num_points);

    for (int i = 0; i < num_points; i++) {
      auto& pt = cloud->at(i);
      if(abs(buffer[i * 4])<1000 and abs(buffer[i * 4 + 1])<1000 and abs(buffer[i * 4 + 2])<1000)
      {
        pt.x = buffer[i * 4 + 0];
        pt.y = buffer[i * 4 + 1];
        pt.z = buffer[i * 4 + 2];
      }else
      {
        pt.x = 0;
        pt.y = 0;
        pt.z = 0;
      }

      // pt.intensity = buffer[i * 4 + 3];
    }
    std::vector<int> indices;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud0(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::removeNaNFromPointCloud(*cloud, *cloud0, indices);
    return cloud0;
  }



private:
  int num_frames;
  std::string dataset_path;
  std::vector<std::string> ret;
};




int main(int argc, char** argv) {
  if (argc < 2) {
    std::cout << "usage: gicp_kitti /your/kitti/path/sequences/00/velodyne" << std::endl;
    //return 0;
  }
  //sleep(5);

  string sequence(argv[1]);
  string res0(argv[2]);
  string res1(argv[3]);
  string use_svd(argv[4]);//0:not use svd;1:use svd
  string use_linear(argv[5]);//0:plane;1:linear;2:raw
  string use_prob_kernal(argv[6]);//0:not use; 1:use
  string use_fec(argv[7]);//0:not use; 1:use
  string use_sampling(argv[8]);
  string scan2model(argv[9]);
  string thre(argv[10]);
  //string sequence("01");
  /*
  string sequence = "pcd";
  string res0 = "0.15";
  string res1 = "0.15";
  string use_svd = "1";//0:not use svd;1:use svd
  string use_linear = "0";//0:plane;1:linear;2:raw
  string use_prob_kernal = "0";//0:not use; 1:use
  string use_fec = "0";//0:not use; 1:use
  string use_sampling = "0";
  string scan2model  = "s";
  string thre ="1";*/

  string path  = "/home/wjk/Dataset/stevens/" + sequence + "/";


  vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> transpose;

  vector<double> Time_stamp;



  KittiLoader kitti(path);

  cout<<"start gicp "<<atof(res0.c_str())<<" "<<atof(res1.c_str())<<endl;
  double downsample_resolution = atof(res0.c_str()),downsample_resolution0 = atof(res1.c_str());
  pcl::ApproximateVoxelGrid<pcl::PointXYZ> voxelgrid,voxelgrid0,voxelgrid1;
  voxelgrid.setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
  voxelgrid0.setLeafSize(downsample_resolution0, downsample_resolution0, downsample_resolution0);

  // registration method
  // you should fine-tune hyper-parameters (e.g., voxel resolution, max correspondence distance) for the best result
  fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ> gicp;
  gicp.use_svd_ = (use_svd=="1");
  gicp.use_kernal_ = (use_prob_kernal=="1");
  gicp.use_fec_ = (use_fec=="1");
  gicp.use_linear_ = (use_linear=="1");
  gicp.use_sampling_ = (use_sampling=="1");
  gicp.ave_filter_no_ = 0;
  gicp.rejected_points_no_ = 0;

  gicp.truncted_thre_ = atof(thre.c_str());
  gicp.setMaxCorrespondenceDistance(1.0);
  if(use_svd=="1")
    gicp.setCorrespondenceRandomness(20);
  else
    gicp.setCorrespondenceRandomness(20);
  gicp.setRegularizationMethod(fast_gicp::RegularizationMethod::PLANE);
  //set initial frame as target


  cout<<"set initial frame as target"<<endl;
  int start_ind_ = 0;
  voxelgrid.setInputCloud(kitti.cloud(start_ind_,path));
  voxelgrid0.setInputCloud(kitti.cloud(start_ind_,path));
  pcl::PointCloud<pcl::PointXYZ>::Ptr target(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr target0(new pcl::PointCloud<pcl::PointXYZ>);
  //voxelgrid.filter(*target);
  voxelgrid0.filter(*target0);
  //IP.cloudHandler(kitti.cloud(0));
  //gicp.setTargetsortedCloud(target);
  gicp.setTargetsortedCloud(target0);

  //pcl::PointCloud<pcl::PointXYZ>::Ptr target_sortedcloud(new pcl::PointCloud<pcl::PointXYZ>);
  //pcl::copyPointCloud(*IP.fullLinearPoints_,*target_sortedcloud);
  //gicp.setInputTarget(target_sortedcloud); //
  gicp.setInputTarget(target0);
  // sensor pose sequence
  std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses(kitti.size());
  poses[0].setIdentity();
  vector<pcl::PointCloud<pcl::PointXYZ>> frames_(kitti.size());
  frames_[0] = *target0;
  // trajectory for visualization
  pcl::PointCloud<pcl::PointXYZ>::Ptr trajectory(new pcl::PointCloud<pcl::PointXYZ>);
  trajectory->push_back(pcl::PointXYZ(0.0f, 0.0f, 0.0f));

  pcl::visualization::PCLVisualizer vis;
  vis.addPointCloud<pcl::PointXYZ>(trajectory, "trajectory");

  // for calculating FPS
  boost::circular_buffer<std::chrono::high_resolution_clock::time_point> stamps(30);
  stamps.push_back(std::chrono::high_resolution_clock::now());
  Eigen::Matrix4d M;
  Eigen::Vector2d Errors_;
  string path0 = "/home/wjk/Fast_gicp_odometry/fast_gicp-master/tmp/errors_" + sequence + "_" + res0 + "_" + res1 + "_" + use_svd +  "_" + use_linear+  "_" + use_prob_kernal +   "_" + use_fec +   "_" + use_sampling +   "_" + scan2model+ "_" + thre+ ".txt";
  string path1 = "/home/wjk/Fast_gicp_odometry/fast_gicp-master/tmp/poseerrors_" + sequence + "_" + res0 + "_" + res1 + "_" + use_svd + "_" + use_linear +  "_" + use_prob_kernal+   "_" + use_fec + "_" + use_sampling + "_" + scan2model+"_" + thre+ ".txt";
  std::ofstream ofs_error(path0.c_str());
  std::ofstream pose_error(path1.c_str());

  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> local_frames;
  pcl::PointCloud<pcl::PointXYZ>::Ptr local_map_(new pcl::PointCloud<pcl::PointXYZ>);
  local_frames.clear();
  bool local_optimization_ = false;
  pcl::PointCloud<pcl::PointXYZ>::Ptr target_localmap(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  vector<Eigen::Matrix4d> local_poses_;
  local_poses_.clear();
  Eigen::Matrix4d M0 = Eigen::Matrix4d::Identity(),M_gt;
  int keyframe_no = 10;
  bool scan_to_model_ = (scan2model.find("m")==0);
  for (int i = start_ind_+1; i < kitti.size(); i++) {
    //cout<<"i : "<<i<<endl;
    gicp.index_ = i;
    voxelgrid.setInputCloud(kitti.cloud(i,path));


    pcl::PointCloud<pcl::PointXYZ>::Ptr source(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr source0(new pcl::PointCloud<pcl::PointXYZ>);
    voxelgrid.filter(*source);
    voxelgrid0.setInputCloud(kitti.cloud(i,path));
    voxelgrid0.filter(*source0);
    //if(source0->size()<1000)
    //  continue;
    //IP.cloudHandler(kitti.cloud(i));

    if(scan_to_model_)
    {
      if(i-start_ind_>50)
      {
        if(local_frames.size()<keyframe_no)
        {
          local_frames.push_back(source0);
        }else
        {
          local_frames.erase(local_frames.begin());
          local_frames.push_back(source0);
        }
      }
      if(local_frames.size()==keyframe_no)
      {
        local_map_->clear();
        M0.setIdentity();
        int count = 1;
        for(int k = local_frames.size()-3; k > 0 ; --k)
        {
          M0 = M0 * local_poses_.at(local_poses_.size()-count).inverse();
          pcl::transformPointCloud(*local_frames.at(k),*transformed_cloud,M0);
          *local_map_ += *transformed_cloud;
          ++count;
        }
        *local_map_ += *local_frames.at(local_frames.size()-2);
        target_localmap->clear();
        //cout<<"local_map_ size before filter: "<<local_map_->size()<<endl;
        voxelgrid0.setInputCloud(local_map_);
        voxelgrid0.filter(*target_localmap);
        //cout<<"local_map_ size after filter: "<<target_localmap->size()<<endl;
        local_optimization_ = true;
        gicp.setInputTarget(target_localmap);
        gicp.setTargetsortedCloud(target_localmap);
        bool visualization = false;
        if(visualization)
        {
          boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));

          //pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> fildColor(target_localmap, "z");

          pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(target_localmap, 0, 255, 0);
          viewer->addPointCloud<pcl::PointXYZ>(target_localmap, single_color, "sample cloud");
          viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud"); // 设置点云大小
          pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_color(source0, 255, 0, 0);

          //pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> fildColor0(target_localmap, "x");
          viewer->addPointCloud<pcl::PointXYZ>(source0, source_color, "source cloud");
          viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "source cloud"); // 设置点云大小

          while (!viewer->wasStopped())
          {
            viewer->spinOnce(100);
            boost::this_thread::sleep(boost::posix_time::microseconds(100000));
          }
        }
      }
    }


    gicp.setSourceSortedCloud(source);
    gicp.setInputSource(source0);
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);

    if(i == start_ind_+1)
    {
      M = Eigen::Matrix4d::Identity();
      gicp.setInit_Guess(M);
    }else{
      gicp.setInit_Guess(M);
    }

    gicp.align(*aligned);
    Errors_ = gicp.get_matchingerror();
    ofs_error<<Errors_[0]<<" "<<Errors_[1]<<endl;
    if(scan_to_model_)
    {
      if(!local_optimization_)
        gicp.swapSourceAndTarget();
    }else
        gicp.swapSourceAndTarget();


    if(gicp.is_converged_)
      M = gicp.getFinalTransformation().cast<double>();
    // accumulate pose


    bool visualization0 = false;
    if(visualization0 and i > 100)
    {
      boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));

      //pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> fildColor(target_localmap, "z");

      pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(target_localmap, 0, 255, 0);
      viewer->addPointCloud<pcl::PointXYZ>(target_localmap, single_color, "sample cloud");
      viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud"); // 设置点云大小
      pcl::transformPointCloud(*source0,*transformed_cloud,M);
      pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_color(transformed_cloud, 255, 0, 0);

      //pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> fildColor0(target_localmap, "x");
      viewer->addPointCloud<pcl::PointXYZ>(transformed_cloud, source_color, "source cloud");
      viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "source cloud"); // 设置点云大小

      while (!viewer->wasStopped())
      {
        viewer->spinOnce(100);
        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
      }
    }


    poses[i-start_ind_] = poses[i - start_ind_-1] * M ;
    frames_[i-start_ind_] = *source0;
    local_poses_.push_back(M);

    // FPS display
    stamps.push_back(std::chrono::high_resolution_clock::now());

    // visualization
    trajectory->push_back(pcl::PointXYZ(poses[i-start_ind_](0, 3), poses[i-start_ind_](1, 3), poses[i-start_ind_](2, 3)));
    vis.updatePointCloud<pcl::PointXYZ>(trajectory, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(trajectory, 255.0, 0.0, 0.0), "trajectory");
    vis.spinOnce();
  }
  std::cout << stamps.size() / (std::chrono::duration_cast<std::chrono::nanoseconds>(stamps.back() - stamps.front()).count() / 1e9) << "fps" << std::endl;
  pose_error<<stamps.size() / (std::chrono::duration_cast<std::chrono::nanoseconds>(stamps.back() - stamps.front()).count() / 1e9)<<endl;
  pose_error<<(double)gicp.ave_filter_no_/(kitti.size()-1)<<" "<<(double)gicp.rejected_points_no_/(kitti.size()-1)<<endl;
  cout<<"ave filter no "<<(double)gicp.ave_filter_no_/(kitti.size()-1)<<" "<<(double)gicp.rejected_points_no_/(kitti.size()-1)<<endl;
  ofs_error.close();
  pose_error.close();
  // save the estimated poses
  string path2 = "/home/wjk/Fast_gicp_odometry/fast_gicp-master/tmp/traj" + sequence + "_" + res0 + "_" + res1 + "_" + use_svd + "_" + use_linear+  "_" + use_prob_kernal +   "_" + use_fec + "_" + use_sampling + "_" + scan2model+ "_" + thre +".txt";
  std::ofstream ofs(path2.c_str());
  for (const auto& pose : poses) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 4; j++) {
        if (i || j) {
          ofs << " ";
        }
        ofs << pose(i, j);
      }
    }
    ofs << std::endl;
  }

  cout<<"generate map"<<endl;
  pcl::PointCloud<pcl::PointXYZ>::Ptr mapPointCloud_(new pcl::PointCloud<pcl::PointXYZ>())
    ,Map_(new pcl::PointCloud<pcl::PointXYZ>());
  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_PointCloud_(new pcl::PointCloud<pcl::PointXYZ>());
  int start_index_ = 40;
  Eigen::Matrix4d T_;
  pcl::ApproximateVoxelGrid<pcl::PointXYZ> voxelgrid_map_;
  voxelgrid_map_.setLeafSize(3,3,3);
  cout<<"pose and frame no: "<<poses.size()<<" "<<frames_.size()<<endl;
  ImageProjection IP_;
  Eigen::Matrix4d M_init = poses[0].matrix();
  for (int i = start_index_; i < frames_.size(); i = i + 5)
  {
    //cout<<"i "<<i<<endl;

    Eigen::Matrix4d M1 = poses[i].matrix();
    T_ = M1 * M_init.inverse();
    //IP_.resetParameters();
    //IP_.cloudHandler(frames_[i]);
    pcl::transformPointCloud(frames_[i],*transformed_PointCloud_,T_);
    *mapPointCloud_ += *transformed_PointCloud_;
  }
  voxelgrid_map_.setInputCloud(mapPointCloud_);
  voxelgrid_map_.filter(*Map_);
  string map_save_path = "/home/wjk/Fast_gicp_odometry/fast_gicp-master/tmp/Map" + sequence + "_" + use_sampling + ".pcd";
  pcl::io::savePCDFileASCII(map_save_path, *Map_);
  cout<<"map size: "<<mapPointCloud_->size()<<" "<<Map_->size()<<endl;

  bool visualization0 = true;
  if(visualization0)
  {
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));

    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> fildColor( Map_,"z");

    viewer->addPointCloud<pcl::PointXYZ>(Map_, fildColor, "sample cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 0.1, "sample cloud"); // 设置点云大小

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(target_localmap, 255, 0, 0);

    viewer->addPointCloud<pcl::PointXYZ>(trajectory, single_color, "source cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "source cloud");

    while (!viewer->wasStopped())
    {
      viewer->spinOnce(100);
      boost::this_thread::sleep(boost::posix_time::microseconds(100000));
    }
  }


  return 0;
}