#include <chrono>
#include <iostream>
#include <boost/format.hpp>
#include <boost/filesystem.hpp>
#include <boost/circular_buffer.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>

#include <fast_gicp/gicp/fast_gicp.hpp>
#include <fast_gicp/gicp/fast_vgicp.hpp>


#include "imageProjection.h"
//#include "featureAssociation.h"


#ifdef USE_VGICP_CUDA
#include <fast_gicp/ndt/ndt_cuda.hpp>
#include <fast_gicp/gicp/fast_vgicp_cuda.hpp>
#endif
typedef pcl::PointXYZ GICP_PointType;

class KittiLoader {
public:
    KittiLoader(const std::string &dataset_path) : dataset_path(dataset_path) {
        for (num_frames = 0;; num_frames++) {
            std::string filename = (boost::format("%s/%06d.bin") % dataset_path % num_frames).str();
            if (!boost::filesystem::exists(filename)) {
                break;
            }
        }

        if (num_frames == 0) {
            std::cerr << "error: no files in " << dataset_path << std::endl;
        }
    }

    ~KittiLoader() {}

    size_t size() const { return num_frames; }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(size_t i) const {
        std::string filename = (boost::format("%s/%06d.bin") % dataset_path % i).str();
        FILE *file = fopen(filename.c_str(), "rb");
        if (!file) {
            std::cerr << "error: failed to load " << filename << std::endl;
            return nullptr;
        }

        std::vector<float> buffer(1000000);
        size_t num_points = fread(reinterpret_cast<char *>(buffer.data()), sizeof(float), buffer.size(), file) / 4;
        fclose(file);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        cloud->resize(num_points);

        for (int i = 0; i < num_points; i++) {
            auto &pt = cloud->at(i);
            pt.x = buffer[i * 4];
            pt.y = buffer[i * 4 + 1];
            pt.z = buffer[i * 4 + 2];
            // pt.intensity = buffer[i * 4 + 3];
        }

        return cloud;
    }

    void load_pose_kitti(string path, vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> &transpose,
                         string sequence, const vector<double> &Time_stamp) {
        ifstream pose(path);
        cout << "path: " << path << endl;
        string line;
        while (getline(pose, line)) {
            if (line.size() > 0) {
                stringstream ss(line);
                Eigen::Matrix4d Pose_(Eigen::Matrix4d::Identity());
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 4; ++j) {
                        ss >> Pose_(i, j);
                    }
                transpose.push_back(Pose_);
            }
        }
        pose.close();
        /*
        ofstream tum_pose(sequence+".txt");
        for (int i=0;i<transpose.size();i++){
            Eigen::Quaterniond quaterniond;
            quaterniond=transpose[i].block<3,3>(0,0);
            tum_pose<<Time_stamp[i]<<" "<<transpose[i](0,3)<<" "<<transpose[i](1,3)<<" "<<transpose[i](2,3)<<" "
                    <<quaterniond.x()<<" "<<quaterniond.y()<<" "<<quaterniond.z()<<" "<<quaterniond.w()<<endl;
        }
        tum_pose.close();*/

    }

private:
    int num_frames;
    std::string dataset_path;
};

int main(int argc, char **argv) {

    cv::FileStorage fs("../include/settings.txt",cv::FileStorage::READ);
    string sequence,res0,res1,use_prob_kernal,use_sampling,use_scan2model,date_path,kitti_path,gt_pose_path,
    save_path,save_error_path,save_pose_error_path,save_pose_path;

    fs["kitti_sequence"]>>sequence;
    fs["resolution_source"]>>res0; // the voxelization resolution of the source point cloud
    fs["resolution_target"]>>res1; // the voxelization resolution of the target point cloud
    fs["use_prob_kernal"]>>use_prob_kernal; //1:use the second-step filter; 0: do not use
    fs["use_sampling"]>>use_sampling; //1: use the first-step filter; 0: do not use
    fs["use_scan2model"]>>use_scan2model; //m: use scan-to-model matching for LiDAR odometry,s:use scan-to-scan matching
    fs["kitti_path"]>>date_path;// path for kitti dataset
    fs["save_path"]>>save_path;//path for poses output

    kitti_path=date_path+"sequences/" + sequence + "/velodyne";
    save_error_path=save_path+"errors_" + sequence + "_" + res0 + "_" + res1 + "_" +
            use_prob_kernal + "_" + use_sampling + "_" + use_scan2model + "_" + ".txt";
    save_pose_error_path=save_path+"pose_errors_" + sequence + "_" + res0 + "_" + res1 + "_" +
            use_prob_kernal + "_" + use_sampling + "_" + use_scan2model + "_" + ".txt";
            save_pose_path=save_path+"traj" + sequence + "_" + res0 + "_" + res1 + "_"
                   + use_prob_kernal + "_" + use_sampling + "_" + use_scan2model + "_" + ".txt";

    vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> transpose;
    std::ofstream ofs_error(save_error_path.c_str()),pose_error(save_pose_error_path.c_str());

    vector<double> Time_stamp;

    Eigen::Matrix4d tr;
    tr << 4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02,
            -7.210626507497e-03, 8.081198471645e-03, -9.999413164504e-01, -5.403984729748e-02,
            9.999738645903e-01, 4.859485810390e-04, -7.206933692422e-03, -2.921968648686e-01,
            0, 0, 0, 1;

    KittiLoader kitti(kitti_path);
    kitti.load_pose_kitti(gt_pose_path, transpose, sequence, Time_stamp);

    double downsample_resolution = atof(res0.c_str()), downsample_resolution0 = atof(res1.c_str());
    pcl::ApproximateVoxelGrid<pcl::PointXYZ> voxelgrid, voxelgrid0, voxelgrid1;
    voxelgrid.setLeafSize(downsample_resolution, downsample_resolution, downsample_resolution);
    voxelgrid0.setLeafSize(downsample_resolution0, downsample_resolution0, downsample_resolution0);

    // registration method
    // you should fine-tune hyper-parameters (e.g., voxel resolution, max correspondence distance) for the best result
    fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ> gicp;
    gicp.use_svd_ = true;
    gicp.use_kernal_ = (use_prob_kernal == "1");
    gicp.use_fec_ = false;
    gicp.use_linear_ = false;
    gicp.use_sampling_ = (use_sampling == "1");
    gicp.ave_filter_no_ = 0;
    gicp.ave_point_ = 0;
    gicp.rejected_points_no_ = 0;

    gicp.setMaxCorrespondenceDistance(1.0);
    gicp.setCorrespondenceRandomness(20);
    gicp.setRegularizationMethod(fast_gicp::RegularizationMethod::PLANE);
    //gicp.setRegularizationMethod(fast_gicp::RegularizationMethod::NONE);
    //set initial frame as target



    voxelgrid.setInputCloud(kitti.cloud(0));
    voxelgrid0.setInputCloud(kitti.cloud(0));
    pcl::PointCloud<pcl::PointXYZ>::Ptr target(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr target0(new pcl::PointCloud<pcl::PointXYZ>);
    voxelgrid0.filter(*target0);
    gicp.setTargetsortedCloud(target0);
    gicp.setInputTarget(target0);
    // sensor pose sequence
    std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> poses(kitti.size()), lidar_poses(
            kitti.size());
    poses[0].setIdentity();
    lidar_poses[0].setIdentity();
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


    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> local_frames;
    pcl::PointCloud<pcl::PointXYZ>::Ptr local_map_(new pcl::PointCloud<pcl::PointXYZ>);
    local_frames.clear();
    bool local_optimization_ = false;
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_localmap(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    vector<Eigen::Matrix4d> local_poses_;
    local_poses_.clear();
    Eigen::Matrix4d M0 = Eigen::Matrix4d::Identity(), M_gt;
    int keyframe_no = 6;
    bool scan_to_model_ = (use_scan2model.find("m") == 0);
    for (int i = 1; i < kitti.size(); i++) {
        gicp.index_ = i;
        voxelgrid.setInputCloud(kitti.cloud(i));


        pcl::PointCloud<pcl::PointXYZ>::Ptr source(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr source0(new pcl::PointCloud<pcl::PointXYZ>);
        voxelgrid.filter(*source);
        voxelgrid0.setInputCloud(kitti.cloud(i));
        voxelgrid0.filter(*source0);

        if (scan_to_model_) {
            if (i > 50) {
                if (local_frames.size() < keyframe_no) {
                    local_frames.push_back(source0);
                } else {
                    local_frames.erase(local_frames.begin());
                    local_frames.push_back(source0);
                }
            }
            if (local_frames.size() == keyframe_no) {
                local_map_->clear();
                M0.setIdentity();
                int count = 1;
                for (int k = local_frames.size() - 3; k > 0; --k) {
                    M0 = M0 * local_poses_.at(local_poses_.size() - count).inverse();
                    pcl::transformPointCloud(*local_frames.at(k), *transformed_cloud, M0);
                    *local_map_ += *transformed_cloud;
                    ++count;
                }
                *local_map_ += *local_frames.at(local_frames.size() - 2);
                target_localmap->clear();
                //cout<<"local_map_ size before filter: "<<local_map_->size()<<endl;
                voxelgrid0.setInputCloud(local_map_);
                voxelgrid0.filter(*target_localmap);
                //cout<<"local_map_ size after filter: "<<target_localmap->size()<<endl;
                local_optimization_ = true;
                gicp.setInputTarget(target_localmap);
                gicp.setTargetsortedCloud(target_localmap);
                bool visualization = false;
                if (visualization) {
                    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
                            new pcl::visualization::PCLVisualizer("3D Viewer"));

                    //pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> fildColor(target_localmap, "z");

                    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(target_localmap, 0,
                                                                                                 255, 0);
                    viewer->addPointCloud<pcl::PointXYZ>(target_localmap, single_color, "sample cloud");
                    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1,
                                                             "sample cloud"); // 设置点云大小
                    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_color(source0, 255, 0, 0);

                    //pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> fildColor0(target_localmap, "x");
                    viewer->addPointCloud<pcl::PointXYZ>(source0, source_color, "source cloud");
                    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1,
                                                             "source cloud"); // 设置点云大小

                    while (!viewer->wasStopped()) {
                        viewer->spinOnce(100);
                        boost::this_thread::sleep(boost::posix_time::microseconds(100000));
                    }
                }
            }
        }


        gicp.setSourceSortedCloud(source);
        gicp.setInputSource(source0);
        pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);

        if (i == 1) {
            M = Eigen::Matrix4d::Identity();
            gicp.setInit_Guess(M);
        } else {
            gicp.setInit_Guess(M);
        }
        M_gt = tr.inverse() * transpose.at(i - 1).inverse() * transpose.at(i) * tr;
        double start = omp_get_wtime();
        gicp.align(*aligned);
        double end = omp_get_wtime();
        Errors_ = gicp.get_matchingerror();
        ofs_error << Errors_[0] << " " << Errors_[1] << endl;
        if (scan_to_model_) {
            if (!local_optimization_)
                gicp.swapSourceAndTarget();
        } else
            gicp.swapSourceAndTarget();


        if (gicp.is_converged_)
            M = gicp.getFinalTransformation().cast<double>();
        // accumulate pose



        bool visualization0 = false;
        if (visualization0 and i > 100) {
            boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(
                    new pcl::visualization::PCLVisualizer("3D Viewer"));

            //pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> fildColor(target_localmap, "z");

            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(target_localmap, 0, 255, 0);
            viewer->addPointCloud<pcl::PointXYZ>(target_localmap, single_color, "sample cloud");
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1,
                                                     "sample cloud"); // 设置点云大小
            pcl::transformPointCloud(*source0, *transformed_cloud, M);
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_color(transformed_cloud, 255, 0, 0);

            //pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> fildColor0(target_localmap, "x");
            viewer->addPointCloud<pcl::PointXYZ>(transformed_cloud, source_color, "source cloud");
            viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1,
                                                     "source cloud"); // 设置点云大小

            while (!viewer->wasStopped()) {
                viewer->spinOnce(100);
                boost::this_thread::sleep(boost::posix_time::microseconds(100000));
            }
        }


        poses[i] = poses[i - 1] * tr * M * tr.inverse();
        lidar_poses[i] = lidar_poses[i - 1] * M;
        frames_[i] = *source0;
        local_poses_.push_back(M);
        Eigen::Matrix4d Error_T = M.inverse() * M_gt;
        double d = sqrt(Error_T(0, 3) * Error_T(0, 3) + Error_T(1, 3) * Error_T(1, 3) + Error_T(2, 3) * Error_T(2, 3));
        double r = (0.5 * (Error_T(0, 0) + Error_T(1, 1) + Error_T(2, 2) - 1));
        r = acos(max(min(r, 1.0), -1.0));

        pose_error << d << " " << r << endl;
        // FPS display
        stamps.push_back(std::chrono::high_resolution_clock::now());

        // visualization
        trajectory->push_back(pcl::PointXYZ(poses[i](0, 3), poses[i](1, 3), poses[i](2, 3)));
        vis.updatePointCloud<pcl::PointXYZ>(trajectory,
                                            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(trajectory,
                                                                                                            255.0, 0.0,
                                                                                                            0.0),
                                            "trajectory");
        vis.spinOnce();
    }
    std::cout << stamps.size() /
                 (std::chrono::duration_cast<std::chrono::nanoseconds>(stamps.back() - stamps.front()).count() / 1e9)
              << "fps" << std::endl;
    pose_error << stamps.size() /
                  (std::chrono::duration_cast<std::chrono::nanoseconds>(stamps.back() - stamps.front()).count() / 1e9)
               << endl;
    pose_error << (double) gicp.ave_filter_no_ / (kitti.size() - 1) << " "
               << (double) gicp.rejected_points_no_ / (kitti.size() - 1) << endl;
    cout << "ave filter no " << (double) gicp.ave_filter_no_ / (kitti.size() - 50) << " "
         << (double) gicp.ave_point_ / (kitti.size() - 50) << endl;
    ofs_error.close();
    pose_error.close();
    // save the estimated poses
    std::ofstream ofs(save_pose_path.c_str());
    for (const auto &pose: poses) {
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

    cout << "generate map" << endl;
    pcl::PointCloud<pcl::PointXYZ>::Ptr mapPointCloud_(new pcl::PointCloud<pcl::PointXYZ>())
    , Map_(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_PointCloud_(new pcl::PointCloud<pcl::PointXYZ>());
    int start_index_ = 0;
    Eigen::Matrix4d T_;
    pcl::ApproximateVoxelGrid<pcl::PointXYZ> voxelgrid_map_;
    voxelgrid_map_.setLeafSize(2, 2, 2);
    cout << "pose and frame no: " << poses.size() << " " << frames_.size() << endl;
    for (int i = start_index_; i < frames_.size(); i = i + 5) {
        //cout<<"i "<<i<<endl;
        Eigen::Matrix4d M0 = lidar_poses[start_index_].matrix();
        Eigen::Matrix4d M1 = lidar_poses[i].matrix();
        T_ = M1 * M0.inverse();
        //IP_.resetParameters();
        //IP_.cloudHandler(frames_[i]);
        pcl::transformPointCloud(frames_[i], *transformed_PointCloud_, T_);
        *mapPointCloud_ += *transformed_PointCloud_;
    }
    voxelgrid_map_.setInputCloud(mapPointCloud_);
    voxelgrid_map_.filter(*Map_);
    string map_save_path =
            "/home/wjk/Fast_gicp_odometry/fast_gicp-master/tmp/Map" + sequence + "_" + use_sampling + ".pcd";
    //pcl::io::savePCDFileASCII(map_save_path, *Map_);
    cout << "map size: " << mapPointCloud_->size() << " " << Map_->size() << endl;

    bool visualization0 = false;
    if (visualization0) {
        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));

        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> fildColor(Map_, "z");

        viewer->addPointCloud<pcl::PointXYZ>(Map_, fildColor, "sample cloud");
        viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 0.3,
                                                 "sample cloud"); // 设置点云大小


        while (!viewer->wasStopped()) {
            viewer->spinOnce(100);
            boost::this_thread::sleep(boost::posix_time::microseconds(100000));
        }
    }


    return 0;
}