#ifndef _UTILITY_LIDAR_ODOMETRY_H_
#define _UTILITY_LIDAR_ODOMETRY_H_

#pragma once
#include <opencv/cv.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/range_image/range_image.h>
#include <pcl/filters/filter.h>	
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/registration/icp.h>

#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <deque>
#include <iostream>
#include <fstream>
#include <ctime>
#include <cfloat>
#include <iterator>
#include <sstream>
#include <string>
#include <limits>
#include <iomanip>
#include <array>
#include <thread>
#include <mutex>

using namespace std;

typedef pcl::PointXYZI  PointType;
typedef pcl::PointXYZINormal PointType_normal_;

const string pointCloudTopic = "/kitti/velo/pointcloud";
const string imuTopic = "/imu/data";


// Save pcd
const string fileDirectory = "/home/robot410/data/kitti/dataset/";

// VLP-16
const int N_SCAN = 16;
const int Horizon_SCAN = 1800;
const float ang_res_x = 0.2;
const float ang_res_y = 2.0;
const float ang_bottom = 15.0+0.1;
const int groundScanInd = 7;

// HDL-32E
//   const int N_SCAN = 32;
//   const int Horizon_SCAN = 1800;
//   const float ang_res_x = 360.0/float(Horizon_SCAN);
//   const float ang_res_y = 41.33/float(N_SCAN-1);
//   const float ang_bottom = 30.67;
//   const int groundScanInd = 20;

// Ouster users may need to uncomment line 159 in imageProjection.cpp
// Usage of Ouster imu data is not fully supported yet, please just publish point cloud data
// Ouster OS1-16
//   const int N_SCAN = 16;
//   const int Horizon_SCAN = 1024;
//   const float ang_res_x = 360.0/float(Horizon_SCAN);
//   const float ang_res_y = 33.2/float(N_SCAN-1);
//   const float ang_bottom = 16.6+0.1;
//   const int groundScanInd = 7;

// Ouster OS1-64
//  const int N_SCAN = 64;
//  const int Horizon_SCAN = 1024;
//  const float ang_res_x = 360.0/float(Horizon_SCAN);
//  const float ang_res_y = 33.2/float(N_SCAN-1);
//  const float ang_bottom = 16.6+0.1;
//  const int groundScanInd = 15;

//Vel 64
//  const int N_SCAN = 64;
//  const int Horizon_SCAN = 1800;
 // const float ang_res_x = 0.2;
 // const float ang_res_y = 0.427;
 // const float ang_bottom = 24.9;
 // const int groundScanInd = 50;

const bool loopClosureEnableFlag = true;
const double mappingProcessInterval = 0.3;

const float scanPeriod = 0.1;
const int systemDelay = 0;
const int imuQueLength = 200;

const float sensorMountAngle = 0.0;
const float segmentTheta = 60.0/180.0*M_PI; // decrese this value may improve accuracy
const int segmentValidPointNum = 5;
const int segmentValidLineNum = 3;
const float segmentAlphaX = ang_res_x / 180.0 * M_PI;
const float segmentAlphaY = ang_res_y / 180.0 * M_PI;


  const int edgeFeatureNum = 2;
  const int surfFeatureNum = 4;
  const int sectionsTotal = 6;
  const float edgeThreshold = 0.1;
  const float surfThreshold = 0.1;
  const float nearestFeatureSearchSqDist = 25;



// Mapping Params
  const float surroundingKeyframeSearchRadius = 50.0; // key frame that is within n meters from current pose will be considerd for scan-to-map optimization (when loop closure disabled)
  const int   surroundingKeyframeSearchNum = 50; // submap size (when loop closure enabled)
// history key frames (history submap for loop closure)
  const float historyKeyframeSearchRadius = 7.0; // key frame that is within n meters from current pose will be considerd for loop closure
  const int   historyKeyframeSearchNum = 25; // 2n+1 number of hostory key frames will be fused into a submap for loop closure
  const float historyKeyframeFitnessScore = 0.3; // the smaller the better alignment

  const float globalMapVisualizationSearchRadius = 500.0; // key frames with in n meters will be visualized


struct smoothness_t{ 
    float value;
    size_t ind;
};

struct by_value{ 
    bool operator()(smoothness_t const &left, smoothness_t const &right) { 
        return left.value < right.value;
    }
};

/*
    * A point cloud type that has 6D pose info ([x,y,z,roll,pitch,yaw] intensity is time stamp)
    */
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time)
)

typedef PointXYZIRPYT  PointTypePose;

struct cloud_info{

    int* startRingIndex;
    int* endRingIndex;

    float startOrientation;
    float endOrientation;
    float orientationDiff;

    bool*    segmentedCloudGroundFlag ;
    int*  segmentedCloudColInd;
    float* segmentedCloudRange;
};


#endif
