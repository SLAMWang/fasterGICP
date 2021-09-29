//
// Created by robot410 on 2020/11/3.
//

#ifndef LOAM_ODOMETRY_IMAGEPROJECTION_H
#define LOAM_ODOMETRY_IMAGEPROJECTION_H

#include "utility.h"

class ImageProjection{
public:

    pcl::PointCloud<PointType>::Ptr laserCloudIn;

    pcl::PointCloud<PointType>::Ptr fullCloud; // projected velodyne raw cloud, but saved in the form of 1-D matrix
    pcl::PointCloud<PointType>::Ptr fullInfoCloud; // same as fullCloud, but with intensity - range

    pcl::PointCloud<PointType>::Ptr groundCloud;
    pcl::PointCloud<PointType>::Ptr segmentedCloud;
    pcl::PointCloud<PointType>::Ptr segmentedCloudPure;
    pcl::PointCloud<PointType>::Ptr outlierCloud;

    PointType nanPoint; // fill in fullCloud at each iteration

    cv::Mat rangeMat; // range matrix for range image
    cv::Mat labelMat; // label matrix for segmentaiton marking
    cv::Mat groundMat; // ground matrix for ground cloud marking
    int labelCount;

    float startOrientation;
    float endOrientation;

    cloud_info segMsg; // info of segmented cloud

    std::vector<std::pair<int8_t, int8_t> > neighborIterator; // neighbor iterator for segmentaiton process

    uint16_t *allPushedIndX; // array for tracking points of a segmented object
    uint16_t *allPushedIndY;

    uint16_t *queueIndX; // array for breadth-first search process of segmentation
    uint16_t *queueIndY;

    ImageProjection();

    void allocateMemory();

    void resetParameters();

    void copyPointCloud(string cloud_path);
    void cloudHandler(string cloud_path);
    void cloudHandler(pcl::PointCloud<pcl::PointXYZ> input_cloud);
    void findStartEndAngle();
    void projectPointCloud();

    void groundRemoval();

    void cloudSegmentation();

    void labelComponents(int row, int col);

    void publishCloud();
};

#endif //LOAM_ODOMETRY_IMAGEPROJECTION_H
