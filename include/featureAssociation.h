//
// Created by robot410 on 2020/11/3.
//

#ifndef LOAM_ODOMETRY_FEATUREASSOCIATION_H
#define LOAM_ODOMETRY_FEATUREASSOCIATION_H

#include "utility.h"

class FeatureAssociation{

public:

    pcl::PointCloud<PointType>::Ptr segmentedCloud;
    pcl::PointCloud<PointType>::Ptr outlierCloud;

    pcl::PointCloud<PointType>::Ptr cornerPointsSharp;
    pcl::PointCloud<PointType>::Ptr cornerPointsLessSharp;
    pcl::PointCloud<PointType>::Ptr surfPointsFlat;
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlat;

    pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan;
    pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScanDS;

    pcl::VoxelGrid<PointType> downSizeFilter;

    double timeScanCur;
    double timeNewSegmentedCloud;
    double timeNewSegmentedCloudInfo;
    double timeNewOutlierCloud;

    bool newSegmentedCloud;
    bool newSegmentedCloudInfo;
    bool newOutlierCloud;

    cloud_info segInfo;

    int systemInitCount;
    bool systemInited;

    std::vector<smoothness_t> cloudSmoothness;
    float cloudCurvature[N_SCAN*Horizon_SCAN];
    int cloudNeighborPicked[N_SCAN*Horizon_SCAN];
    int cloudLabel[N_SCAN*Horizon_SCAN];

    int imuPointerFront;
    int imuPointerLast;
    int imuPointerLastIteration;

    float imuRollStart, imuPitchStart, imuYawStart;
    float cosImuRollStart, cosImuPitchStart, cosImuYawStart, sinImuRollStart, sinImuPitchStart, sinImuYawStart;
    float imuRollCur, imuPitchCur, imuYawCur;

    float imuVeloXStart, imuVeloYStart, imuVeloZStart;
    float imuShiftXStart, imuShiftYStart, imuShiftZStart;

    float imuVeloXCur, imuVeloYCur, imuVeloZCur;
    float imuShiftXCur, imuShiftYCur, imuShiftZCur;

    float imuShiftFromStartXCur, imuShiftFromStartYCur, imuShiftFromStartZCur;
    float imuVeloFromStartXCur, imuVeloFromStartYCur, imuVeloFromStartZCur;

    float imuAngularRotationXCur, imuAngularRotationYCur, imuAngularRotationZCur;
    float imuAngularRotationXLast, imuAngularRotationYLast, imuAngularRotationZLast;
    float imuAngularFromStartX, imuAngularFromStartY, imuAngularFromStartZ;

    double imuTime[imuQueLength];
    float imuRoll[imuQueLength];
    float imuPitch[imuQueLength];
    float imuYaw[imuQueLength];

    float imuAccX[imuQueLength];
    float imuAccY[imuQueLength];
    float imuAccZ[imuQueLength];

    float imuVeloX[imuQueLength];
    float imuVeloY[imuQueLength];
    float imuVeloZ[imuQueLength];

    float imuShiftX[imuQueLength];
    float imuShiftY[imuQueLength];
    float imuShiftZ[imuQueLength];

    float imuAngularVeloX[imuQueLength];
    float imuAngularVeloY[imuQueLength];
    float imuAngularVeloZ[imuQueLength];

    float imuAngularRotationX[imuQueLength];
    float imuAngularRotationY[imuQueLength];
    float imuAngularRotationZ[imuQueLength];

    int skipFrameNum;
    bool systemInitedLM;

    int laserCloudCornerLastNum;
    int laserCloudSurfLastNum;

    int pointSelCornerInd[N_SCAN*Horizon_SCAN];
    float pointSearchCornerInd1[N_SCAN*Horizon_SCAN];
    float pointSearchCornerInd2[N_SCAN*Horizon_SCAN];

    int pointSelSurfInd[N_SCAN*Horizon_SCAN];
    float pointSearchSurfInd1[N_SCAN*Horizon_SCAN];
    float pointSearchSurfInd2[N_SCAN*Horizon_SCAN];
    float pointSearchSurfInd3[N_SCAN*Horizon_SCAN];

    float transformCur[6];
    float transformSum[6];

    float imuRollLast, imuPitchLast, imuYawLast;
    float imuShiftFromStartX, imuShiftFromStartY, imuShiftFromStartZ;
    float imuVeloFromStartX, imuVeloFromStartY, imuVeloFromStartZ;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerLast;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfLast;
    pcl::PointCloud<PointType>::Ptr laserCloudOri;
    pcl::PointCloud<PointType>::Ptr coeffSel;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerLast;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfLast;

    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;

    vector<double> cornor_point_ld2s,surf_point_pd2s;

    PointType pointOri, pointSel, tripod1, tripod2, tripod3, pointProj, coeff;

    bool isDegenerate;
    cv::Mat matP;

    int frameCount;

    double average_s=0,num_s=0;

    FeatureAssociation();

    void initializationValue();

    void updateImuRollPitchYawStartSinCos();

    void ShiftToStartIMU(float pointTime);

    void VeloToStartIMU();

    void TransformToStartIMU(PointType *p);

    void AccumulateIMUShiftAndRotation();

    void laserCloudHandler();

    void laserCloudInfoHandler(const cloud_info *msgIn);

    void adjustDistortion();

    void calculateSmoothness();

    void markOccludedPoints();

    void extractFeatures();

    void publishCloud();

    void TransformToStart(PointType const * const pi, PointType * const po);

    void TransformToEnd(PointType const * const pi, PointType * const po);

    void PluginIMURotation(float bcx, float bcy, float bcz, float blx, float bly, float blz,
                           float alx, float aly, float alz, float &acx, float &acy, float &acz);

    void AccumulateRotation(float cx, float cy, float cz, float lx, float ly, float lz,
                            float &ox, float &oy, float &oz);

    double rad2deg(double radians);

    double deg2rad(double degrees);

    void findCorrespondingCornerFeatures(int iterCount);

    void findCorrespondingSurfFeatures(int iterCount);

    bool calculateTransformationSurf(int iterCount);

    bool calculateTransformationCorner(int iterCount);

    bool calculateTransformation(int iterCount);

    void checkSystemInitialization();

    void updateInitialGuess();

    void updateTransformation();

    //void integrateTransformation();

    void publishOdometry();

    void adjustOutlierCloud();

    void publishCloudsLast();

    void runFeatureAssociation();
    
   // void AccumulateRotation(float cx, float cy, float cz, float lx, float ly, float lz,
                                    //        float &ox, float &oy, float &oz);
    void integrateTransformation();

};
#endif //LOAM_ODOMETRY_FEATUREASSOCIATION_H
