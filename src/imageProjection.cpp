#include "imageProjection.h"
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/ply_io.h>

ImageProjection::ImageProjection(){

    nanPoint.x = std::numeric_limits<float>::quiet_NaN();
    nanPoint.y = std::numeric_limits<float>::quiet_NaN();
    nanPoint.z = std::numeric_limits<float>::quiet_NaN();
    nanPoint.intensity = -1;

    allocateMemory();
    resetParameters();
}

void ImageProjection::allocateMemory(){

    laserCloudIn.reset(new pcl::PointCloud<PointType>());

    fullCloud.reset(new pcl::PointCloud<PointType>());
    fullInfoCloud.reset(new pcl::PointCloud<PointType>());

    groundCloud.reset(new pcl::PointCloud<PointType>());
    segmentedCloud.reset(new pcl::PointCloud<PointType>());
    segmentedCloudPure.reset(new pcl::PointCloud<PointType>());
    outlierCloud.reset(new pcl::PointCloud<PointType>());

    fullCloud->points.resize(N_SCAN*Horizon_SCAN);
    fullInfoCloud->points.resize(N_SCAN*Horizon_SCAN);

    segMsg.startRingIndex= new int[N_SCAN];
    segMsg.endRingIndex= new int [N_SCAN];

    segMsg.segmentedCloudGroundFlag= new bool [N_SCAN*Horizon_SCAN];
    segMsg.segmentedCloudColInd= new int [N_SCAN*Horizon_SCAN];
    segMsg.segmentedCloudRange= new float [N_SCAN*Horizon_SCAN];

    for (int i=0;i<N_SCAN*Horizon_SCAN;i++){
        segMsg.segmentedCloudGroundFlag[i]=false;
        segMsg.segmentedCloudColInd[i]=0;
        segMsg.segmentedCloudRange[i]=0;
    }

    std::pair<int8_t, int8_t> neighbor;
    neighbor.first = -1; neighbor.second =  0; neighborIterator.push_back(neighbor);
    neighbor.first =  0; neighbor.second =  1; neighborIterator.push_back(neighbor);
    neighbor.first =  0; neighbor.second = -1; neighborIterator.push_back(neighbor);
    neighbor.first =  1; neighbor.second =  0; neighborIterator.push_back(neighbor);

    allPushedIndX = new uint16_t[N_SCAN*Horizon_SCAN];
    allPushedIndY = new uint16_t[N_SCAN*Horizon_SCAN];

    queueIndX = new uint16_t[N_SCAN*Horizon_SCAN];
    queueIndY = new uint16_t[N_SCAN*Horizon_SCAN];
}

void ImageProjection::resetParameters(){
    laserCloudIn->clear();
    groundCloud->clear();
    segmentedCloud->clear();
    segmentedCloudPure->clear();
    outlierCloud->clear();

    rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));
    groundMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_8S, cv::Scalar::all(0));
    labelMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32S, cv::Scalar::all(0));
    labelCount = 1;

    std::fill(fullCloud->points.begin(), fullCloud->points.end(), nanPoint);
    std::fill(fullInfoCloud->points.begin(), fullInfoCloud->points.end(), nanPoint);
}

void ImageProjection::copyPointCloud(string cloud_path){

    int32_t Num = 1000000;
    auto *data = new float[Num];

    // pointers
    float *px = data+0;
    float *py = data+1;
    float *pz = data+2;
    float *pr = data+3;

    FILE *lidar;
    lidar = fopen (cloud_path.c_str(),"rb");
    Num = fread(data,sizeof(float),Num,lidar)/4;
    for (int32_t i=0; i<Num; i++) {
        PointType p;
        p.x=*px;
        p.y=*py;
        p.z=*pz;
//        if (abs(p.x+p.y+p.z)>0.1)
            laserCloudIn->points.push_back(p);
        px+=4; py+=4; pz+=4; pr+=4;
    }
    fclose(lidar);
    free(data);

    // Remove Nan points
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*laserCloudIn, *laserCloudIn, indices);
}

void ImageProjection::cloudHandler(pcl::PointCloud<pcl::PointXYZ> input_cloud){
    for (int i=0;i<input_cloud.size();i++)
    {
      PointType p;
      p.x = input_cloud.points[i].x;
      p.y = input_cloud.points[i].y;
      p.z = input_cloud.points[i].z;
      p.intensity = 0;
      laserCloudIn->push_back(p);
    }

    findStartEndAngle();
    projectPointCloud();
    groundRemoval();
    cloudSegmentation();
}

void ImageProjection::cloudHandler(string cloud_path){

    // 1. Convert ros message to pcl point cloud
    copyPointCloud(cloud_path);
    // 2. Start and end angle of a scan
    findStartEndAngle();
//     3. Range image projection
    projectPointCloud();
    // 4. Mark ground points
    groundRemoval();
    // 5. Point cloud segmentation
    cloudSegmentation();
//        // 6. Publish all clouds
//        publishCloud();
    // 7. Reset parameters for next iteration
//    resetParameters();
}

void ImageProjection::findStartEndAngle(){
    // start and end orientation of this cloud
    //segMsg.startOrientation = -atan2(laserCloudIn->points[0].y, laserCloudIn->points[0].x);
    //segMsg.endOrientation   = -atan2(laserCloudIn->points[laserCloudIn->points.size() - 1].y,
    //                                 laserCloudIn->points[laserCloudIn->points.size() - 1].x) + 2 * M_PI;
  segMsg.startOrientation = 0;
    segMsg.endOrientation   = 2 * M_PI;
    if (segMsg.endOrientation - segMsg.startOrientation > 3 * M_PI) {
        segMsg.endOrientation -= 2 * M_PI;
    } else if (segMsg.endOrientation - segMsg.startOrientation < M_PI)
        segMsg.endOrientation += 2 * M_PI;
    segMsg.orientationDiff = segMsg.endOrientation - segMsg.startOrientation;
}

void ImageProjection::projectPointCloud(){
    // range image projection
    float verticalAngle, horizonAngle, range;
    size_t rowIdn, columnIdn, index, cloudSize;
    PointType thisPoint;

    cloudSize = laserCloudIn->points.size();

    for (size_t i = 0; i < cloudSize; ++i){

        thisPoint.x = laserCloudIn->points[i].x;
        thisPoint.y = laserCloudIn->points[i].y;
        thisPoint.z = laserCloudIn->points[i].z;
        // find the row and column index in the iamge for this point
        verticalAngle = atan2(thisPoint.z, sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y)) * 180 / M_PI;
        rowIdn = (verticalAngle + ang_bottom) / ang_res_y;
        if (rowIdn < 0 || rowIdn >= N_SCAN)
            continue;

        horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;

        columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
        if (columnIdn >= Horizon_SCAN)
            columnIdn -= Horizon_SCAN;

        if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
            continue;

        range = sqrt(thisPoint.x * thisPoint.x + thisPoint.y * thisPoint.y + thisPoint.z * thisPoint.z);
        if (range < 0.1)
            continue;

        rangeMat.at<float>(rowIdn, columnIdn) = range;

        thisPoint.intensity = (float)rowIdn + (float)columnIdn / 10000.0;

        index = columnIdn  + rowIdn * Horizon_SCAN;
        fullCloud->points[index] = thisPoint;
        fullInfoCloud->points[index] = thisPoint;
        fullInfoCloud->points[index].intensity = range; // the corresponding range of a point is saved as "intensity"
    }
}

void ImageProjection::groundRemoval(){
    size_t lowerInd, upperInd;
    float diffX, diffY, diffZ, angle;
    // groundMat
    // -1, no valid info to check if ground of not
    //  0, initial value, after validation, means not ground
    //  1, ground
    for (size_t j = 0; j < Horizon_SCAN; ++j){
        for (size_t i = 0; i < groundScanInd; ++i){

            lowerInd = j + ( i )*Horizon_SCAN;
            upperInd = j + (i+1)*Horizon_SCAN;

            if (fullCloud->points[lowerInd].intensity == -1 ||
                fullCloud->points[upperInd].intensity == -1){
                // no info to check, invalid points
                groundMat.at<int8_t>(i,j) = -1;
                continue;
            }

            diffX = fullCloud->points[upperInd].x - fullCloud->points[lowerInd].x;
            diffY = fullCloud->points[upperInd].y - fullCloud->points[lowerInd].y;
            diffZ = fullCloud->points[upperInd].z - fullCloud->points[lowerInd].z;

            angle = atan2(diffZ, sqrt(diffX*diffX + diffY*diffY) ) * 180 / M_PI;

            if (abs(angle - sensorMountAngle) <= 10){
                groundMat.at<int8_t>(i,j) = 1;
                groundMat.at<int8_t>(i+1,j) = 1;
            }
        }
    }
    // extract ground cloud (groundMat == 1)
    // mark entry that doesn't need to label (ground and invalid point) for segmentation
    // note that ground remove is from 0~N_SCAN-1, need rangeMat for mark label matrix for the 16th scan
    for (size_t i = 0; i < N_SCAN; ++i){
        for (size_t j = 0; j < Horizon_SCAN; ++j){
            if (groundMat.at<int8_t>(i,j) == 1 || rangeMat.at<float>(i,j) == FLT_MAX){
                labelMat.at<int>(i,j) = -1;
            }
        }
    }
}

void ImageProjection::cloudSegmentation(){
    // segmentation process
    for (size_t i = 0; i < N_SCAN; ++i)
        for (size_t j = 0; j < Horizon_SCAN; ++j)
            if (labelMat.at<int>(i,j) == 0)
                labelComponents(i, j);

    int sizeOfSegCloud = 0;
    // extract segmented cloud for lidar odometry
    for (size_t i = 0; i < N_SCAN; ++i) {

        segMsg.startRingIndex[i] = sizeOfSegCloud-1 + 5;

        for (size_t j = 0; j < Horizon_SCAN; ++j) {
            if (labelMat.at<int>(i,j) > 0 || groundMat.at<int8_t>(i,j) == 1){
                // outliers that will not be used for optimization (always continue)
                if (labelMat.at<int>(i,j) == 999999){
                    if (i > groundScanInd && j % 5 == 0){
                        outlierCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                        continue;
                    }else{
                        continue;
                    }
                }
                // majority of ground points are skipped
                if (groundMat.at<int8_t>(i,j) == 1){
                    if (j%5!=0 && j>5 && j<Horizon_SCAN-5)
                        continue;
                }
                // mark ground points so they will not be considered as edge features later
                segMsg.segmentedCloudGroundFlag[sizeOfSegCloud] = (groundMat.at<int8_t>(i,j) == 1);
                // mark the points' column index for marking occlusion later
                segMsg.segmentedCloudColInd[sizeOfSegCloud] = j;
                // save range info
                segMsg.segmentedCloudRange[sizeOfSegCloud]  = rangeMat.at<float>(i,j);
                // save seg cloud
                segmentedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                // size of seg cloud
                ++sizeOfSegCloud;
            }
        }

        segMsg.endRingIndex[i] = sizeOfSegCloud-1 - 5;
    }

}

void ImageProjection::labelComponents(int row, int col){
    // use std::queue std::vector std::deque will slow the program down greatly
    float d1, d2, alpha, angle;
    int fromIndX, fromIndY, thisIndX, thisIndY;
    bool lineCountFlag[N_SCAN] = {false};

    queueIndX[0] = row;
    queueIndY[0] = col;
    int queueSize = 1;
    int queueStartInd = 0;
    int queueEndInd = 1;

    allPushedIndX[0] = row;
    allPushedIndY[0] = col;
    int allPushedIndSize = 1;

    while(queueSize > 0){
        // Pop point
        fromIndX = queueIndX[queueStartInd];
        fromIndY = queueIndY[queueStartInd];
        --queueSize;
        ++queueStartInd;
        // Mark popped point
        labelMat.at<int>(fromIndX, fromIndY) = labelCount;
        // Loop through all the neighboring grids of popped grid
        for (auto iter = neighborIterator.begin(); iter != neighborIterator.end(); ++iter){
            // new index
            thisIndX = fromIndX + (*iter).first;
            thisIndY = fromIndY + (*iter).second;
            // index should be within the boundary
            if (thisIndX < 0 || thisIndX >= N_SCAN)
                continue;
            // at range image margin (left or right side)
            if (thisIndY < 0)
                thisIndY = Horizon_SCAN - 1;
            if (thisIndY >= Horizon_SCAN)
                thisIndY = 0;
            // prevent infinite loop (caused by put already examined point back)
            if (labelMat.at<int>(thisIndX, thisIndY) != 0)
                continue;

            d1 = std::max(rangeMat.at<float>(fromIndX, fromIndY),
                          rangeMat.at<float>(thisIndX, thisIndY));
            d2 = std::min(rangeMat.at<float>(fromIndX, fromIndY),
                          rangeMat.at<float>(thisIndX, thisIndY));

            if ((*iter).first == 0)
                alpha = segmentAlphaX;
            else
                alpha = segmentAlphaY;

            angle = atan2(d2*sin(alpha), (d1 -d2*cos(alpha)));

            if (angle > segmentTheta){

                queueIndX[queueEndInd] = thisIndX;
                queueIndY[queueEndInd] = thisIndY;
                ++queueSize;
                ++queueEndInd;

                labelMat.at<int>(thisIndX, thisIndY) = labelCount;
                lineCountFlag[thisIndX] = true;

                allPushedIndX[allPushedIndSize] = thisIndX;
                allPushedIndY[allPushedIndSize] = thisIndY;
                ++allPushedIndSize;
            }
        }
    }

    // check if this segment is valid
    bool feasibleSegment = false;
    if (allPushedIndSize >= 30)
        feasibleSegment = true;
    else if (allPushedIndSize >= segmentValidPointNum){
        int lineCount = 0;
        for (size_t i = 0; i < N_SCAN; ++i)
            if (lineCountFlag[i] == true)
                ++lineCount;
        if (lineCount >= segmentValidLineNum)
            feasibleSegment = true;
    }
    // segment is valid, mark these points
    if (feasibleSegment == true){
        ++labelCount;
    }else{ // segment is invalid, mark these points
        for (size_t i = 0; i < allPushedIndSize; ++i){
            labelMat.at<int>(allPushedIndX[i], allPushedIndY[i]) = 999999;
        }
    }
}

void ImageProjection::publishCloud(){
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("viewer"));

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>),cloud_outliner(new pcl::PointCloud<pcl::PointXYZRGB>);;
    for (size_t i = 0; i < segmentedCloud->size(); ++i) {
        pcl::PointXYZRGB p;
        p.x=segmentedCloud->points[i].x;
        p.y=segmentedCloud->points[i].y;
        p.z=segmentedCloud->points[i].z;
        p.r=255;
        p.g=0;
        p.b=0;
        cloud->push_back(p);
    }

    for (size_t i = 0; i < outlierCloud->size(); ++i) {
        pcl::PointXYZRGB p;
        p.x=outlierCloud->points[i].x;
        p.y=outlierCloud->points[i].y;
        p.z=outlierCloud->points[i].z;
        p.r=0;
        p.g=255;
        p.b=0;
        cloud_outliner->push_back(p);
    }

    cout<<cloud->size()<<" "<<cloud_outliner->size()<<endl;

    viewer->addPointCloud(cloud, "data");
    viewer->addPointCloud(cloud_outliner, "outliner");
    while (!viewer->wasStopped())
        viewer->spinOnce();

}