#!/bin/bash

cd cmake-build-debug
./gicp_kitti 03 3 3 1 0 0 0 0 m3 

echo "testing 07"
./gicp_kitti 04 3 3 1 0 0 0 0 m3
