# fasterGICP

This package is an improvement of [fast_gicp](https://github.com/SMRT-AIST/fast_gicp) 

Please cite our paper if possible.

W. Jikai, M. Xu, F. Farzin, D. Dai and Z. Chen, "FasterGICP: Acceptance-rejection Sampling based 3D Lidar Odometry," in IEEE Robotics and Automation Letters, doi: 10.1109/LRA.2021.3124072.

## Installation

### Dependencies
- PCL
- Eigen
- OpenMP
- CUDA (optional)
- [Sophus](https://github.com/strasdat/Sophus)
- [nvbio](https://github.com/NVlabs/nvbio)

We have tested this package on Ubuntu 18.04/20.04 and CUDA 11.1.

### CUDA

To enable the CUDA-powered implementations, set ```BUILD_VGICP_CUDA``` cmake option to ```ON```.

### Non-ROS
```bash
git clone https://github.com/SMRT-AIST/fast_gicp --recursive
mkdir fast_gicp/build && cd fast_gicp/build
cmake .. -DCMAKE_BUILD_TYPE=Release
# enable cuda-based implementations
# cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_VGICP_CUDA=ON
make -j8
```

## Test on KITTI

### C++

```bash
cd fasterGICP/build
# reading program parameters from fasterGICP/include/settings.txt 
./gicp_kitti
```

## Related packages
- [ndt_omp](https://github.com/koide3/ndt_omp)
- [fast_gicp](https://github.com/SMRT-AIST/fast_gicp)


## Papers

## Contact

