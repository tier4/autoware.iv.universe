# NDT_OCL

The ndt_ocl is a yet another ndt based scan-matching package which is based on ndt_omp in AutowareArchitectureProposal.
Parallel algorithms in ndt scan matcher would be executed efficiently on various devices using OpenCL.

## Environments

* Ubuntu 18.04
* AutowareArchitectureProposal (v0.9.1 based)
* Devices supporting OpenCL 2.x
* ROS melodic

## Setting up OpenCL 2.1 (ex. Intel OpenCL Environment)

### Install and Setup OpenCL™ Runtimes for Intel Processors

1. Download Runtime Installer from https://software.intel.com/content/www/us/en/develop/articles/opencl-drivers.html#cpu-section
  * Version: 18.1
  * l_opencl_p_18.1.0.015.tgz
2. Extract and execute setup command below
   ```
   $ sudo ./install_GUI.sh
   ```
3. Install the runtime from the GUI installer.
4. Execute `clinfo` to check if the install was seccessful.
   You can see the Intel Platform.
   ```
   $ clinfo
   ...
   Number of platforms                               2
     Platform Name                                   Intel(R) CPU Runtime for OpenCL(TM) Applications
     Platform Vendor                                 Intel(R) Corporation
     Platform Version                                OpenCL 2.1 LINUX
     Platform Profile                                FULL_PROFILE
   ...
   platform Name                                   Intel(R) CPU Runtime for OpenCL(TM) Applications
   Number of devices                                 1
     Device Name                                     Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
     Device Vendor                                   Intel(R) Corporation
     Device Vendor ID                                0x8086
     Device Version                                  OpenCL 2.1 (Build 0)
   ```

### Install and Setup THE INTEL® SDK FOR OPENCL™ APPLICATIONS

1. Download SDK Installer from https://software.seek.intel.com/intel-opencl?os=linux
   * Version: 2020.3.494
   * intel_sdk_for_opencl_applications_2020.3.494.tar.gz
2. Extract and execute setup command below
   ```
   $ sudo ./install.sh
   ```
3. Install the SDK from the GUI installer.


## Configurations for NDT OCL

### include/ndt_ocl/ndt_ocl.h

* MAX_SOURCE_SIZE: Maximum size of the source kernel code. Be careful to this value when you modify the `compute_derivatives.cl`.

* **MAX_PCL_INPUT_NUM**: Maximum size of # of point cloud in LiDAR input for `computeDerivatives()`. Default is 1,500.

* **LIMIT_NUM**: Maximum size of # of voxels retured from neighbor search method, or `radiusSearch`.

### include/ndt_ocl/voxel_grid_covariance_ocl.h

* **MAX_PCL_MAP_NUM**: Maximum size of # of point cloud in MAP.

* MAX_DEPTH: Maximum searching deapth of DFS for KD-tree in neighbor search method. When the given KD-tree is heigher than MAX_DEPTH, the DFS is force quitting at the MAX_DEPTH.

## Launch Configurations

Change the `ndt_implement_type` to "3" in `launch/autoware_launcher/localization_launch/config/ndt_scan_matcher.yaml` as below.

You can still set "2" to use ndt_omp, OpenMP accelerated ndt scan matcher.

```
# NDT implementation type
# 0=PCL_GENERIC, 1=PCL_MODIFIED, 2=OMP, 3=OpenCL
ndt_implement_type: 3
```
