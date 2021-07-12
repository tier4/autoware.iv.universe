# NDT_OCL

The ndt_ocl is a yet another ndt based scan-matching package which is based on ndt_omp in AutowareArchitectureProposal.
Parallel algorithms in ndt scan matcher would be executed efficiently on various devices using OpenCL.

## Environments

* Ubuntu 18.04
* AutowareArchitectureProposal (v0.9.1 based)
* Devices supporting OpenCL 2.x
* ROS melodic

## Requirements (ex. Intel OpenCL Environment)

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
