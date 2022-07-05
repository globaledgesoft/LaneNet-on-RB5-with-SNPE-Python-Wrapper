# Lane Detection on RB5 using SNPE Python Wrapper
## Introduction
The project is used to demonstrate the Lane detection on the Qualcomm’s Robotics Platform RB5. In this, it shows developers that how to utilize the Qualcomm’s Robotics platform for the Autonomous Vehicle use cases. LaneNet is the architecture used to implement this solution on top of the TuSimple Dataset. the model trained was converted from PyTorch to ONNX & then ONNX to SNPE. The inference using SNPE on RB5 with DSP Hardware accelerator, it achieves the performance of 45 FPS.


## Prerequisites 
- A Linux host system with Ubuntu 18.04.
- Install Android Platform tools (ADB, Fastboot) 
- Download and install the SDK Manager for RB5
- Flash the RB5 firmware image on to the RB5
- Setup the Network on RB5.
- Installed Python3.6 on RB5.



## Steps to Setup the Lane Detection on RB5
### Installing Dependencies
- OpenCV Installation on RB5

Run the command given below to install the OpenCV on RB5,

```sh
sh4.4 # python3 -m pip install --upgrade pip
sh4.4 # python3 -m pip install opencv-python 
```

- Setting ONNX & SNPE on Host System
  - Download the SNPE SDK from the following link on the host system: https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk/tools

  - Setup the ONNX on the Host System

    Run the command given below on host system to setup the ONNX
    ```sh
    sh4.4 # python3 -m pip install onnxruntime
    ```
  - Follow the instruction in link mentioned below to setup the SNPE
https://developer.qualcomm.com/sites/default/files/docs/snpe/overview.html

- PyBind11 Installation on RB5 

  Run the command given below to setting up the PyBind11
  ```sh
  sh4.4 # apt update && apt install python3-pybind11
  ```

- Setting up the SNPE Libraries on RB5
  1. Copy the SNPE header files & runtime libraries for `aarch64-ubuntu-gcc7.5` on RB5 from host system using ADB
  ```sh
  sh4.4 # adb push <SNPE_ROOT>/include/ /data/snpe/include/
  sh4.4 # adb push <SNPE_ROOT>/lib/aarch64-ubuntu-gcc7.5/* /data/snpe/
  sh4.4 # adb push <SNPE_ROOT>/lib/dsp/* /data/snpe/
  ```
  `Note: If device is connected via SSH, please use scp tool for copying the SNPE runtime libraries in /data/snpe folder on RB5.`

  2.	Open the terminal of RB5 and append the lines given below at the end of `~/.bashrc` file.
  ```sh
  export PATH=$PATH:/data/snpe/
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/snpe/
  export ADSP_LIBRARY_PATH="/data/snpe;/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp"
  ```

  3.	Run the command given below to reinitialize the RB5's terminal session
  ```sh
  sh4.4 # source ~/.bashrc
  ```


### Model Conversion to DLC using SNPE on Host System
1. Follow the instructions in link given below to train your own model on host
   
   https://github.com/cfzd/Ultra-Fast-Lane-Detection
   
   `Note: Must recommended to use GPU Server for model training`

2. Clone the project and run below command to convert the trained PyTorch model to ONNX model on host.
   ```sh
   ubuntu@ubuntu $ cd <PROJECT_PATH>/models
   ubuntu@ubuntu $ python3 pytorch_to_onnx.py
   ```
3. Make sure that for performing the above steps Pytorch & ONNX is installed on your host system.
4. Before running the steps given below, make sure that SNPE SDK & ONNX Runtime has been installed on the host-system.
5. Run below command from <SNPE_ROOT> for Initializing the SNPE SDK for the ONNX Environment
```sh
ubuntu@ubuntu $ source bin/envsetup.sh -o <ONNX_DIR>
```
`Note: <ONNX_DIR> is path to python package path of ONNX package.

6. Run the command given below for converting the model to DLC
```sh
ubuntu@ubuntu $ snpe-onnx-to-dlc --input_network lanenet.onnx --input_dim input.1 1,3,288,800 –output_path lanenet.dlc
```
7.	Copy the converted lanenet.dlc inside the RB5 in <PROJECT_PATH>
```sh
ubuntu@ubuntu $ adb push lanenet.dlc <PROJECT_PATH_ON_RB5>/models
```

### Building the SNPE Python Wrapper for Lane Detection Project
1.	Clone the project from the link below on the RB5,
```sh
sh4.4 # git clone https://github.com/globaledgesoft/LaneNet-on-RB5-with-SNPE-Python-Wrapper.git
```

2.	Go inside the src folder of cloned project,
```sh
sh4.4 # cd <PROJECT_PATH>
```

3.	Run the command below  in order to build the shared library for Python wrapper of the SNPE.
```sh
sh4.4 # g++ -std=c++11 -fPIC -shared -o qcsnpe.so src/qcsnpe.cpp -I include/ -I /data/snpe/include/zdl/ -I /usr/include/python3.6m/ -I /usr/local/lib/python3.6/dist-packages/pybind11/include -L /data/snpe/ -lSNPE `pkg-config --cflags --libs opencv`
```

## Running the Lane Detection using SNPE Python Wrapper on the RB5
1.	Go to the <PROJECT_PATH>
```sh
sh4.4 # cd <PROJECT_PATH>/
```

2.	Running the Lane Detection application,
```sh
sh4.4 # python3 main_realtime_lane_detect.py
```
