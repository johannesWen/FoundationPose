# FoundationPose: Unified 6D Pose Estimation and Tracking of Novel Objects

## Hardware & Software
- Nvidia Jetson Orin AGX 64GB
    - nvidia-jetpack 6.2+b77
    - linux-firmware-nvidia-tegra 36.4.3-20250107174145-0ubuntu0.22.04
    - Cuda release 12.6, V12.6.68
- Intel RealSense D435

## Installation

The gitub repository [Jetpose](https://github.com/Kaivalya192/Jetpose) was very helpful in getting the code to work. The following steps were followed to install [Foundationpose](https://github.com/johannesWen/FoundationPose) on the Nvidia Jetson Orin AGX.

### Prerequisites

1. Clone the repository

    ```bash
    git clone https://git.lcm.at/mecon/smart/keba-reinforcement-learning/object-tracking/custom-object-tracker.git
    ```

1. Checkout the submodules

    ```bash
    git submodule init
    ```
    ```bash
    git submodule update --init --recursive
    ```

### Foundationpose++
1. Create a virtual environment

2. Install the [Foundationpose](https://github.com/johannesWen/FoundationPose) / [Foundationpose++](https://github.com/teal024/FoundationPose-plus-plus) dependencies and follow the steps from [install_venv.sh](https://github.com/johannesWen/FoundationPose/blob/main/install_venv.sh).

    - slow `pytorch3d` build

        If the build of `pytorch3d` is very slow, probably the ninja build system is not installed. You can install it with the following command:

        ```bash
        sudo apt install ninja-build
        ```

        Then install / build `pytorch3d` with the following command:

        ```bash
        MAKEFLAGS="-j$(nproc)" pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
        ```


3. Copy the weights from [Google Drive](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i) or [LCM Sharepoint](https://lcmgmbh-my.sharepoint.com/:f:/g/personal/ba2_od1_lcm_at/EipBdJQpUN9KqPjUwHIVpKIBMGjBON5ZnvuqDWAMGrs8FQ?e=1w6fJ9) to `submodules/FoundationPose/weights`.

### Intel RealSense D435

1. Install the RealSense SDK and follow the steps [install_realsense.sh](https://github.com/johannesWen/FoundationPose/blob/main/install_realsense.sh).

2. To enable `pyrealsense2` support, copy the required `.so` files to the folder where the script will run in the Jetpose directory. Example:

```bash
cp ~/librealsense_build/librealsense-master/build/release/pyrealsense2.cpython-310-aarch64-linux-gnu.so.2.55.1 ~/Jetpose/FoundationPose/pyrealsense2.so
cp ~/librealsense_build/librealsense-master/build/release/librealsense2.so.2.55.1 ~/Jetpose/FoundationPose/librealsense2.so
cp ~/librealsense_build/librealsense-master/build/release/librealsense2-gl.so.2.55.1 ~/Jetpose/FoundationPose/librealsense2-gl.so
```

## Usage

### Run the demo with the Intel RealSense D435

1. Activate the virtual environment:

    ```bash
    source ./.venv/bin/activate
    ```

1. Enable Maximum Performance Mode: Ensure your device operates at its highest performance capacity by setting it to maximum performance mode. Execute the following commands:

    ```bash
    sudo nvpmodel -m 0
    sudo jetson_clocks
    ```

1. Edit the arguments of [run_live_FoundationPose_pp.py](https://github.com/johannesWen/FoundationPose/blob/main/run_live_FoundationPose_pp.py) file.
    - Set the `object_dir` argument to directory of the object you want to track. The directory should contain the following files:
        - `mesh.obj`: The mesh file of the object.
        - `mesh.mtl`: (optional) The material file of the object.
        - `texture.png`: (optional) The texture file of the object.
    - Set the `mesh_file` argument to the relative path (object_dir) of the mesh file (.obj) you want to use.
    - Set the `camera_calibration_file` argument to the relative path of the camera calibration file. The camera calibration file is a `.txt` file that contains the intrinsic and extrinsic parameters of the camera. The camera calibration file is extracted from the RealSense camera.

1. Run the live demo:

    ```bash
    python run_live_FoundationPose_pp.py
    ```

