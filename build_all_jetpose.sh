#!/bin/bash

# Tested on Jetson Orin nx 
# Jetpack 6.1 
# cuda 12.4
# python 3.10 (conda env) 
# torch 2.3

# IMP Note : Install all dependencies in environment from offiecial docs and forums (nvidia)

PROJ_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Install miniconda first
if [ ! -d "$HOME/miniconda3" ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O ~/Downloads/miniconda.sh
    bash ~/Downloads/miniconda.sh -b -p $HOME/miniconda3
    echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc
    conda init bash
fi

# Create conda environment
if [ ! -d "$HOME/miniconda3/envs/foundationpose" ]; then
    conda create -n foundationpose python=3.10
fi
conda activate foundationpose

## gcc and g++ version 11
sudo apt-get update && sudo apt-get install -y gcc-11 g++-11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 100

#for GLIBCXX_3.4.30 error
conda install -c conda-forge libstdcxx-ng

export TORCH_CUDA_ARCH_LIST="8.7"
export OPENCV_IO_ENABLE_OPENEXR=1

# Install dependencies

# install torch < 2.5 to avoid conflict with pytorch3d 
# link for pytorch wheels for jetson https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
wget https://nvidia.box.com/shared/static/zvultzsmd4iuheykxy17s4l2n91ylpl8.whl -O ~/Downloads/torch-2.3.0-cp310-cp310-linux_aarch64.whl
wget https://nvidia.box.com/shared/static/xpr06qe6ql3l6rj22cu3c45tz1wzi36p.whl -O ~/Downloads/torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl
wget https://nvidia.box.com/shared/static/9si945yrzesspmg9up4ys380lqxjylc3.whl -O ~/Downloads/torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl
pip install ~/Downloads/torch-2.3.0-cp310-cp310-linux_aarch64.whl ~/Downloads/torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl ~/Downloads/torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl 
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
python -m pip install scipy joblib scikit-learn ruamel.yaml trimesh pyyaml opencv-python imageio open3d transformations warp-lang einops kornia pyrender

# Install dependencies
pip install gdown
# Create the weights directory and download the pretrained weights from FoundationPose
gdown --folder https://drive.google.com/drive/folders/1BEQLZH69UO5EOfah-K9bfI3JyP9Hf7wC -O FoundationPose/weights/2023-10-28-18-33-37 
gdown --folder https://drive.google.com/drive/folders/12Te_3TELLes5cim1d7F7EBTwUSe7iRBj -O FoundationPose/weights/2024-01-11-20-02-45

# Install pybind11
cd ${PROJ_ROOT}/FoundationPose && git clone https://github.com/pybind/pybind11 && \
    cd pybind11 && git checkout v2.10.0 && \
    mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DPYBIND11_INSTALL=ON -DPYBIND11_TEST=OFF && \
    make -j6 && make install

# Install Eigen
cd ${PROJ_ROOT}/FoundationPose && wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz && \
    tar xvzf ./eigen-3.4.0.tar.gz && rm ./eigen-3.4.0.tar.gz && \
    cd eigen-3.4.0 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make install

# Clone and install nvdiffrast
cd ${PROJ_ROOT}/FoundationPose && git clone https://github.com/NVlabs/nvdiffrast && \
    cd /nvdiffrast && pip install .

# Install mycpp
cd ${PROJ_ROOT}/FoundationPose/mycpp/ && \
rm -rf build && mkdir -p build && cd build && \
cmake .. && \
make -j$(nproc)

# Install mycuda (not needed for inference)
# cd ${PROJ_ROOT}/FoundationPose/bundlesdf/mycuda && \
# rm -rf build *egg* *.so && \
# python3 -m pip install -e .

cd ${PROJ_ROOT}