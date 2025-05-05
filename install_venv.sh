# Create python venv
python3 -m venv .venv

# Activate venv
source .venv/bin/activate

# Upgrade setuptools and wheel
# This is required for torch installation
pip install --upgrade setuptools wheel

### Install dependencies

# install torch < 2.5 to avoid conflict with pytorch3d 
# link for pytorch wheels for jetson https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
wget https://nvidia.box.com/shared/static/zvultzsmd4iuheykxy17s4l2n91ylpl8.whl -O ~/Downloads/torch-2.3.0-cp310-cp310-linux_aarch64.whl
wget https://nvidia.box.com/shared/static/xpr06qe6ql3l6rj22cu3c45tz1wzi36p.whl -O ~/Downloads/torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl
wget https://nvidia.box.com/shared/static/9si945yrzesspmg9up4ys380lqxjylc3.whl -O ~/Downloads/torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl
pip install ~/Downloads/torch-2.3.0-cp310-cp310-linux_aarch64.whl ~/Downloads/torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl ~/Downloads/torchaudio-2.3.0+952ea74-cp310-cp310-linux_aarch64.whl 

# Install pytorch3d
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Install other dependencies
pip install scipy joblib scikit-learn ruamel.yaml trimesh pyyaml opencv-python imageio open3d transformations warp-lang einops kornia pyrender

pip install psutil pandas matplotlib omegaconf h5py

# Downgrade numpy
pip install "numpy<2"

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

# Install Cutie (FoundationPose++)
cd ${PROJ_ROOT}/submodules/FoundationPose/Cutie
pip install -e .