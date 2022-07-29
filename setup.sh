#!/bin/bash
# Milan Ondrasovic <milan.ondrasovic@gmail.com>

# Install required packages before building Python from source.
sudo apt-get -y install liblzma-dev lzma wget git

# Compile Python from source
wget https://www.python.org/ftp/python/3.7.13/Python-3.7.13.tar.xz
tar xf Python-3.7.13.tar.xz
cd Python-3.7.13
./configure --enable-optimization
make -j 16
cd ..

sed --in-place '26 i int_classes=int' /home/mond/s/ssd/venvs/ml/lib/python3.7/site-packages/torch/_six.py

pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install \
    ipython click tqdm \
    cython numpy opencv-python matplotlib imgaug Pillow \
    gluoncv motmetrics pycocotools yacs \
    mxnet tensorboard scikit_learn \
    ninja addict decord indexed fire backports.lzma timm \
    ffmpeg ffmpeg-python

# FFMPEG needs to be uninstalled and then installed again to prevent certain runtime errors.
pip uninstall -y ffmpeg-python && pip install ffmpeg-python

git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI/
python3 setup.py build_ext install
cd ../..

git clone https://github.com/mcordts/cityscapesScripts.git
cd cityscapesScripts/
python3 setup.py build_ext install
cd ..

git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./
cd ..

git clone https://github.com/facebookresearch/maskrcnn-benchmark.git
cd maskrcnn-benchmark
cuda_dir="maskrcnn_benchmark/csrc/cuda"
perl -i -pe 's/AT_CHECK/TORCH_CHECK/' $cuda_dir/deform_pool_cuda.cu $cuda_dir/deform_conv_cuda.cu
sed -i 's/PY3/PY37/g' ./maskrcnn_benchmark/utils/imports.py
python3 setup.py build develop
cd ..
