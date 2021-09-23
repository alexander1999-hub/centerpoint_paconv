FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu20.04

#install required libs and packages
RUN apt-get update && apt-get install -y --no-install-recommends wget build-essential git curl ca-certificates libjpeg-dev libpng-dev python3-pip \
	lsb-release ninja-build

ENV FORCE_CUDA="1"

# ROS Noetic installation
ARG DEBIAN_FRONTEND=noninteractive
RUN apt update && sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
	&& apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 \
	&& apt update && apt install -y ros-noetic-desktop


# Update pip & install pytorch
RUN python3 -m pip install pip --upgrade
RUN python3 -m pip install wheel
RUN pip install pyyaml numpy ipython
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html

# CMake installation
RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.16.4/cmake-3.16.4-Linux-x86_64.tar.gz
RUN tar -xzf cmake-3.16.4-Linux-x86_64.tar.gz --exclude=bin/cmake-gui --exclude=doc/cmake --exclude=share/cmake-3.12/Help  \
	&& cp -fR cmake-3.16.4-Linux-x86_64/* /usr  \
	&& rm -rf cmake-3.16.4-Linux-x86_64  \
	&& rm cmake-3.16.4-Linux-x86_64.tar.gz

# SpConv installation
RUN git clone https://github.com/traveller59/spconv.git --recursive
RUN cd spconv  \
	# && git checkout 7342772  \
	&& SPCONV_FORCE_BUILD_CUDA=1 python3 setup.py bdist_wheel  \
	&& cd ./dist  \
	&& pip install *  \
	&& cd ../..

# # Apex installation
WORKDIR /tmp/unique_for_apex
RUN git clone https://github.com/NVIDIA/apex.git  \
 	&& cd apex \
 	&& git checkout 5633f6 \
	&& python3 setup.py install

# installing ROS and centerpoint packages
RUN apt update && apt install -y ros-noetic-pcl-msgs ros-noetic-pcl-ros ros-noetic-image-geometry ros-noetic-tf2 ros-noetic-tf2-sensor-msgs \
	ros-noetic-tf-conversions ros-noetic-geographic-info && apt clean all && rm -rf /var/lib/apt/lists/*
RUN	pip install rospkg scikit-image scipy numba matplotlib fire protobuf opencv-python opencv-contrib-python pybind11 vtk joblib \
	easydict open3d-python terminaltables pytest-runner addict pycocotools imagecorruptions objgraph cachetools descartes \
	motmetrics pandas Pillow pyquaternion scikit-learn Shapely tqdm requests empy rosdep filterpy

RUN git clone https://github.com/eric-wieser/ros_numpy.git \
	&& cd ros_numpy  \
	&& python3 setup.py install \
	&& cd .. \ 
	&& rm -r ros_numpy

RUN pip install scipy

RUN pip install h5py

RUN pip install tensorboardX

RUN python3 -m pip install --upgrade pip

RUN pip install --ignore-installed PyYAML

RUN pip install testresources

RUN pip install open3d

RUN git clone https://github.com/gdlg/simple-waymo-open-dataset-reader.git \
	&& cd simple-waymo-open-dataset-reader \
	&& python3 setup.py install \
	&& cd .. \
	&& rm -r simple-waymo-open-dataset-reader

RUN pip install waymo-open-dataset-tf-2-3-0==1.3.1

ARG USER=docker_centerpoint
ARG UID=1000
ARG GID=1000
# default password
ARG PW=user 

RUN useradd -m ${USER} --uid=${UID} && echo "${USER}:${PW}" | chpasswd && adduser ${USER} sudo

WORKDIR /home/${USER}
RUN cd /home/${USER} && mkdir -p catkin_ws/src && chown -R ${UID}:${GID} ./

USER ${UID}:${GID}

