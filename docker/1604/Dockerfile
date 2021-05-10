
FROM ubuntu:16.04

#RUN add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"
RUN apt-get update
RUN apt install -y libjasper-dev
RUN apt-get install -y cmake git libgtk2.0-dev pkg-config libavcodec-dev \
libavformat-dev libswscale-dev python-dev python-numpy libtbb2 libtbb-dev \
libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev unzip

RUN apt-get install -y wget

# Just get a simple editor for convienience (you could just cancel this line)
RUN apt-get install -y vim


# Second: get and build OpenCV 3.2
#
RUN cd \
    && wget https://github.com/opencv/opencv/archive/3.2.0.zip \
    && unzip 3.2.0.zip \
    && cd opencv-3.2.0 \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make -j8 \
    && make install \
    && cd \
    && rm 3.2.0.zip


# Third: install and build opencv_contrib
#
RUN cd \
    && wget https://github.com/opencv/opencv_contrib/archive/3.2.0.zip \
    && unzip 3.2.0.zip \
    && cd opencv-3.2.0/build \
    && cmake -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-3.2.0/modules/ .. \
    && make -j8 \
    && make install \
    && cd ../.. \
    && rm 3.2.0.zip

#RUN apt-get install -y libgflags−dev libgoogle−glog−dev libgtest−dev
RUN apt-get install -y libcxsparse3.1.4 

# DOWNLOAD EIGEN FROM SOURCE AND INSTALL IT
RUN cd \
    && git clone https://gitlab.com/libeigen/eigen.git \
    && cd eigen \
    && mkdir build \
    && cd build \
    && cmake ..
    && make install

#RUN apt-get install -y libeigen3-dev
#RUN apt-get install -y liblapack−dev \
#                   libsuitesparse−dev \

# https://github.com/ceres-solver/ceres-solver/issues/541#issuecomment-662823031 
RUN cd \ 
    && git clone https://github.com/ceres-solver/ceres-solver.git \
    && cd ceres-solver \
    && git reset --hard e51e9b46f6 \
    && mkdir build \
    && cd build 
#    && cd build \
#    && cmake .. \
#    && make -j4 \
#    && make install


RUN cd \
    && git clone https://github.com/strasdat/Sophus.git \
    && cd Sophus \
    && mkdir build \
    && cd build 
#    && cd build \
#    && cmake .. \
#    && make -j4 \
#    && make install

# Forth: get and build the Learning OpenCV 3 examples:
#    I copy the needed data to where the executables will be: opencv-3.2.0/build/bin
#
#RUN cd \
#    && git clone https://github.com/oreillymedia/Learning-OpenCV-3_examples.git \
#    && cd Learning-OpenCV-3_examples \
#    && mkdir build \
#    && cd build \
#    && cmake .. \
#    && make -j8

#You could then run python and do your stuff...
#CMD ["python"]
