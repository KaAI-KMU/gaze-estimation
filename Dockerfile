FROM ubuntu:latest

# 기본 패키지 업데이트 및 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    python3-dev \
    python3-pip

# 사용자 전환
RUN useradd -ms /bin/bash dockeruser
USER dockeruser

# pip 업그레이드
RUN python3 -m pip install --upgrade pip

# dlib 설치
RUN pip3 install dlib

# 작업 디렉토리로 이동
WORKDIR /workspace

