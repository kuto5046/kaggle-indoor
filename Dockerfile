# pytorch versionに注意
FROM nvidia/cuda:11.1-cudnn8-runtime-ubuntu20.04

# 時間設定
RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime

ENV DEBIAN_FRONTEND=noninteractive
# install basic dependencies
RUN apt-get -y update && apt-get install -y \
    sudo \
    wget \
    cmake \
    vim \
    git \
    tmux \
    zip \
    unzip \
    gcc \
    g++ \
    build-essential \
    ca-certificates \
    software-properties-common \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libpng-dev \
    libfreetype6-dev \
    libgl1-mesa-dev \
    libsndfile1 \
    zsh \
    xonsh \
    neovim

# download anaconda package and install anaconda
WORKDIR /opt
RUN wget https://repo.continuum.io/archive/Anaconda3-2020.11-Linux-x86_64.sh && \
sh /opt/Anaconda3-2020.11-Linux-x86_64.sh -b -p /opt/anaconda3 && \
rm -f Anaconda3-2020.11-Linux-x86_64.sh

# set path
ENV PATH /opt/anaconda3/bin:$PATH
# ENV SLACK_WEBHOOK_URL=$SLACK_WEBHOOK_URL

COPY ./requirements.txt /
# install common python packages
RUN pip install --upgrade pip setuptools && \
    pip install -r /requirements.txt
# https://qiita.com/Hiroaki-K4/items/c1be8adba18b9f0b4cef
RUN pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html

# jupyter用にportを開放
EXPOSE 8888
EXPOSE 5000
EXPOSE 6006

# add user
ARG DOCKER_UID=1000
ARG DOCKER_USER=kuzira
ARG DOCKER_PASSWORD=kuzira
RUN useradd -m --uid ${DOCKER_UID} --groups sudo ${DOCKER_USER} \
  && echo ${DOCKER_USER}:${DOCKER_PASSWORD} | chpasswd

# kaggle setup
# for root
# RUN mkdir ~/.kaggle
# COPY ./kaggle.json /root/.kaggle/
# RUN chmod 600 /root/.kaggle/kaggle.json
# RUN git clone https://github.com/kuto5046/dotfiles.git
# RUN bash dotfiles/.bin/install.sh 

# for user
RUN mkdir /home/${DOCKER_USER}/.kaggle
COPY ./kaggle.json /home/${DOCKER_USER}/.kaggle/
# set working directory
RUN mkdir /home/${DOCKER_USER}/work
WORKDIR /home/${DOCKER_USER}/work
# 本当はよくないがkaggle cliがuserで使えないので600 -> 666
RUN chmod 666 /home/${DOCKER_USER}/.kaggle/kaggle.json

# switch user
USER ${DOCKER_USER}

RUN git clone https://github.com/kuto5046/dotfiles.git /home/${DOCKER_USER}/dotfiles
RUN bash /home/${DOCKER_USER}/dotfiles/.bin/install.sh 
