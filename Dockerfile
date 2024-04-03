FROM fairembodied/habitat-challenge:testing_2020_habitat_base_docker


RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update
RUN apt-get install -y libglvnd-dev libnvidia-gl-450 libgl1-mesa-dev libegl1-mesa-dev libglm-dev libjpeg-dev libpng-dev libglfw3-dev build-essential cmake
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    vim \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libglfw3-dev \
    libglm-dev \
    libx11-dev \
    libomp-dev \
    libegl1-mesa-dev \
    pkg-config \
    wget \
    zip \
    htop \
    tmux \
    unzip 
RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev
SHELL [ "/bin/bash", "-c" ]
RUN echo "source activate habitat" > ~/.bashrc
RUN echo "alias ll='ls -alh'" >> ~/.bashrc
RUN conda init && source ~/.bashrc && conda activate habitat && python3 -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
RUN conda init && source ~/.bashrc && conda activate habitat && conda install -y pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
COPY requirements.txt .
RUN conda init && source ~/.bashrc && conda activate habitat && pip install -r requirements.txt
RUN mkdir /code
WORKDIR /code

CMD [ "/bin/bash" ]