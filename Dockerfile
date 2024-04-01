FROM fairembodied/habitat-challenge:testing_2020_habitat_base_docker

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