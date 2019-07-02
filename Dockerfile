FROM pytorch/pytorch:0.4.1-cuda9-cudnn7-devel 
MAINTAINER Tahsin Kurc

RUN     apt-get -y update && \
        apt-get install --yes python3-openslide wget zip libgl1-mesa-glx libgl1-mesa-dev && \
        pip install --upgrade pip && \
        conda install --yes scikit-learn && \
        pip install Pillow pymongo && \
        pip install torchvision==0.2.1 && \
        pip install openslide-python && \
        conda install --yes -c conda-forge opencv

COPY    . /root/quip_paad_cancer_detection/.

RUN     chmod 0755 /root/quip_paad_cancer_detection/scripts/*

ENV BASE_DIR="/root/quip_paad_cancer_detection"
ENV PATH="./":$PATH
WORKDIR /root/quip_paad_cancer_detection/scripts

CMD ["/bin/bash"]
