FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-devel
MAINTAINER https://github.com/immune-health/quip_paad_cancer_detection/

ENTRYPOINT []

# install required apt

RUN  apt-get --yes update
RUN  apt-get --yes update --fix-missing
RUN  apt-get --yes install wget aria2 htop git vim zip
RUN  apt-get --yes install libgl1-mesa-glx libgl1-mesa-dev

# python

RUN  pip install --upgrade pip
RUN  pip install conda Pillow pymongo
RUN  conda update -n base -c defaults conda
RUN  conda install --yes scikit-learn
RUN  conda install --yes -c conda-forge opencv
RUN  conda install --yes pytorch=0.4.1 cuda90 -c pytorch

# open slide must be installed after conda

RUN  pip install openslide-python
RUN  apt-get --yes install python3-openslide tar vim curl

# last: reset torchvision, torch

RUN  pip install torchvision==0.2.1 && \
     pip install torch==0.4.1

# download code and data

ADD   https://api.github.com/repos/immune-health/quip_paad_cancer_detection/git/refs/heads/master version.json
RUN  cd /root/ && \
				git clone https://github.com/immune-health/quip_paad_cancer_detection
RUN  mkdir -p /root/quip_paad_cancer_detection/data/models_cnn

# download data

RUN  cd /root/quip_paad_cancer_detection/data/models_cnn && \
			aria2c -x 16 -j 128 -s 64 \
			--allow-overwrite=true --auto-file-renaming=false --file-allocation=falloc \
			"https://get.rech.io/paad_baseline_preact-res34_train_TCGA_ensemble_epoch_7_auc_0.8595125864960883"

WORKDIR /root
ENV	PATH="./":$PATH
CMD ["/bin/bash"]
