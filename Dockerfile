FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-devel 
MAINTAINER Tahsin Kurc

RUN	apt-get -y update && \
	apt-get install --yes python3-openslide wget zip libgl1-mesa-glx libgl1-mesa-dev && \
	pip install pip==21.0.1 && \
	conda update -n base -c defaults conda && \
	pip3 install setuptools==45 && \
	pip install cython && \
	conda install --yes pytorch=0.4.1 cuda90 -c pytorch && \
	conda install --yes scikit-learn && \
	pip install "Pillow<7" pymongo pandas && \
	pip install torchvision==0.2.1 && \
	conda install --yes -c conda-forge opencv

RUN	pip install openslide-python

ENV 	BASE_DIR="/quip_app/quip_paad_cancer_detection"
ENV 	PATH="./":$PATH

ENV	MODEL_VER="v1.0"
ENV	MODEL_URL="https://github.com/SBU-BMI/quip_paad_cancer_detection/blob/develop/models_cnn/paad_baseline_preact-res34_train_TCGA_ensemble_epoch_7_auc_0.8595125864960883"

COPY    . ${BASE_DIR}/.

RUN     chmod 0755 ${BASE_DIR}/scripts/*

WORKDIR ${BASE_DIR}/scripts

CMD ["/bin/bash"]
