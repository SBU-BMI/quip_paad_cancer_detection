FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-devel
MAINTAINER https://github.com/immune-health/quip_paad_cancer_detection/

ENTRYPOINT []

RUN   apt-get --yes update
RUN   apt-get --yes update --fix-missing
RUN   apt-get --yes install aria2 git htop wget zip
RUN   apt-get --yes install libgl1-mesa-glx libgl1-mesa-dev
RUN   apt-get install --yes python3-openslide tar vim curl
RUN   pip install --upgrade pip
RUN   pip install conda Pillow pymongo

RUN   pip install openslide-python
RUN   conda update -n base -c defaults conda
RUN   conda install --yes scikit-learn
RUN   conda install --yes -c conda-forge opencv &&
RUN   conda install --yes pytorch=0.4.1 cuda90 -c pytorch
RUN   pip install torchvision==0.2.1 && \
      pip install torch==0.4.1

WORKDIR /root

ADD https://github.com/immune-health/quip_paad_cancer_detection/git/refs/heads/master version.json
RUN	git clone https://github.com/immune-health/quip_paad_cancer_detection

RUN   mkdir -p /root/quip_paad_cancer_detection/data/models_cnn
RUN		cd /root/quip_paad_cancer_detection/data/models_cnn
RUN		aria2c -x 16 -j 128 -s 64 --auto-file-renaming=false --file-allocation=falloc \
"https://get.rech.io/inceptionv4_cancer_350px_lr_1e-2_decay_5_jitter_val6slides_harder_pretrained_none_0423_0449_0.8854108440469536_11.t7"
RUN		aria2c -x 16 -j 128 -s 64 --auto-file-renaming=false --file-allocation=falloc \
"https://get.rech.io/RESNET_34_cancer_350px_lr_1e-2_decay_5_jitter_val6slides_harder_pretrained_none_1117_0044_0.8715164676076728_17.t7"
RUN		aria2c -x 16 -j 128 -s 64 --auto-file-renaming=false --file-allocation=falloc \
"https://get.rech.io/VGG16_cancer_350px_lr_1e-2_decay_5_jitter_val6slides_harder_pretrained_none_0423_0456_0.8766822301565503_11.t7"
RUN		aria2c -x 16 -j 128 -s 64 --auto-file-renaming=false --file-allocation=falloc \
"https://get.rech.io/paad_baseline_preact-res34_train_TCGA_ensemble_epoch_7_auc_0.8595125864960883"
RUN		aria2c -x 16 -j 128 -s 64 --auto-file-renaming=false --file-allocation=falloc \
"https://get.rech.io/RESNET_34_prostate_trueVal_hard_train__0530_0015_0.954882634484846_1919.t7"

CMD ["/bin/bash"]
