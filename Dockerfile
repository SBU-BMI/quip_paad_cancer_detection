FROM continuumio/miniconda3:4.9.2
LABEL maintainer="Jakub Kaczmarzyk <jakub.kaczmarzyk@stonybrookmedicine.edu>"

RUN	apt-get update \
	&& apt-get install --yes \
		gcc \
		libgl1-mesa-glx \
		openslide-tools \
	&& rm -rf /var/lib/apt/lists/*

# TODO: this should go in an environment.yml file.
RUN conda create -n histo --yes --quiet \
		--channel pytorch \
		--channel conda-forge \
			cuda90 \
			pandas \
			"pillow<7" \
			pytorch=0.4.1 \
			scikit-learn \
			torchvision==0.2.1 \
	&& /opt/conda/envs/histo/bin/python -m pip install --no-cache-dir \
		opencv-python \
		openslide-python \
		pymongo
ENV "/opt/conda/envs/histo/bin:$PATH"

WORKDIR /quip_app/quip_paad_cancer_detection
COPY . .
RUN chmod +x scripts/svs_2_heatmap.sh
ENV PATH="/quip_app/quip_paad_cancer_detection/scripts:$PATH"
WORKDIR scripts
CMD ["/bin/bash"]
