# Pancreas cancer (PAAD) detection pipeline

This software implements the pipeline for the Pancrease cancer detection project. The repo contains codes to perform the prediction to detect Pancreatic cancer in Whole Slide Images. The paper was publised in MICCAI 2019 "Pancreatic Cancer Detection in Whole Slide Images Using Noisy Label Annotations"

## TCGA Data:
+ Output heatmaps: *.png files, pixel's value is the predicted probablity of the patch to contain cancerous cells. Download output heatmaps [here](https://drive.google.com/drive/folders/14z84TUy6R_UEEAdbXXOWNPzT0e2NKkOJ?usp=sharing).


## Citation:
    @inproceedings{le2019paad,
      title={Pancreatic Cancer Detection in Whole Slide Images Using Noisy Label Annotations},
      author={Han, Le and Dimitris, Samaras and Tahsin, Kurc and Rajarsi, Gupta and Kenneth, Shroyer and Joel, Saltz },
      booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
      year={2019},
      organization={Springer}
    }



# Dependencies

 - [Pytorch 0.4.0](http://pytorch.org/)
 - Torchvision 0.2.0
 - cv2 (3.4.1)
 - [Openslide 1.1.1](https://openslide.org/api/python/)
 - [sklearn](https://scikit-learn.org/stable/)
 - [PIL](https://pillow.readthedocs.io/en/3.1.x/reference/Image.html)

# List of folders and functionalities are below: 

- scripts/: contains scripts that connect several sub-functionalities together for complete functionalities such as generating camicroscope heatmaps given svs images.

- conf/: contains configuration. 

- data/: a place where should contain all logs, input/output images, trained CNN models, and large files. 

- download_heatmap/: downloads grayscale lymphocyte or tumor heatmaps

- heatmap_gen/: generate json files that represents heatmaps for camicroscope, using the lymphocyte and necrosis CNNs' raw output txt files. 

- patch_extraction_tumor_40X/: extracts all patches from svs images. Mainly used in the test phase. 

- prediction/: CNN prediction code. 

- training/: CNN training code. 

# Docker Instructions 

Build the docker image by: 

docker build -t cancer_prediction .  (Note the dot at the end). 

## Step 1:
Create folder named "data" and subfoders below on the host machine:

- data/svs: to contains *.svs files
- data/patches: to contain output from patch extraction
- data/log: to contain log files
- data/heatmap_txt: to contain prediction output
- data/heatmap_jsons: to contain prediction output as json files

## Step 2:
- Run the docker container as follows: 

nvidia-docker run --name cancer_prediction_pipeline -itd -v <path-to-data>:/data -e CUDA_VISIBLE_DEVICES='<cuda device id>' cancer_prediction svs_2_heatmap.sh 

CUDA_VISIBLE_DEVICES -- set to select the GPU to use 

The following example runs the cancer detection pipeline. It will process images in /home/user/data/svs and output the results to /home/user/data. 

nvidia-docker run --name cancer_prediction_pipeline -itd -v /home/user/data:/data -e CUDA_VISIBLE_DEVICES='0' cancer_prediction svs_2_heatmap.sh
