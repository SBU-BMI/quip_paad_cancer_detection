#!/bin/bash

# Variables
DEFAULT_OBJ=20
DEFAULT_MPP=0.5
MONGODB_HOST=XXX
MONGODB_PORT=27017
CANCER_TYPE=paad

if [[ -z "${HEATMAP_VERSION}" ]]; then
	export HEATMAP_VERSION=cancer-paad
fi

# Base data and output directories
export BASE_DIR=/root/quip_paad_cancer_detection
export DATA_DIR=/data
export OUT_DIR=${DATA_DIR}/output

# Prediction folders
# Paths of data, log, input, and output
export JSON_OUTPUT_FOLDER=${OUT_DIR}/heatmap_jsons
export HEATMAP_TXT_OUTPUT_FOLDER=${OUT_DIR}/heatmap_txt
export LOG_OUTPUT_FOLDER=${OUT_DIR}/log
export SVS_INPUT_PATH=${DATA_DIR}/svs
export PATCH_PATH=${DATA_DIR}/patches

export LYM_NECRO_CNN_MODEL_PATH=${BASE_DIR}/models_cnn
export MODEL="paad_baseline_preact-res34_train_TCGA_ensemble_epoch_7_auc_0.8595125864960883"

# Training folders
# The list of case_ids you want to download heaetmaps from
export CASE_LIST=${DATA_DIR}/raw_marking_to_download_case_list/case_list.txt
export PATCH_SAMPLING_LIST_PATH=${DATA_DIR}/patch_sample_list
export RAW_MARKINGS_PATH=${DATA_DIR}/raw_marking_xy
export MODIFIED_HEATMAPS_PATH=${DATA_DIR}/modified_heatmaps
export TUMOR_HEATMAPS_PATH=${DATA_DIR}/tumor_labeled_heatmaps
export TUMOR_GROUND_TRUTH=${DATA_DIR}/tumor_ground_truth_maps
export TUMOR_IMAGES_TO_EXTRACT=${DATA_DIR}/tumor_images_to_extract
export GRAYSCALE_HEATMAPS_PATH=${DATA_DIR}/grayscale_heatmaps
export THRESHOLDED_HEATMAPS_PATH=${DATA_DIR}/thresholded_heatmaps
export PATCH_FROM_HEATMAP_PATH=${DATA_DIR}/patches_from_heatmap
export THRESHOLD_LIST=${DATA_DIR}/threshold_list/threshold_list.txt
export CAE_TRAINING_DATA=${DATA_DIR}/training_data_cae
export CAE_TRAINING_DEVICE=0
export CAE_MODEL_PATH=${DATA_DIR}/models_cae
export LYM_CNN_TRAINING_DATA=${DATA_DIR}/training_data_cnn
export LYM_CNN_TRAINING_DEVICE=0
export LYM_CNN_PRED_DEVICE=0
export NEC_CNN_TRAINING_DATA=${DATA_DIR}/training_data_cnn
export NEC_CNN_TRAINING_DEVICE=1
export NEC_CNN_PRED_DEVICE=0

if [[ -z "${CUDA_VISIBLE_DEVICES}" ]]; then
	export LYM_CNN_TRAINING_DEVICE=0
	export LYM_CNN_PRED_DEVICE=0
else
	export LYM_CNN_TRAINING_DEVICE=${CUDA_VISIBLE_DEVICES}
	export LYM_CNN_PRED_DEVICE=${CUDA_VISIBLE_DEVICES}
fi

