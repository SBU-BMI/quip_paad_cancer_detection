#!/bin/bash

source ../conf/variables.sh

cd tumor_pred
nohup bash pred_thread_lym.sh \
    ${PATCH_PATH} 0 2 ${LYM_CNN_PRED_DEVICE} \
    &> ${LOG_OUTPUT_FOLDER}/log.pred_thread_tumor_0.txt &

nohup bash pred_thread_lym.sh \
    ${PATCH_PATH} 1 2 ${LYM_CNN_PRED_DEVICE} \
    &> ${LOG_OUTPUT_FOLDER}/log.pred_thread_tumor_1.txt &
    
wait

exit 0
