#!/bin/bash

source ../conf/variables.sh
# Generate meta and heatmap files for high-res and low-res heatmaps.

bash gen_all_json.sh &> ${LOG_OUTPUT_FOLDER}/log.gen_all_json.txt
cp ./json/* ${JSON_OUTPUT_FOLDER}/      #/data/heatmap_jsons

# Put all jsons to camicroscope
#bash upload_heatmaps.sh &> ${LOG_OUTPUT_FOLDER}/log.upload_heatmaps.txt

exit 0
