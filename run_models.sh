#!/bin/bash

# set the output directory
output_dir="./outputs/"
# set the dataset directory
dataset_dir="./Dataset/"
# create the output directory
mkdir -p "${output_dir}"

for feature in {0..3}
do
  for model in {0..3}
  do
    container_name="feature-${feature}-model-${model}"
    docker run --name "${container_name}" -e "VAR1=${feature}" -e "VAR2=${model}" -v "${dataset_dir}:/train/dataset" THEIMAGE > "${output_dir}/${container_name}.txt" 2>&1 &
  done
done

wait