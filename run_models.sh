#!/bin/bash

for feature in {0..3}
do
  for model in {0..3}
  do
    container_name="feature-${feature}-model-${model}"
    output_dir="./outputs/"
    dataset_dir="./dataset/"
    mkdir -p "${output_dir}"
    docker run --name "${container_name}" -e "VAR1=${feature}" -e "VAR2=${model}" -v "${dataset_dir}:/train/dataset" THEIMAGE > "${output_dir}/${container_name}.txt" 2>&1 &
  done
done

wait