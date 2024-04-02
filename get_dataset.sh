#!/bin/bash

# For more information, visit: https://github.com/fastai/imagenette

dataset_name="imagenette2-320"
dataset_url="https://s3.amazonaws.com/fast-ai-imageclas/${dataset_name}.tgz"

mkdir -p data && cd data
curl -o "${dataset_name}.tgz" -# $dataset_url
tar -xzvf "${dataset_name}.tgz"
