#!/bin/bash
source ../tensors/bin/activate && \
export CUDA_VISIBLE_DEVICES=0 && \
python3 model_main_tf2_gpu.py --model_dir=./models/ssd_mobilenet_v2 --pipeline_config_path=./models/ssd_mobilenet_v2/pipeline.config --checkpoint_dir=./models/ssd_mobilenet_v2 --alsologtostderr