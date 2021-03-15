#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/bhome/berntae/cudnn/lib64:/usr/local/cuda-11.2/lib64 && \
source /bhome/berntae/tensors/env/bin/activate && \
export CUDA_VISIBLE_DEVICES=3 && \
python3 ./src/transfer-learning.py
