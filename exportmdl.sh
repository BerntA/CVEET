#!/bin/bash
source ../tensors/bin/activate && \
python3 ./exporter_main_v2.py --input_type image_tensor --pipeline_config_path ./models/ssd_mobilenet_v2/pipeline.config --trained_checkpoint_dir ./models/ssd_mobilenet_v2 --output_directory ./exported-models/ssd_mobilenet_v2
