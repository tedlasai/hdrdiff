#!/bin/bash

# Environment variables
export PYTHONPATH="/data2/saikiran.tedla/hdrvideo/diff:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OPENCV_IO_ENABLE_OPENEXR=1

# Training command
accelerate launch examples/wanvideo/model_training/train.py --config /data2/saikiran.tedla/hdrvideo/diff/diffsynth/configs/firstweek.yaml