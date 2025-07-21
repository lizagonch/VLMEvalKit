#!/bin/bash
# Example script to evaluate a PyramidDrop processed LLaVA 7B model

NPROC=${NPROC:-$(nvidia-smi --list-gpus | wc -l)}

torchrun --nproc-per-node=${NPROC} run.py --model llava_pdrop_v1.5_7b "$@"
