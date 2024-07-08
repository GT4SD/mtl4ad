#!/bin/bash

print_usage() {
    echo "Usage: $0 <config_file>"
}

config_file=$1
NUM_GPUS=${NUM_GPUS:-1}

if [ -z "$config_file" ]; then
    echo "Error: config_file parameter is required!"
    print_usage
    exit 1
fi

torchrun --nproc-per-node ${NUM_GPUS} /dccstor/yna/multi_modal/mtl4ad/src/mtl4ad/train.py --config "${config_file}"
