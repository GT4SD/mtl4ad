#!/bin/bash

NVCC_COMMAND=nvcc
if [ -n "$CUDA_HOME" ]; then
  NVCC_COMMAND=${CUDA_HOME}/bin/nvcc
fi

if ! command -v $NVCC_COMMAND &> /dev/null
then
    echo "nvcc not installed, so nothing to do"
    exit 1
fi

CUDA_VERSION=`${NVCC_COMMAND} --version | grep release | awk -F " " '{ print $5 }' | sed -e 's/[.,]//g'`
TORCH_VERSION=`pip3 list | grep torch | awk -F " " ' {print $2 }'`

if [[ ${TORCH_VERSION} != *"+"* ]];then
    echo "Torch is not installed with CUDA support"
    pip3 uninstall torch -y
    pip3 install torch==${TORCH_VERSION} \
        --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION}
fi