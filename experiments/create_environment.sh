#!/bin/bash
conda create -y -n clrs_env python=3.11
conda activate clrs_env
conda install -y nvidia/label/cuda-12.3.0::cuda-toolkit
conda install -y nvidia/label/cuda-12.3.0::libcusparse
conda install -y -c nvidia cuda-nvcc
pip install -y -r requirements/requirements.txt
pip install -y jinja2
pip uninstall -y jax jaxlib
pip install -y -U "jax[cuda12_pip]==0.4.21" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
module avail nvidia-cuda-toolkit
export LD_LIBRARY_PATH=~/.conda/envs/clrs_env/lib:$LD_LIBRARY_PATH
export PATH=~/.conda/envs/clrs_env/bin:$PATH