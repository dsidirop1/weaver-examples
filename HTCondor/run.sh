#!/bin/bash

inputdir=./samples

# path to miniconda3 installation
condadir=./miniconda3

# path to the data config yaml and network config python file
# the example here uses this repo: https://github.com/jet-universe/particle_transformer
workdir=./top_tagging

# path to store the output model
modeldir=./output

# suffix for the job
suffix='particlenet'

# activate miniconda environment
. "${condadir}/etc/profile.d/conda.sh"
conda activate weaver

nvcc --version
nvidia-smi

# this is needed: setting `TMPDIR` to be the scratch area of the condor job
export TMPDIR=$(pwd)
set -x

#model=ParT
weaver \
    --data-train ${inputdir}/prep/top_train_*.root --copy-inputs \
    --data-val ${inputdir}/prep/top_val_*.root --copy-inputs \
    --data-config ${workdir}/data/pf_points_features.yaml \
    --network-config ${workdir}/networks/particlenet_pf.py --use-amp \
    --model-prefix ${modeldir}/${suffix} \
    --num-workers 1 --fetch-step 0.5 \
    --batch-size 1024 --start-lr 1e-3 --optimizer ranger --num-epochs 20 --gpus 0 \
    --log ${modeldir}/${suffix}.train.log
