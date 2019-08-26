#!/bin/bash

echo "Loading GCC"
module load gcc/4.9.3
echo "Sourcing Conda"
source /dls/science/groups/i04-1/conor_dev/anaconda/bin/activate env_pytorch
echo "Running trainscript"
python /dls/science/groups/i04-1/conor_dev/pandda_nn/frag_nn/pytorch/trainscript_cuda.py