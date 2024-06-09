#! /bin/bash

# setup conda in the slurm shell env
eval "$(conda shell.bash hook)"

# activate the python environment
conda activate ai-for-se-grp-c

echo "Starting"

# execute the script
python3 BigVulMain.py
