#!/bin/bash

#SBATCH --job-name=register_llava
#SBATCH --output=./register_llava_output.log
#SBATCH --error=./register_llava_output.err
#SBATCH --partition=normal
#SBATCH --environment=tinyllava
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=460000
#SBATCH --time=01:00:00

source /iopsstor/scratch/cscs/tkerimog/tinyllama_env/bin/activate

export MIXTERA_SERVER_ADDR=$(hostname)
export MIXTERA_SERVER_DIR=/iopsstor/scratch/cscs/tkerimog/tinyllava/mixtera_server
export MIXTERA_SERVER_PORT=12345

export PYTHON_SCRIPT=/iopsstor/scratch/cscs/tkerimog/tinyllava/TinyLLaVA_Factory/mixtera_scripts/register_dataset.py

/iopsstor/scratch/cscs/tkerimog/tinyllama_env/bin/python -u -m mixtera.network.server.entrypoint \
    $MIXTERA_SERVER_DIR \
    --host $MIXTERA_SERVER_ADDR\
    --port $MIXTERA_SERVER_PORT &

sleep 5

/iopsstor/scratch/cscs/tkerimog/tinyllama_env/bin/python $PYTHON_SCRIPT

sleep 300