#!/bin/bash
#$ -N your_job_name         # Name of the job
#$ -q LUNG                   # Queue name
#$ -pe smp 112               # Number of slots (cores)
#$ -l gpu=1                 # Number of GPUs
#$ -cwd                     # Use the current working directory
#$ -o output.log            # Output log file
#$ -e error.log             # Error log file

conda activate unet3D
export CUDA_VISIBLE_DEVICES=0
#python MLFlowFineTune.py --max_iterations=50000 --dropout=0 --experiment_name=Prototyping --note="corrected the validation(0--1) issue"

#python MLFlowFineTune.py --max_iterations=50000 --dropout=0 --experiment_name=Prototyping
python MLFlowFineTune.py --max_iterations=50000 --dropout=0.1 --experiment_name=Prototyping --note="replaced input image range from 0-255 to 0-1"
#python MLFlowFineTune.py --max_iterations=100000 --dropout=0.5 --experiment_name=Prototyping --note="corrected the validation(0--1) issue"