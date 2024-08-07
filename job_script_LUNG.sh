#!/bin/bash
#$ -N FourDSeg         # Name of the job
#$ -q LUNG                   # Queue name
#$ -pe smp 16               # Number of slots (cores)
#$ -l gpu_a100=true         # Number of GPUs
#$ -cwd                     # Use the current working directory
#$ -o ./logs/outputLUNG.log            # Output log file
#$ -e ./logs/errorLUNG.log             # Error log file

conda activate unet3D
export CUDA_VISIBLE_DEVICES=0
#python MLFlowFineTune.py --max_iterations=50000 --dropout=0 --experiment_name=Prototyping --note="corrected the validation(0--1) issue"

#python MLFlowFineTune.py --max_iterations=50000 --dropout=0 --experiment_name=Prototyping
#python MLFlowFineTune.py --max_iterations=50000 --dropout=0.1 --experiment_name=Prototyping --note="replaced input image range from 0-255 to 0-1"
#python MLFlowFineTune.py --max_iterations=100000 --dropout=0.5 --experiment_name=Prototyping --note="corrected the validation(0--1) issue"
python MLFlowFineTuneLatestStack.py --max_iterations=20000 --dropout=0.1 --experiment_name=Test --note="3DTime stack" --pretrained_path="./mlruns/910001340442692122/a0937dd7ad2e4151b5652de14c3e2953/artifacts/models/Test_final/data/model.pth"
