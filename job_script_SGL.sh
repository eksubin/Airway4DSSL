#!/bin/bash
#$ -N FourDFine         # Name of the job
#$ -q SGL                   # Queue name
#$ -pe smp 16               # Number of slots (cores)
#$ -l ngpus=1                 # Number of GPUs
#$ -cwd                     # Use the current working directory
#$ -o ./logs/FinalSSL/output.log            # Output log file
#$ -e ./logs/FinalSSL/error.log             # Error log file

conda activate unet3D
#export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=1
#python MLFlowFineTune.py --max_iterations=50000 --dropout=0 --experiment_name=Prototyping --note="corrected the validation(0--1) issue"

#python MLFlowFineTune.py --max_iterations=50000 --dropout=0 --experiment_name=Prototyping
#python MLFlowFineTuneLatest.py --max_iterations=100000 --dropout=0.1 --experiment_name=Prototyping --note="0-1 Train and 0-1 Fine tuning" --pretrained_path="./mlruns/710251981863385259/fb2a1895681f4d8098735c9e1f04e3cf/artifacts/models/TrainingFrench_final/data/model.pth"
#python MLFlowFineTuneLatest.py --max_iterations=50000 --dropout=0.5 --experiment_name=Prototyping --note="0-1 Train and 0-1 Fine tuning" --pretrained_path="./mlruns/710251981863385259/fb2a1895681f4d8098735c9e1f04e3cf/artifacts/models/TrainingFrench_final/data/model.pth"
#python MLFlowFineTuneLatestStack.py --max_iterations=50000 --dropout=0.1 --experiment_name=Test --note="3DTime stack" --pretrained_path="./mlruns/910001340442692122/a0937dd7ad2e4151b5652de14c3e2953/artifacts/models/Test_final/data/model.pth"
#python MLFlowFineTune.py --max_iterations=100000 --dropout=0.5 --experiment_name=Prototyping --note="corrected the validation(0--1) issue"
#python SSLStackTrain.py --max_iterations=1000 --image_max=1 --image_min=0 --threshold=0.08 --model_name=Test --note=Parallel --dataset=01 


# testing the french speaker dataset
#python MLFlowFineTuneLatestStackFrenchSpeaker.py --train_dir="./Data/VoiceUsers/Train/userX/" --val_dir="./Data/VoiceUsers/Train/user2/" --max_iterations=10000 --dropout=0.1 --experiment_name=Test --note="3DTime single speaker" --use_pretrained --pretrained_path="./mlruns/910001340442692122/a0937dd7ad2e4151b5652de14c3e2953/artifacts/models/Test_final/data/model.pth"

#python FourDUnetRFineTune.py --train_dir="./Data/VoiceUsers/Train/users32/" --val_dir="./Data/VoiceUsers/Train/user2/" --max_iterations=10000 --dropout=0.1 --experiment_name="FourD" # --pretrained_path="./logs/FourDTrain/FourDTrain.pth"

#python SSLStackTrain.py --max_iterations=1000 --image_max=1 --image_min=0 --threshold=0.08 --model_name="FourDTrain" --note=Parallel --dataset=01 

#python SingleDUnetFineTune.py --train_dir="./Data/VoiceUsers/Train/users32/" --val_dir="./Data/VoiceUsers/Val/Nasal25/" --max_iterations=50000 --dropout=0.1 --lr=0.001 --experiment_name="FinalSSL" --pretrained_path="./logs/SingleDTrain/SingleDTrain_16patch.pth"
python SingleDUnetFineTune.py --train_dir="./Data/VoiceUsers/Train/Train/" --val_dir="./Data/VoiceUsers/Val/Nasal25/" --max_iterations=50000 --dropout=0.1 --lr=0.001 --experiment_name="FinalSSL" --pretrained_path="./logs/SingleDTrain/SingleDTrain_128_French.pth" --note="updatedDims"

#python MLFlowSingleDUnetFineTune.py --train_dir="./Data/VoiceUsers/Train/users32/" --val_dir="./Data/VoiceUsers/Train/Nasal25/" --max_iterations=10000 --experiment_name="HyperParam" --pretrained_path="./logs/SingleDTrain/SingleDTrain_16patch.pth"