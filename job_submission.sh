echo hello world!

#login into the source
conda activate unet3D

#python /home/erattakulangara/hpchome/DeepLearningAlgo/2024-SSLUnetR/test.py > ./DeepLearningAlgo/MLFlow/script_output.log 2>&1
python ./DeepLearningAlgo/2024-SSLUnetR/MLFlowFineTune.py --max_iterations=50000 --dropout=0 --experiment_name=Prototyping
python ./DeepLearningAlgo/2024-SSLUnetR/MLFlowFineTune.py --max_iterations=50000 --dropout=0.5 --experiment_name=Prototyping