gpu=1
source=syn
target=rea
name=pose_estimation

python train.py --name ${name} --source ${souece} --target ${target} --gpu_ids ${gpu} --est_mnist

python test.py --name ${name} --source ${souece} --target ${target} --gpu_ids ${gpu} --est_mnist --which_epoch 200
