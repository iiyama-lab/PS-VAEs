gpu=1
source=mnist
target=usps
name=${sourece}2${target}

python train.py --name ${name} --source ${souece} --target ${target} --gpu_ids ${gpu} --est_mnist --max_dataset_size 5000

python test.py --name ${name} --source ${souece} --target ${target} --gpu_ids ${gpu} --est_mnist --which_epoch 200
