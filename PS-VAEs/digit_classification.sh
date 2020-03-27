gpu=1
source=mnist_0
target=usps_50
name=${source}2${target}

python train.py --name ${name} --source ${source} --target ${target} --gpu_ids ${gpu} --est_mnist --max_dataset_size 5000

python test.py --name ${name} --source ${source} --target ${target} --gpu_ids ${gpu} --est_mnist --which_epoch 200
