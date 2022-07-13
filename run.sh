
python main.py --arch resnet18_quan_8 --dataset cifar10 --target 0 --b-bits 10  --batch-size 128 --n-clean 128 --gamma 1000 --max-rho 100 --lr-grid 0.00001 --lr-noise 0.00001  --lr-weight 0.0001 --epsilon 0.04 --kappa 0.01 --save-dir ./save_tmp/ --gpu-id 0

