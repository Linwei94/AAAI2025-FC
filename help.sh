CUDA_VISIBLE_DEVICES=1 python train.py --model resnet18 --loss focal_loss --dataset cifar100 --name c100_resnet18_fl & 
CUDA_VISIBLE_DEVICES=0 python train.py --model resnet34 --loss focal_loss --dataset cifar100 --name c100_resnet34_fl
CUDA_VISIBLE_DEVICES=1 python train.py --model resnet50 --loss focal_loss --dataset cifar100 --name c100_resnet50_fl &
CUDA_VISIBLE_DEVICES=0 python train.py --model resnet110 --loss focal_loss --dataset cifar100 --name c100_resnet110_fl
CUDA_VISIBLE_DEVICES=1 python train.py --model wide_resnet --loss focal_loss --dataset cifar100 --name c100_wide_resnet_fl &
CUDA_VISIBLE_DEVICES=0 python train.py --model densenet121 --loss focal_loss --dataset cifar100 --name c100_densenet121_fl