# cifar10 fl linwei-lab2
CUDA_VISIBLE_DEVICES=1 python train.py --model resnet18 --loss focal_loss --dataset cifar10 --name c10_resnet18_fl & 
CUDA_VISIBLE_DEVICES=0 python train.py --model resnet34 --loss focal_loss --dataset cifar10 --name c10_resnet34_fl
CUDA_VISIBLE_DEVICES=1 python train.py --model resnet50 --loss focal_loss --dataset cifar10 --name c10_resnet50_fl &
CUDA_VISIBLE_DEVICES=0 python train.py --model resnet110 --loss focal_loss --dataset cifar10 --name c10_resnet110_fl
CUDA_VISIBLE_DEVICES=1 python train.py --model wide_resnet --loss focal_loss --dataset cifar10 --name c10_wide_resnet_fl &
CUDA_VISIBLE_DEVICES=0 python train.py --model densenet121 --loss focal_loss --dataset cifar10 --name c10_densenet121_fl

# cifar10 ce  xuchang-lab1
CUDA_VISIBLE_DEVICES=3 python train.py --model resnet18 --loss cross_entropy --dataset cifar10 --name c10_resnet18_ce & 
CUDA_VISIBLE_DEVICES=3 python train.py --model resnet34 --loss cross_entropy --dataset cifar10 --name c10_resnet34_ce &
CUDA_VISIBLE_DEVICES=2 python train.py --model resnet50 --loss cross_entropy --dataset cifar10 --name c10_resnet50_ce &
CUDA_VISIBLE_DEVICES=1 python train.py --model resnet110 --loss cross_entropy --dataset cifar10 --name c10_resnet110_ce &
CUDA_VISIBLE_DEVICES=0 python train.py --model wide_resnet --loss cross_entropy --dataset cifar10 --name c10_wide_resnet_ce &
CUDA_VISIBLE_DEVICES=0 python train.py --model densenet121 --loss cross_entropy --dataset cifar10 --name c10_densenet121_ce


# cifar10 mmce  xuchang-lab1
CUDA_VISIBLE_DEVICES=3 python train.py --model resnet18 --loss mmce --dataset cifar10 --name c10_resnet18_mmce & 
CUDA_VISIBLE_DEVICES=3 python train.py --model resnet34 --loss mmce --dataset cifar10 --name c10_resnet34_mmce  & 
CUDA_VISIBLE_DEVICES=2 python train.py --model resnet50 --loss mmce --dataset cifar10 --name c10_resnet50_mmce &
CUDA_VISIBLE_DEVICES=1 python train.py --model resnet110 --loss mmce --dataset cifar10 --name c10_resnet110_mmce & 
CUDA_VISIBLE_DEVICES=0 python train.py --model wide_resnet --loss mmce --dataset cifar10 --name c10_wide_resnet_mmce &
CUDA_VISIBLE_DEVICES=0 python train.py --model densenet121 --loss mmce --dataset cifar10 --name c10_densenet121_mmce

# cifar100 fl linwei-lab1
CUDA_VISIBLE_DEVICES=0 python train.py --model resnet18 --loss focal_loss --dataset cifar100 --name c100_resnet18_fl 
CUDA_VISIBLE_DEVICES=0 python train.py --model resnet34 --loss focal_loss --dataset cifar100 --name c100_resnet34_fl
CUDA_VISIBLE_DEVICES=0 python train.py --model resnet50 --loss focal_loss --dataset cifar100 --name c100_resnet50_fl 
CUDA_VISIBLE_DEVICES=0 python train.py --model resnet110 --loss focal_loss --dataset cifar100 --name c100_resnet110_fl
CUDA_VISIBLE_DEVICES=0 python train.py --model wide_resnet --loss focal_loss --dataset cifar100 --name c100_wide_resnet_fl 
CUDA_VISIBLE_DEVICES=0 python train.py --model densenet121 --loss focal_loss --dataset cifar100 --name c100_densenet121_fl

# cifar100 ce
CUDA_VISIBLE_DEVICES=0 python train.py --model resnet18 --loss cross_entropy --dataset cifar100 --name c100_resnet18_ce 
CUDA_VISIBLE_DEVICES=0 python train.py --model resnet34 --loss cross_entropy --dataset cifar100 --name c100_resnet34_ce
CUDA_VISIBLE_DEVICES=0 python train.py --model resnet50 --loss cross_entropy --dataset cifar100 --name c100_resnet50_ce 
CUDA_VISIBLE_DEVICES=0 python train.py --model resnet110 --loss cross_entropy --dataset cifar100 --name c100_resnet110_ce
CUDA_VISIBLE_DEVICES=0 python train.py --model wide_resnet --loss cross_entropy --dataset cifar100 --name c100_wide_resnet_ce 
CUDA_VISIBLE_DEVICES=0 python train.py --model densenet121 --loss cross_entropy --dataset cifar100 --name c100_densenet121_ce

# cifar100 mmce
CUDA_VISIBLE_DEVICES=0 python train.py --model resnet18 --loss mmce --dataset cifar100 --name c100_resnet18_mmce 
CUDA_VISIBLE_DEVICES=0 python train.py --model resnet34 --loss mmce --dataset cifar100 --name c100_resnet34_mmce
CUDA_VISIBLE_DEVICES=0 python train.py --model resnet50 --loss mmce --dataset cifar100 --name c100_resnet50_mmce 
CUDA_VISIBLE_DEVICES=0 python train.py --model resnet110 --loss mmce --dataset cifar100 --name c100_resnet110_mmce
CUDA_VISIBLE_DEVICES=0 python train.py --model wide_resnet --loss mmce --dataset cifar100 --name c100_wide_resnet_mmce 
CUDA_VISIBLE_DEVICES=0 python train.py --model densenet121 --loss mmce --dataset cifar100 --name c100_densenet121_mmce