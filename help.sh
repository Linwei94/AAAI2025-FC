CUDA_VISIBLE_DEVICES=1 python train.py --model resnet18 --loss mmce --dataset cifar100 --name c100_resnet18_mmce& 
CUDA_VISIBLE_DEVICES=0 python train.py --model resnet34 --loss mmce --dataset cifar100 --name c100_resnet34_mmce&
CUDA_VISIBLE_DEVICES=2 python train.py --model resnet50 --loss mmce --dataset cifar100 --name c100_resnet50_mmce&
CUDA_VISIBLE_DEVICES=3 python train.py --model resnet110 --loss mmce --dataset cifar100 --name c100_resnet110_mmce&
CUDA_VISIBLE_DEVICES=0 python train.py --model wide_resnet --loss mmce --dataset cifar100 --name c100_wide_resnet_mmce &
CUDA_VISIBLE_DEVICES=2 python train.py --model densenet121 --loss mmce --dataset cifar100 --name c100_densenet121_mmce&