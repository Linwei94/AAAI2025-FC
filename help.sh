CUDA_VISIBLE_DEVICES=0 python train.py --model resnet18 --name resnet18_ce
CUDA_VISIBLE_DEVICES=0 python train.py --model resnet34 --name resnet34_ce
CUDA_VISIBLE_DEVICES=0 python train.py --model resnet50 --name resnet50_ce
CUDA_VISIBLE_DEVICES=0 python train.py --model resnet101 --name resnet101_ce
CUDA_VISIBLE_DEVICES=0 python train.py --model resnet110 --name resnet110_ce
CUDA_VISIBLE_DEVICES=0 python train.py --model wide_resnet --name wide_resnet_ce
CUDA_VISIBLE_DEVICES=0 python train.py --model densenet121 --name densenet121_ce