CUDA_VISIBLE_DEVICES=0 python train.py --model resnet34 --name resnet34_ce& 
CUDA_VISIBLE_DEVICES=2 python train.py --model resnet50 --name resnet50_ce&
CUDA_VISIBLE_DEVICES=3 python train.py --model resnet110 --name resnet110_ce&
CUDA_VISIBLE_DEVICES=1 python train.py --model wide_resnet --name wide_resnet_ce &
CUDA_VISIBLE_DEVICES=2 python train.py --model densenet121 --name densenet121_ce