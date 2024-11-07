conda activate feature-clipping

# cifar10
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar10 --model-name resnet50 --loss cross_entropy
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar10 --model-name resnet50 --loss brier_score
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar10 --model-name resnet50 --loss mmce
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar10 --model-name resnet50 --loss cross_entropy_smoothed_0.05
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar10 --model-name resnet50 --loss focal_loss_gamma_3.0
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar10 --model-name resnet50 --loss mbls