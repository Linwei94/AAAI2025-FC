# cifar10
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar10 --model-name resnet50
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar10 --model-name resnet110
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar10 --model-name densenet121
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar10 --model-name wide_resnet

# cifar100
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar100 --model-name resnet50
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar100 --model-name resnet110
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar100 --model-name densenet121
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar100 --model-name wide_resnet

# imagenet
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset imagenet --model-name resnet50
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset imagenet --model-name densenet121
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset imagenet --model-name wide_resnet
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset imagenet --model-name efficientnet_b0
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset imagenet --model-name mobilenet_v2
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset imagenet --model-name swin_b
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset imagenet --model-name vit_b_16
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset imagenet --model-name vit_l_16
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset imagenet --model-name vit_b_32
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset imagenet --model-name vit_l_32
