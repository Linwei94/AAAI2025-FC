conda activate feature-clipping

# cifar10
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar10 --model-name resnet50 --loss cross_entropy
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar10 --model-name resnet50 --loss brier_score
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar10 --model-name resnet50 --loss mmce
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar10 --model-name resnet50 --loss label_smoothing 
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar10 --model-name resnet50 --loss focal_loss
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar10 --model-name resnet50 --loss focal_loss_53

CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar10 --model-name resnet110 --loss cross_entropy
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar10 --model-name resnet110 --loss brier_score
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar10 --model-name resnet110 --loss mmce
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar10 --model-name resnet110 --loss label_smoothing 
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar10 --model-name resnet110 --loss focal_loss
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar10 --model-name resnet110 --loss focal_loss_53

CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar10 --model-name densenet121 --loss cross_entropy
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar10 --model-name densenet121 --loss brier_score
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar10 --model-name densenet121 --loss mmce
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar10 --model-name densenet121 --loss label_smoothing 
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar10 --model-name densenet121 --loss focal_loss
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar10 --model-name densenet121 --loss focal_loss_53

CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar10 --model-name wide_resnet --loss cross_entropy
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar10 --model-name wide_resnet --loss brier_score
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar10 --model-name wide_resnet --loss mmce
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar10 --model-name wide_resnet --loss label_smoothing 
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar10 --model-name wide_resnet --loss focal_loss
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar10 --model-name wide_resnet --loss focal_loss_53





# # cifar100
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar100 --model-name resnet50 --loss cross_entropy
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar100 --model-name resnet50 --loss brier_score
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar100 --model-name resnet50 --loss mmce
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar100 --model-name resnet50 --loss label_smoothing 
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar100 --model-name resnet50 --loss focal_loss
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar100 --model-name resnet50 --loss focal_loss_53

CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar100 --model-name resnet110 --loss cross_entropy
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar100 --model-name resnet110 --loss brier_score
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar100 --model-name resnet110 --loss mmce
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar100 --model-name resnet110 --loss label_smoothing 
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar100 --model-name resnet110 --loss focal_loss
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar100 --model-name resnet110 --loss focal_loss_53

CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar100 --model-name densenet121 --loss cross_entropy
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar100 --model-name densenet121 --loss brier_score
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar100 --model-name densenet121 --loss mmce
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar100 --model-name densenet121 --loss label_smoothing 
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar100 --model-name densenet121 --loss focal_loss
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar100 --model-name densenet121 --loss focal_loss_53

CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar100 --model-name wide_resnet --loss cross_entropy
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar100 --model-name wide_resnet --loss brier_score
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar100 --model-name wide_resnet --loss mmce
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar100 --model-name wide_resnet --loss label_smoothing 
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar100 --model-name wide_resnet --loss focal_loss
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar100 --model-name wide_resnet --loss focal_loss_53