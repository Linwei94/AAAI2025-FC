# cifar10
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar10 --model-name resnet50 --saved_model_name resnet50_c10_cross_entropy.model
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar10 --model-name resnet110 --saved_model_name resnet110_c10_cross_entropy.model
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar10 --model-name densenet121 --saved_model_name densenet121_c10_cross_entropy.model
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar10 --model-name wide_resnet --saved_model_name wide_resnet_c10_cross_entropy.model

# cifar100
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar100 --model-name resnet50 --saved_model_name resnet50_c100_cross_entropy.model
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar100 --model-name resnet110 --saved_model_name resnet110_c100_cross_entropy.model
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar100 --model-name densenet121 --saved_model_name densenet121_c100_cross_entropy.model
CUDA_VISIBLE_DEVICES=0 python evaluate.py --cverror nll --dataset cifar100 --model-name wide_resnet --saved_model_name wide_resnet_c100_cross_entropy.model