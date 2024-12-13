# Feature Clipping
### Pretrained models

All logits and features and extracted from the following models:

- CIFAR10 (from [focal loss calibration](https://github.com/torrvision/focal_calibration?tab=readme-ov-file))
  - [Resnet-50](https://www.robots.ox.ac.uk/~viveka/focal_calibration/CIFAR10/resnet50_cross_entropy_350.model)
  - [Resnet-110](https://www.robots.ox.ac.uk/~viveka/focal_calibration/CIFAR10/resnet110_cross_entropy_350.model)
  - [DenseNet-121](https://www.robots.ox.ac.uk/~viveka/focal_calibration/CIFAR10/densenet121_cross_entropy_350.model)
- CIFAR100 (from [focal loss calibration](https://github.com/torrvision/focal_calibration?tab=readme-ov-file))
  - [Resnet-50](https://www.robots.ox.ac.uk/~viveka/focal_calibration/CIFAR100/resnet50_cross_entropy_350.model)
  - [Resnet-110](https://www.robots.ox.ac.uk/~viveka/focal_calibration/CIFAR100/resnet110_cross_entropy_350.model)
  - [DenseNet-121](https://www.robots.ox.ac.uk/~viveka/focal_calibration/CIFAR100/densenet121_cross_entropy_350.model)
- IMAGENET (from [pytorch's torchvision.models](https://pytorch.org/vision/main/models.html))
  - Resnet-50: torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
  - DenseNet-121: torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1)
  - Wide-Resnet-50: torchvision.models.wide_resnet50_2(weights=torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
  - MobileNet-V2: torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
  - ViT-L-16: torchvision.models.vit_l_16(weights=torchvision.models.ViT_L_16_Weights.IMAGENET1K_V1)

### Dependencies
`conda create -n feature-clipping python=3.10`
`python -m pip install -r requirements.txt`

### Evalutation

run `bash evaluate_scripts_post_hoc.sh` to evaluate post hoc methods

run `bash evaluate_scripts_train-time.sh` to evaluate train time methods