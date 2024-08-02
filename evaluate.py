import torch
import argparse
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import pandas as pd
import numpy as np
from omegaconf import OmegaConf # yaml config for group calibration
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import random
import os

# Import dataloaders
import dataset.cifar10 as cifar10
import dataset.cifar100 as cifar100

# Import network architectures
from models.resnet import resnet50, resnet110
from models.densenet import densenet121
from models.resnet_imagenet import ResNet_ImageNet
from models.densenet_imagenet import DenseNet121_ImageNet
from models.wide_resnet_imagenet import Wide_ResNet_ImageNet
from models.mobilenet_v2_imagenet import MobileNet_V2_ImageNet

# Import metrics to compute
from metrics.metrics import test_classification_net_logits
from metrics.metrics import ECELoss, AdaptiveECELoss, ClasswiseECELoss

# Import post hoc calibration methods
from calibration.feature_clipping import FeatureClippingCalibrator
from calibration.pts_cts_ets import calibrator, calibrator_mapping, dataloader, dataset_mapping, loss_mapping, opt
from calibration.group_calibration.methods import calibrate



# Dataset params
dataset_num_classes = {
    'cifar10': 10,
    'cifar100': 100,
    'imagenet': 1000
}

dataset_loader = {
    'cifar10': cifar10,
    'cifar100': cifar100,
}

# Mapping model name to model function
cifar_models = {
    'resnet50': resnet50,
    'resnet110': resnet110,
    'densenet121': densenet121
}
imagenet_models = {
    'resnet50': ResNet_ImageNet(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1),
    'densenet121': DenseNet121_ImageNet(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1),
    'wide_resnet': Wide_ResNet_ImageNet(weights=torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V2),
    'mobilenet_v2': MobileNet_V2_ImageNet(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2),
    'vit_l_16':torchvision.models.vit_l_16(weights=torchvision.models.ViT_L_16_Weights.IMAGENET1K_V1),
}

def parseArgs():
    default_dataset = 'cifar10'
    dataset_root = '/datasets'
    num_bins = 15
    model_name = None
    train_batch_size = 128
    test_batch_size = 512
    cross_validation_error = 'ece'

    parser = argparse.ArgumentParser(
        description="Evaluating a single model on calibration metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default=default_dataset,
                        dest="dataset", help='dataset to test on')
    parser.add_argument("--dataset-root", type=str, default=dataset_root,
                        dest="dataset_root", help='root path of the dataset')
    parser.add_argument("--model-name", type=str, default=model_name,
                        dest="model_name", help='name of the model')
    parser.add_argument("--num-bins", type=int, default=num_bins, dest="num_bins",
                        help='Number of bins')
    parser.add_argument("-g", action="store_true", dest="gpu",
                        help="Use GPU")
    parser.set_defaults(gpu=True)
    parser.add_argument("-da", action="store_true", dest="data_aug",
                        help="Using data augmentation")
    parser.set_defaults(data_aug=True)
    parser.add_argument("-b", type=int, default=train_batch_size,
                        dest="train_batch_size", help="Batch size")
    parser.add_argument("-tb", type=int, default=test_batch_size,
                        dest="test_batch_size", help="Test Batch size")
    parser.add_argument("--feature_clamp", type=float, default=0.3)
    parser.add_argument("--cverror", type=str, default=cross_validation_error,
                        dest="cross_validation_error", help='Error function to do temp scaling')
    parser.add_argument("--debug", action="store_true", dest="debug",
                        help="whether to debug the code")
    parser.add_argument("--loss", type=str, default='cross_entropy')
    parser.add_argument("--save_loc", type=str, default='./pre_calculated_logits')
    parser.add_argument("--fc_type", type=str, default='fc')
    
    
    return parser.parse_args()


def get_logits_labels(data_loader, net, return_feature=False):
    logits_list = []
    labels_list = []
    features_list = []
    net.eval()
    if return_feature:
        with torch.no_grad():
            for data, label in data_loader:
                data = data.cuda()
                logits, features = net(data, return_feature=return_feature)
                logits_list.append(logits)
                labels_list.append(label)
                features_detach = features.detach().cpu()
                features_list.append(features_detach)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()
            features = torch.cat(features_list).cuda()
        return logits, labels, features
    else:
        with torch.no_grad():
            for data, label in data_loader:
                data = data.cuda()
                logits = net(data)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()
        return logits, labels


if __name__ == "__main__":

    # Checking if GPU is available
    cuda = False
    if (torch.cuda.is_available()):
        cuda = True

    # Setting additional parameters
    torch.manual_seed(1)
    device = torch.device("cuda" if cuda else "cpu")

    args = parseArgs()

    dataset = args.dataset
    dataset_root = args.dataset_root
    args.n_class = dataset_num_classes[dataset]
    model_name = args.model_name
    num_bins = args.num_bins
    cross_validation_error = args.cross_validation_error
    
    # define the calibration criterion
    nll_criterion = nn.CrossEntropyLoss().cuda()
    ece_criterion = ECELoss().cuda()
    adaece_criterion = AdaptiveECELoss().cuda()
    cece_criterion = ClasswiseECELoss().cuda()

    # load the datasets
    num_classes = dataset_num_classes[dataset]
    if (args.dataset == 'tiny_imagenet'):
        val_loader = dataset_loader[args.dataset].get_data_loader(
            root=args.dataset_root,
            split='val',
            batch_size=args.test_batch_size,
            pin_memory=args.gpu)

        test_loader = dataset_loader[args.dataset].get_data_loader(
            root=args.dataset_root,
            split='val',
            batch_size=args.test_batch_size,
            pin_memory=args.gpu)
    elif (args.dataset == 'cifar10' or args.dataset == 'cifar100'):
        _, val_loader = dataset_loader[args.dataset].get_train_valid_loader(
            batch_size=args.train_batch_size,
            augment=args.data_aug,
            random_seed=1,
            pin_memory=args.gpu
        )
        test_loader = dataset_loader[args.dataset].get_test_loader(
            batch_size=args.test_batch_size,
            pin_memory=args.gpu,
        )
    elif (args.dataset == 'imagenet'):
        # split 20% of the val set as validation set and the rest 80% as the test set
        # Define the transformation
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
        ])

        # Load the entire validation dataset
        full_val_set = torchvision.datasets.ImageFolder(
            root=args.dataset_root + '/imagenet/val',
            transform=transform,
        )

        # Calculate lengths for validation and test sets
        val_size = int(0.2 * len(full_val_set))
        test_size = len(full_val_set) - val_size

        # Split the dataset
        seed = 42
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        val_set, test_set = random_split(full_val_set, [val_size, test_size])

        # Create DataLoaders
        val_loader = DataLoader(val_set, batch_size=args.test_batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True)


    # Load the model
    if (args.dataset == 'cifar10' or args.dataset == 'cifar100'):
        model = cifar_models[model_name]
        net = model(num_classes=num_classes)
        net.cuda()
        net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
        net.load_state_dict(torch.load(os.path.join(args.save_loc, args.dataset, f"pretrained_weights/{args.model_name}_{args.loss}.model"), weights_only=True))
        net.classifier = net.module.classifier
    elif (args.dataset == 'imagenet'):
        model = imagenet_models[model_name]
        net = model.cuda()


    # if file not exist, calculated logits, feature and labels
    logit_path = f'pre_calculated_logits/{args.dataset}/{args.model_name}_{args.loss}.pt'
    if not os.path.exists(logit_path):
        logits_val, labels_val, features_val = get_logits_labels(val_loader, net, return_feature=True)
        logits_test, labels_test, features_test = get_logits_labels(test_loader, net, return_feature=True)

        torch.save({
            'logits_val': logits_val,
            'labels_val': labels_val,
            'features_val': features_val,
            'logits_test': logits_test,
            'labels_test': labels_test,
            'features_test': features_test,
        }, logit_path)

    # load logits, feature and labels
    data = torch.load(logit_path, weights_only=False)
    logits_val = data['logits_val']
    labels_val = data['labels_val']
    features_val = data['features_val']
    logits_test = data['logits_test']
    labels_test = data['labels_test']
    features_test = data['features_test']


    if args.debug:
        logits_val, labels_val, features_val = get_logits_labels(val_loader, net, return_feature=True)
        logits_test, labels_test, features_test = get_logits_labels(test_loader, net, return_feature=True)

    
    '''
    practice the feature clipping calibration
    '''
    fc_cal = FeatureClippingCalibrator(net, cross_validate=cross_validation_error)
    fc_cal.set_feature_clip(features_val, logits_val, labels_val)
    C_opt_fc = fc_cal.get_feature_clip()
    logits_val_fc, labels_val_fc, features_val_fc = fc_cal(features_val), labels_val, fc_cal.feature_clipping(features_val)
    logits_test_fc, labels_test_fc, features_test_fc = fc_cal(features_test), labels_test, fc_cal.feature_clipping(features_test)
    data['logits_val_fc'], data['labels_val_fc'], data['features_val_fc'] = logits_val_fc.detach(), labels_val_fc.detach(), features_val_fc.detach()
    data['logits_test_fc'], data['labels_test_fc'], data['features_test_fc'] = logits_test_fc.detach(), labels_test_fc.detach(), features_test_fc.detach()
    logits_val_fc = data['logits_val_fc']
    labels_val_fc = data['labels_val_fc']
    features_val_fc = data['features_val_fc']
    logits_test_fc = data['logits_test_fc']
    labels_test_fc = data['labels_test_fc']
    features_test_fc = data['features_test_fc']

    fc_logit_path = f'pre_calculated_logits/{args.dataset}/{args.model_name}_{args.loss}_fc.pt'
    torch.save(data, fc_logit_path)

    

    
    print("=="*20)
    print(args.model_name, args.dataset, args.loss)
    print("=="*20)

    # evalution results
    results = {
        "cal": [],
        "ece":[],
        "adaece":[],
        "cece":[],
        "nll":[],
        "accuracy":[]
    }
    run_methods = ['Vanilla', 'FC', 'TS', 'FC_TS', 'ETS', 'FC_ETS', 'PTS', 'FC_PTS', 'CTS', 'FC_CTS', 'GC', 'FC_GC']
    # vanilla
    if "Vanilla" in run_methods:
        ece = ece_criterion(logits_test, labels_test).item()
        adaece = adaece_criterion(logits_test, labels_test).item()
        cece = cece_criterion(logits_test, labels_test).item()
        nll = nll_criterion(logits_test, labels_test).item()
        accuracy = logits_test.argmax(dim=1).eq(labels_test).float().mean().item()
        print(f"Vanilla: ECE={round(ece*100, 2)}, Accuracy={round(accuracy*100, 2)}")
        results['cal'].append('Vanilla')
        results['ece'].append(ece)
        results['adaece'].append(adaece)
        results['cece'].append(cece)
        results['nll'].append(nll)
        results['accuracy'].append(accuracy)

    # FC
    if "FC" in run_methods:
        accuracy_fc = logits_test_fc.argmax(dim=1).eq(labels_test_fc).float().mean().item()
        ece_fc = ece_criterion(logits_test_fc, labels_test_fc).item()
        adaece_fc = adaece_criterion(logits_test_fc, labels_test_fc).item()
        cece_fc = cece_criterion(logits_test_fc, labels_test_fc).item()
        nll_fc = nll_criterion(logits_test_fc, labels_test_fc).item()
        print(f"FC(C={round(C_opt_fc,2)}): ECE={round(ece_fc*100, 2)}, Accuracy={round(accuracy_fc*100, 2)}")
        results['cal'].append(f'FC(C={round(C_opt_fc,2)})')
        results['ece'].append(ece_fc)
        results['adaece'].append(adaece_fc)
        results['cece'].append(cece_fc)
        results['nll'].append(nll_fc)
        results['accuracy'].append(accuracy_fc)

    # TS
    if "TS" in run_methods:
        args.cal = 'TS'
        cbt = calibrator(args).cuda()
        cbt.train(logits_val, labels_val)
        logits_ts = cbt(logits_test)
        ece_ts = ece_criterion(logits_ts, labels_test).item()
        adaece_ts = adaece_criterion(logits_ts, labels_test).item()
        cece_ts = cece_criterion(logits_ts, labels_test).item()
        nll_ts = nll_criterion(logits_ts, labels_test).item()
        accuracy_ts = logits_ts.argmax(dim=1).eq(labels_test).float().mean().item()
        print(f"TS: ECE={round(ece_ts*100, 2)}, Accuracy={round(accuracy_ts*100, 2)}")
        results['cal'].append('TS')
        results['ece'].append(ece_ts)
        results['adaece'].append(adaece_ts)
        results['cece'].append(cece_ts)
        results['nll'].append(nll_ts)
        results['accuracy'].append(accuracy_ts)

    # FC then TS
    if "FC_TS" in run_methods:
        args.cal = 'TS'
        cbt = calibrator(args).cuda()
        cbt.train(logits_val_fc, labels_val_fc)
        logits_fc_ts = cbt(logits_test_fc)
        ece_fc_ts = ece_criterion(logits_fc_ts, labels_test_fc).item()
        adaece_fc_ts = adaece_criterion(logits_fc_ts, labels_test_fc).item()
        cece_fc_ts = cece_criterion(logits_fc_ts, labels_test_fc).item()
        nll_fc_ts = nll_criterion(logits_fc_ts, labels_test_fc).item()
        accuracy_fc_ts = logits_fc_ts.argmax(dim=1).eq(labels_test_fc).float().mean().item()
        print(f"FC_TS: ECE={round(ece_fc_ts*100, 2)}, Accuracy={round(accuracy_fc_ts*100, 2)}")
        results['cal'].append('FC_TS')
        results['ece'].append(ece_fc_ts)
        results['adaece'].append(adaece_fc_ts)
        results['cece'].append(cece_fc_ts)
        results['nll'].append(nll_fc_ts)
        results['accuracy'].append(accuracy_fc_ts)


    # ETS
    if "ETS" in run_methods:
        args.cal = 'ETS'
        cbt = calibrator(args).cuda()
        cbt.train(logits_val, labels_val)
        logits_ets = cbt(logits_test)
        ece_ets = ece_criterion(logits_ets, labels_test).item()
        adaece_ets = adaece_criterion(logits_ets, labels_test).item()
        cece_ets = cece_criterion(logits_ets, labels_test).item()
        nll_ets = nll_criterion(logits_ets, labels_test).item()
        accuracy_ets = logits_ets.argmax(dim=1).eq(labels_test).float().mean().item()
        print(f"ETS: ECE={round(ece_ets*100, 2)}, Accuracy={round(accuracy_ets*100, 2)}")
        results['cal'].append('ETS')
        results['ece'].append(ece_ets)
        results['adaece'].append(adaece_ets)
        results['cece'].append(cece_ets)
        results['nll'].append(nll_ets)
        results['accuracy'].append(accuracy_ets)

    # FC then ETS
    if "FC_ETS" in run_methods:
        args.cal = 'ETS'
        cbt = calibrator(args).cuda()
        cbt.train(logits_val_fc, labels_val_fc)
        logits_fc_ets = cbt(logits_test_fc)
        ece_fc_ets = ece_criterion(logits_fc_ets, labels_test_fc).item()
        adaece_fc_ets = adaece_criterion(logits_fc_ets, labels_test_fc).item()
        cece_fc_ets = cece_criterion(logits_fc_ets, labels_test_fc).item()
        nll_fc_ets = nll_criterion(logits_fc_ets, labels_test_fc).item()
        accuracy_fc_ets = logits_fc_ets.argmax(dim=1).eq(labels_test_fc).float().mean().item()
        print(f"FC_ETS: ECE={round(ece_fc_ets*100, 2)}, Accuracy={round(accuracy_fc_ets*100, 2)}")
        results['cal'].append('FC_ETS')
        results['ece'].append(ece_fc_ets)
        results['adaece'].append(adaece_fc_ets)
        results['cece'].append(cece_fc_ets)
        results['nll'].append(nll_fc_ets)
        results['accuracy'].append(accuracy_fc_ets)


    # PTS
    if "PTS" in run_methods:
        args.cal = 'PTS'
        cbt = calibrator(args).cuda()
        cbt.train(logits_val, labels_val)
        logits_pts = cbt(logits_test)
        ece_pts = ece_criterion(logits_pts, labels_test).item()
        adaece_pts = adaece_criterion(logits_pts, labels_test).item()
        cece_pts = cece_criterion(logits_pts, labels_test).item()
        nll_pts = nll_criterion(logits_pts, labels_test).item()
        accuracy_pts = logits_pts.argmax(dim=1).eq(labels_test).float().mean().item()
        print(f"PTS: ECE={round(ece_pts*100, 2)}, Accuracy={round(accuracy_pts*100, 2)}")
        results['cal'].append('PTS')
        results['ece'].append(ece_pts)
        results['adaece'].append(adaece_pts)
        results['cece'].append(cece_pts)
        results['nll'].append(nll_pts)
        results['accuracy'].append(accuracy_pts)
    
    # FC then PTS
    if "FC_PTS" in run_methods:
        args.cal = 'PTS'
        cbt = calibrator(args).cuda()
        cbt.train(logits_val_fc, labels_val_fc)
        logits_fc_pts = cbt(logits_test_fc)
        ece_fc_pts = ece_criterion(logits_fc_pts, labels_test_fc).item()
        adaece_fc_pts = adaece_criterion(logits_fc_pts, labels_test_fc).item()
        cece_fc_pts = cece_criterion(logits_fc_pts, labels_test_fc).item()
        nll_fc_pts = nll_criterion(logits_fc_pts, labels_test_fc).item()
        accuracy_fc_pts = logits_fc_pts.argmax(dim=1).eq(labels_test_fc).float().mean().item()
        print(f"FC_PTS: ECE={round(ece_fc_pts*100, 2)}, Accuracy={round(accuracy_fc_pts*100, 2)}")
        results['cal'].append('FC_PTS')
        results['ece'].append(ece_fc_pts)
        results['adaece'].append(adaece_fc_pts)
        results['cece'].append(cece_fc_pts)
        results['nll'].append(nll_fc_pts)
        results['accuracy'].append(accuracy_fc_pts)


    # CTS
    if "CTS" in run_methods:
        args.cal = 'CTS'
        cbt = calibrator(args).cuda()
        cbt.train(logits_val, labels_val)
        logits_cts = cbt(logits_test)
        ece_cts = ece_criterion(logits_cts, labels_test).item()
        adaece_cts = adaece_criterion(logits_cts, labels_test).item()
        cece_cts = cece_criterion(logits_cts, labels_test).item()
        nll_cts = nll_criterion(logits_cts, labels_test).item()
        accuracy_cts = logits_cts.argmax(dim=1).eq(labels_test).float().mean().item()
        print(f"CTS: ECE={round(ece_cts*100, 2)}, Accuracy={round(accuracy_cts*100, 2)}")
        results['cal'].append('CTS')
        results['ece'].append(ece_cts)
        results['adaece'].append(adaece_cts)
        results['cece'].append(cece_cts)
        results['nll'].append(nll_cts)
        results['accuracy'].append(accuracy_cts)
    
    if "FC_CTS" in run_methods:
        # FC then CTS
        args.cal = 'PTS'
        cbt = calibrator(args).cuda()
        cbt.train(logits_val_fc, labels_val_fc)
        logits_fc_cts = cbt(logits_test_fc)
        ece_fc_cts = ece_criterion(logits_fc_cts, labels_test_fc).item()
        adaece_fc_cts = adaece_criterion(logits_fc_cts, labels_test_fc).item()
        cece_fc_cts = cece_criterion(logits_fc_cts, labels_test_fc).item()
        nll_fc_cts = nll_criterion(logits_fc_cts, labels_test_fc).item()
        accuracy_fc_cts = logits_fc_cts.argmax(dim=1).eq(labels_test_fc).float().mean().item()
        print(f"FC_CTS: ECE={round(ece_fc_cts*100, 2)}, Accuracy={round(accuracy_fc_cts*100, 2)}")
        results['cal'].append('FC_CTS')
        results['ece'].append(ece_fc_cts)
        results['adaece'].append(adaece_fc_cts)
        results['cece'].append(cece_fc_cts)
        results['nll'].append(nll_fc_cts)
        results['accuracy'].append(accuracy_fc_cts)


    if "GC" in run_methods:
        # Group Calibration 
        conf = OmegaConf.load("calibration/group_calibration/conf/method/group_calibration_combine_ets.yaml")
        logits_val, labels_val, features_val = logits_val.cpu(), labels_val.cpu(), features_val.cpu()
        logits_test, labels_test, features_test = logits_test.cpu(), labels_test.cpu(), features_test.cpu()
        calibrated_test_test = calibrate(method_config=conf,
                                            val_data={"logits": logits_val, "labels": labels_val, "features": features_val},
                                            test_train_data={"logits": logits_val, "labels": labels_val, "features": features_val},
                                            test_test_data={"logits": logits_test, "labels": labels_test, "features": features_test},
                                            seed=1,
                                            cfg=None)
        probs_gc = calibrated_test_test.get("prob", None)
        ece_gc = ece_criterion(logits=None, labels=labels_test, probs=probs_gc).item()
        adaece_gc = adaece_criterion(logits=None, labels=labels_test, probs=probs_gc).item()
        cece_gc = cece_criterion(logits=None, labels=labels_test, probs=probs_gc).item()
        nll_gc = torch.mean(-torch.log(probs_gc[range(len(labels_test)), labels_test])).item()
        accuracy_gc = torch.mean(torch.argmax(probs_gc, dim=1).eq(labels_test).float()).item()
        print(f"GC: ECE={round(ece_gc*100, 2)}, Accuracy={round(accuracy_gc*100, 2)}")
        results['cal'].append('GC')
        results['ece'].append(ece_gc)
        results['adaece'].append(adaece_gc)
        results['cece'].append(cece_gc)
        results['nll'].append(nll_gc)
        results['accuracy'].append(accuracy_gc)

    if "FC_GC" in run_methods:
        # FC then Group Calibration 
        conf = OmegaConf.load("calibration/group_calibration/conf/method/group_calibration_combine_ets.yaml")
        logits_test_fc, labels_test_fc, features_test_fc = logits_test_fc.cpu(), labels_test_fc.cpu(), features_test_fc.cpu()
        logits_test_fc, labels_test_fc, features_test_fc = logits_test_fc.cpu(), labels_test_fc.cpu(), features_test_fc.cpu()
        calibrated_test_test = calibrate(method_config=conf,
                                            val_data={"logits": logits_test_fc, "labels": labels_test_fc, "features": features_test_fc},
                                            test_train_data={"logits": logits_test_fc, "labels": labels_test_fc, "features": features_test_fc},
                                            test_test_data={"logits": logits_test_fc, "labels": labels_test_fc, "features": features_test_fc},
                                            seed=1,
                                            cfg=None)
        probs_fc_gc = calibrated_test_test.get("prob", None)
        ece_fc_gc = ece_criterion(logits=None, labels=labels_test_fc, probs=probs_fc_gc).item()
        adaece_fc_gc = adaece_criterion(logits=None, labels=labels_test_fc, probs=probs_fc_gc).item()
        cece_fc_gc = cece_criterion(logits=None, labels=labels_test_fc, probs=probs_fc_gc).item()
        nll_fc_gc = torch.mean(-torch.log(probs_fc_gc[range(len(labels_test_fc)), labels_test_fc])).item()
        accuracy_fc_gc = torch.mean(torch.argmax(probs_fc_gc, dim=1).eq(labels_test_fc).float()).item()
        print(f"FC_GC: ECE={round(ece_fc_gc*100, 2)}({round(C_opt_fc,2)}), Accuracy={round(accuracy_fc_gc*100, 2)},")
        results['cal'].append('FC_GC')
        results['ece'].append(ece_fc_gc)
        results['adaece'].append(adaece_fc_gc)
        results['cece'].append(cece_fc_gc)
        results['nll'].append(nll_fc_gc)
        results['accuracy'].append(accuracy_fc_gc)


    # print out a result table, drop the row index, decimal to 2
    results_table = pd.DataFrame({
        'Model': args.model_name,
        'Dataset': args.dataset,
        'ECE': [round(i*100, 2) for i in results['ece']],
        'AdaECE': [round(i*100, 2) for i in results['adaece']],
        'CECE': [round(i*100, 2) for i in results['cece']],
        'NLL': [round(i*100, 2) for i in results['nll']],
        'Accuracy': [round(i*100, 2) for i in results['accuracy']]
    }, index=results['cal'])
    print("\n",results_table,"\n")
    result_str = f"ECE Latex scipts: " \
    + f"{round(ece*100, 2):.2f}&\cellgray{round(ece_fc*100, 2):.2f}({round(C_opt_fc,2)}){' greendown' if ece_fc<ece else ' redup'}" \
    + f"&{round(ece_ts*100, 2):.2f}&\cellgray{round(ece_fc_ts*100, 2):.2f}{' greendown' if ece_fc_ts<ece_ts else ' redup'}" \
    + f"&{round(ece_ets*100, 2):.2f}&\cellgray{round(ece_fc_ets*100, 2):.2f}{' greendown' if ece_fc_ets<ece_ets else ' redup'}" \
    + f"&{round(ece_pts*100, 2):.2f}&\cellgray{round(ece_fc_pts*100, 2):.2f}{' greendown' if ece_fc_pts<ece_pts else ' redup'}" \
    + f"&{round(ece_cts*100, 2):.2f}&\cellgray{round(ece_fc_cts*100, 2):.2f}{' greendown' if ece_fc_cts<ece_cts else ' redup'}" \
    + f"&{round(ece_gc*100, 2):.2f}&\cellgray{round(ece_fc_gc*100, 2):.2f}{' greendown' if ece_fc_gc<ece_gc else ' redup'}" \
    + f"\\\\"
    # replace textcolor with \textcolor; replace blacktriangle with \blacktriangle
    result_str = result_str.replace('greendown', '\\greendown').replace('redup', '\\redup')
    # highlight the lowest results
    lowest_results = "{:.2f}".format(round(min(results['ece'])*100,2))
    result_str = result_str.replace(lowest_results, '\\textbf{'+lowest_results+'}')
    print(result_str)

    result_str = f"AdaECE Latex scipts: " \
    + f"{round(adaece*100, 2):.2f}&\cellgray{round(adaece_fc*100, 2):.2f}({round(C_opt_fc,2)}){' greendown' if adaece_fc<adaece else ' redup'}" \
    + f"&{round(adaece_ts*100, 2):.2f}&\cellgray{round(adaece_fc_ts*100, 2):.2f}{' greendown' if adaece_fc_ts<adaece_ts else ' redup'}" \
    + f"&{round(adaece_ets*100, 2):.2f}&\cellgray{round(adaece_fc_ets*100, 2):.2f}{' greendown' if adaece_fc_ets<adaece_ets else ' redup'}" \
    + f"&{round(adaece_pts*100, 2):.2f}&\cellgray{round(adaece_fc_pts*100, 2):.2f}{' greendown' if adaece_fc_pts<adaece_pts else ' redup'}" \
    + f"&{round(adaece_cts*100, 2):.2f}&\cellgray{round(adaece_fc_cts*100, 2):.2f}{' greendown' if adaece_fc_cts<adaece_cts else ' redup'}" \
    + f"&{round(adaece_gc*100, 2):.2f}&\cellgray{round(adaece_fc_gc*100, 2):.2f}{' greendown' if adaece_fc_gc<adaece_gc else ' redup'}" \
    + f"\\\\"
    # replace textcolor with \textcolor; replace blacktriangle with \blacktriangle
    result_str = result_str.replace('greendown', '\\greendown').replace('redup', '\\redup')
    # highlight the lowest results
    lowest_results = "{:.2f}".format(round(min(results['ece'])*100,2))
    result_str = result_str.replace(lowest_results, '\\textbf{'+lowest_results+'}')
    print(result_str)

