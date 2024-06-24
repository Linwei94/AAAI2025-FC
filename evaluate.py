import torch
import argparse
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import pandas as pd
import numpy as np
from omegaconf import OmegaConf # yaml config for group calibration

# Import dataloaders
import dataset.cifar10 as cifar10
import dataset.cifar100 as cifar100
import dataset.tiny_imagenet as tiny_imagenet
import dataset.tiny_imagenet as imagenet

# Import network architectures
from models.resnet_tiny_imagenet import resnet50 as resnet50_ti
from models.resnet import resnet50, resnet110
from models.wide_resnet import wide_resnet_cifar
from models.densenet import densenet121

# Import metrics to compute
from metrics.metrics import test_classification_net_logits
from metrics.metrics import ECELoss, AdaptiveECELoss, ClasswiseECELoss

# Import post hoc calibration methods
from calibration.temperature_scaling import ModelWithTemperature
from calibration.feature_clipping import ModelWithFeatureClipping
from calibration.pts_cts_ets import calibrator, calibrator_mapping, dataloader, dataset_mapping, loss_mapping, opt
from calibration.group_calibration.methods import calibrate



# Dataset params
dataset_num_classes = {
    'cifar10': 10,
    'cifar100': 100,
    'tiny_imagenet': 200,
    'imagenet': 1000
}

dataset_loader = {
    'cifar10': cifar10,
    'cifar100': cifar100,
    'tiny_imagenet': tiny_imagenet,
    'imagenet': imagenet
}

# Mapping model name to model function
models = {
    'resnet50': resnet50,
    'resnet50_ti': resnet50_ti,
    'resnet110': resnet110,
    'wide_resnet': wide_resnet_cifar,
    'densenet121': densenet121
}


def parseArgs():
    default_dataset = 'cifar10'
    dataset_root = './'
    model = 'resnet50'
    save_loc = './pretrained_weights/'
    saved_model_name = 'resnet50_focal_loss_gamma_3.0_350.model'
    # saved_model_name = 'resnet50_cross_entropy_350.model'
    num_bins = 15
    model_name = None
    train_batch_size = 128
    test_batch_size = 128
    cross_validation_error = 'ece'

    parser = argparse.ArgumentParser(
        description="Evaluating a single model on calibration metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default=default_dataset,
                        dest="dataset", help='dataset to test on')
    parser.add_argument("--dataset-root", type=str, default=dataset_root,
                        dest="dataset_root", help='root path of the dataset (for tiny imagenet)')
    parser.add_argument("--model-name", type=str, default=model_name,
                        dest="model_name", help='name of the model')
    parser.add_argument("--model", type=str, default=model, dest="model",
                        help='Model to test')
    parser.add_argument("--save-path", type=str, default=save_loc,
                        dest="save_loc",
                        help='Path to import the model')
    parser.add_argument("--saved_model_name", type=str, default=saved_model_name,
                        dest="saved_model_name", help="file name of the pre-trained model")
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
    parser.add_argument("-log", action="store_true", dest="log",
                        help="whether to print log data")
    
    parser.add_argument("--drop_index", type=int, default=None,
                        dest="drop_index", help="Index to drop from the dataset")
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
                features_list.append(features)
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

    if args.model_name is None:
        args.model_name = args.model

    dataset = args.dataset
    dataset_root = args.dataset_root
    args.n_class = dataset_num_classes[dataset]
    model_name = args.model_name
    save_loc = args.save_loc
    saved_model_name = args.saved_model_name
    num_bins = args.num_bins
    cross_validation_error = args.cross_validation_error

    # Taking input for the dataset
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
    else:
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


    # for c in list(range(26, 100, 1)):
    # args.feature_clamp = c/100
    drop_results = []
    args.feature_clamp = 100
    model = models[model_name]

    net = model(num_classes=num_classes, temp=1.0, feature_clamp=args.feature_clamp)
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
    net.load_state_dict(torch.load(args.save_loc + args.saved_model_name))
    net.classifier = net.module.classifier

    nll_criterion = nn.CrossEntropyLoss().cuda()
    ece_criterion = ECELoss().cuda()
    adaece_criterion = AdaptiveECELoss().cuda()
    cece_criterion = ClasswiseECELoss().cuda()

    # vanilla
    logits, labels = get_logits_labels(test_loader, net)
    conf_matrix, accuracy, _, _, _ = test_classification_net_logits(logits, labels)
    ece = ece_criterion(logits, labels).item()
    adaece = adaece_criterion(logits, labels).item()
    cece = cece_criterion(logits, labels).item()
    nll = nll_criterion(logits, labels).item()

    # TS
    model_ts = ModelWithTemperature(net, args.log)
    model_ts.set_temperature(val_loader, cross_validate=cross_validation_error)
    T_opt_ts = model_ts.get_temperature()
    logits_ts, labels_ts = get_logits_labels(test_loader, model_ts)
    conf_matrix_ts, accuracy_ts, _, _, _ = test_classification_net_logits(logits_ts, labels_ts)
    ece_ts = ece_criterion(logits_ts, labels_ts).item()
    adaece_ts = adaece_criterion(logits_ts, labels_ts).item()
    cece_ts = cece_criterion(logits_ts, labels_ts).item()
    nll_ts = nll_criterion(logits_ts, labels_ts).item()

    # FC
    model_fc = ModelWithFeatureClipping(net)
    model_fc.set_feature_clip(val_loader, cross_validate=cross_validation_error)
    C_opt_fc = model_fc.get_feature_clip()
    logits_fc, labels_fc = get_logits_labels(test_loader, model_fc)
    conf_matrix_fc, accuracy_fc, _, _, _ = test_classification_net_logits(logits_fc, labels_fc)
    ece_fc = ece_criterion(logits_fc, labels_fc).item()
    adaece_fc = adaece_criterion(logits_fc, labels_fc).item()
    cece_fc = cece_criterion(logits_fc, labels_fc).item()
    nll_fc = nll_criterion(logits_fc, labels_fc).item()

    # FC then TS
    model_fc_ts = ModelWithTemperature(model_fc, args.log)
    model_fc_ts.set_temperature(val_loader, cross_validate=cross_validation_error)
    T_opt_fc_ts = model_fc_ts.get_temperature()
    logits_fc_ts, labels_fc_ts = get_logits_labels(test_loader, model_fc_ts)
    conf_matrix_fc_ts, accuracy_fc_ts, _, _, _ = test_classification_net_logits(logits_fc_ts, labels_fc_ts)
    ece_fc_ts = ece_criterion(logits_fc_ts, labels_fc_ts).item()
    adaece_fc_ts = adaece_criterion(logits_fc_ts, labels_fc_ts).item()
    cece_fc_ts = cece_criterion(logits_fc_ts, labels_fc_ts).item()
    nll_fc_ts = nll_criterion(logits_fc_ts, labels_fc_ts).item()

    # ETS
    valid_logits, valid_labels = get_logits_labels(val_loader, net)
    test_logits, labels_ets = get_logits_labels(test_loader, net)
    args.cal = 'ETS'
    cbt = calibrator(args).cuda()
    cbt.train(valid_logits, valid_labels)
    logits_ets = cbt(test_logits)
    conf_matrix_ets, accuracy_ets, _, _, _ = test_classification_net_logits(logits_ets, labels_ets)
    ece_ets = ece_criterion(logits_ets, labels_ets).item()
    adaece_ets = adaece_criterion(logits_ets, labels_ets).item()
    cece_ets = cece_criterion(logits_ets, labels_ets).item()
    nll_ets = nll_criterion(logits_ets, labels_ets).item()

    # FC then ETS
    valid_logits, valid_labels = get_logits_labels(val_loader, model_fc)
    test_logits, labels_fc_ets = get_logits_labels(test_loader, model_fc)
    args.cal = 'ETS'
    cbt = calibrator(args).cuda()
    cbt.train(valid_logits, valid_labels)
    logits_fc_ets = cbt(test_logits)
    conf_matrix_fc_ets, accuracy_fc_ets, _, _, _ = test_classification_net_logits(logits_fc_ets, labels_fc_ets)
    ece_fc_ets = ece_criterion(logits_fc_ets, labels_fc_ets).item()
    adaece_fc_ets = adaece_criterion(logits_fc_ets, labels_fc_ets).item()
    cece_fc_ets = cece_criterion(logits_fc_ets, labels_fc_ets).item()
    nll_fc_ets = nll_criterion(logits_fc_ets, labels_fc_ets).item()

    # PTS
    valid_logits, valid_labels = get_logits_labels(val_loader, net)
    test_logits, labels_pts = get_logits_labels(test_loader, net)
    args.cal = 'PTS'
    cbt = calibrator(args).cuda()
    cbt.train(valid_logits, valid_labels)
    logits_pts = cbt(test_logits)
    conf_matrix_pts, accuracy_pts, _, _, _ = test_classification_net_logits(logits_pts, labels_pts)
    ece_pts = ece_criterion(logits_pts, labels_pts).item()
    adaece_pts = adaece_criterion(logits_pts, labels_pts).item()
    cece_pts = cece_criterion(logits_pts, labels_pts).item()
    nll_pts = nll_criterion(logits_pts, labels_pts).item()
    
    # FC then PTS
    valid_logits, valid_labels = get_logits_labels(val_loader, model_fc)
    test_logits, labels_fc_pts = get_logits_labels(test_loader, model_fc)
    args.cal = 'PTS'
    cbt = calibrator(args).cuda()
    cbt.train(valid_logits, valid_labels)
    logits_fc_pts = cbt(test_logits)
    conf_matrix_fc_pts, accuracy_fc_pts, _, _, _ = test_classification_net_logits(logits_fc_pts, labels_fc_pts)
    ece_fc_pts = ece_criterion(logits_fc_pts, labels_fc_pts).item()
    adaece_fc_pts = adaece_criterion(logits_fc_pts, labels_fc_pts).item()
    cece_fc_pts = cece_criterion(logits_fc_pts, labels_fc_pts).item()
    nll_fc_pts = nll_criterion(logits_fc_pts, labels_fc_pts).item()

    # CTS
    valid_logits, valid_labels = get_logits_labels(val_loader, net)
    test_logits, labels_cts = get_logits_labels(test_loader, net)
    args.cal = 'CTS'
    cbt = calibrator(args).cuda()
    cbt.train(valid_logits, valid_labels)
    logits_cts = cbt(test_logits)
    conf_matrix_cts, accuracy_cts, _, _, _ = test_classification_net_logits(logits_cts, labels_cts)
    ece_cts = ece_criterion(logits_cts, labels_cts).item()
    adaece_cts = adaece_criterion(logits_cts, labels_cts).item()
    cece_cts = cece_criterion(logits_cts, labels_cts).item()
    nll_cts = nll_criterion(logits_cts, labels_cts).item()
    
    # FC then CTS
    valid_logits, valid_labels = get_logits_labels(val_loader, model_fc)
    test_logits, labels_fc_cts = get_logits_labels(test_loader, model_fc)
    args.cal = 'PTS'
    cbt = calibrator(args).cuda()
    cbt.train(valid_logits, valid_labels)
    logits_fc_cts = cbt(test_logits)
    conf_matrix_fc_cts, accuracy_fc_cts, _, _, _ = test_classification_net_logits(logits_fc_cts, labels_fc_cts)
    ece_fc_cts = ece_criterion(logits_fc_cts, labels_fc_cts).item()
    adaece_fc_cts = adaece_criterion(logits_fc_cts, labels_fc_cts).item()
    cece_fc_cts = cece_criterion(logits_fc_cts, labels_fc_cts).item()
    nll_fc_cts = nll_criterion(logits_fc_cts, labels_fc_cts).item()

    # Group Calibration 
    conf = OmegaConf.load("calibration/group_calibration/conf/method/group_calibration_combine_ets.yaml")
    valid_logits, valid_labels, valid_features = get_logits_labels(val_loader, net, return_feature=True)
    test_logits, labels_gc, test_features = get_logits_labels(test_loader, net, return_feature=True)
    valid_logits, valid_labels, valid_features = valid_logits.cpu(), valid_labels.cpu(), valid_features.cpu()
    test_logits, labels_gc, test_features = test_logits.cpu(), labels_gc.cpu(), test_features.cpu()
    calibrated_test_test = calibrate(method_config=conf,
                                        val_data={"logits": valid_logits, "labels": valid_labels, "features": valid_features},
                                        test_train_data={"logits": valid_logits, "labels": valid_labels, "features": valid_features},
                                        test_test_data={"logits": test_logits, "labels": labels_gc, "features": test_features},
                                        seed=1,
                                        cfg=None)
    probs_gc = calibrated_test_test.get("prob", None)
    conf_matrix_gc, accuracy_gc, _, _, _ = test_classification_net_logits(logits=None, labels=labels_gc, probs=probs_gc)
    ece_gc = ece_criterion(logits=None, labels=labels_gc, probs=probs_gc).item()
    adaece_gc = adaece_criterion(logits=None, labels=labels_gc, probs=probs_gc).item()
    cece_gc = cece_criterion(logits=None, labels=labels_gc, probs=probs_gc).item()
    nll_gc = torch.mean(-torch.log(probs_gc[range(len(labels_gc)), labels_gc])).item()

    # FC then Group Calibration 
    conf = OmegaConf.load("calibration/group_calibration/conf/method/group_calibration_combine_ets.yaml")
    valid_logits, valid_labels, valid_features = get_logits_labels(val_loader, model_fc, return_feature=True)
    test_logits, labels_fc_gc, test_features = get_logits_labels(test_loader, model_fc, return_feature=True)
    valid_logits, valid_labels, valid_features = valid_logits.cpu(), valid_labels.cpu(), valid_features.cpu()
    test_logits, labels_fc_gc, test_features = test_logits.cpu(), labels_fc_gc.cpu(), test_features.cpu()
    calibrated_test_test = calibrate(method_config=conf,
                                        val_data={"logits": valid_logits, "labels": valid_labels, "features": valid_features},
                                        test_train_data={"logits": valid_logits, "labels": valid_labels, "features": valid_features},
                                        test_test_data={"logits": test_logits, "labels": labels_fc_gc, "features": test_features},
                                        seed=1,
                                        cfg=None)
    probs_fc_gc = calibrated_test_test.get("prob", None)
    conf_matrix_fc_gc, accuracy_fc_gc, _, _, _ = test_classification_net_logits(logits=None, labels=labels_fc_gc, probs=probs_fc_gc)
    ece_fc_gc = ece_criterion(logits=None, labels=labels_fc_gc, probs=probs_fc_gc).item()
    adaece_fc_gc = adaece_criterion(logits=None, labels=labels_fc_gc, probs=probs_fc_gc).item()
    cece_fc_gc = cece_criterion(logits=None, labels=labels_fc_gc, probs=probs_fc_gc).item()
    nll_fc_gc = torch.mean(-torch.log(probs_fc_gc[range(len(labels_fc_gc)), labels_fc_gc])).item()


    # print out a result table, drop the row index, decimal to 2
    results_table = pd.DataFrame({
        'Model': args.model_name,
        'Dataset': args.dataset,
        'ECE': [round(ece*100,2), round(ece_ts*100,2), round(ece_fc*100,2), round(ece_fc_ts*100,2), 
                round(ece_ets*100,2), round(ece_fc_ets*100,2), round(ece_pts*100,2), round(ece_fc_pts*100,2),
                round(ece_cts*100,2), round(ece_fc_cts*100,2), round(ece_gc*100,2), round(ece_fc_gc*100,2)],
        'AdaECE': [round(adaece*100,2), round(adaece_ts*100,2), round(adaece_fc*100,2), round(adaece_fc_ts*100,2), 
                   round(adaece_ets*100,2), round(adaece_fc_ets*100,2), round(adaece_pts*100,2), round(adaece_fc_pts*100,2),
                   round(adaece_cts*100,2), round(adaece_fc_cts*100,2), round(adaece_gc*100,2), round(adaece_fc_gc*100,2)],
        'CECE': [round(cece*100,2), round(cece_ts*100,2), round(cece_fc*100,2), round(cece_fc_ts*100,2), 
                 round(cece_ets*100,2), round(cece_fc_ets*100,2), round(cece_pts*100,2), round(cece_fc_pts*100,2),
                 round(cece_cts*100,2), round(cece_fc_cts*100,2), round(cece_gc*100,2), round(cece_fc_gc*100,2)],
        'NLL': [round(nll,2), round(nll_ts,2), round(nll_fc,2), round(nll_fc_ts,2), round(nll_ets,2), round(nll_fc_ets,2), 
                round(nll_pts,2), round(nll_fc_pts,2), round(nll_cts,2), round(nll_fc_cts,2), round(nll_gc,2), round(nll_fc_gc,2)],
        'Accuracy': [round(accuracy*100,2), round(accuracy_ts*100,2), round(accuracy_fc*100,2), round(accuracy_fc_ts*100,2),
                      round(accuracy_ets*100,2), round(accuracy_fc_ets*100,2), round(accuracy_pts*100,2), round(accuracy_fc_pts*100,2),
                        round(accuracy_cts*100,2), round(accuracy_fc_cts*100,2), round(accuracy_gc*100,2), round(accuracy_fc_gc*100,2)]
    }, index=['Vanilla Confidence', 
              f'TS(T={round(T_opt_ts,2)})', 
              f'FC(C={round(C_opt_fc,2)})', 
              f'FC_TS(T={round(T_opt_fc_ts,2)})',
                'ETS', f'FC_ETS',
                'PTS', f'FC_PTS',
                'CTS', f'FC_CTS',
                'GC', f'FC_GC'])
    print(results_table)
    print(f"Latex scipts: {round(ece*100, 2)}&{round(ece_fc*100, 2)}({round(C_opt_fc,2)})&{round(ece_ts*100, 2)}&{round(ece_fc_ts*100, 2)}&{round(ece_ets*100, 2)}&{round(ece_fc_ets*100, 2)}&{round(ece_pts*100, 2)}&{round(ece_fc_pts*100, 2)}&{round(ece_cts*100, 2)}&{round(ece_fc_cts*100, 2)}&{round(ece_gc*100, 2)}&{round(ece_fc_gc*100, 2)}\\\\")