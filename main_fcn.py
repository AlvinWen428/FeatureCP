import torch
import torch.nn as nn
import random
import ipdb
import os
from tqdm import tqdm
from functools import partial
import numpy as np
np.warnings.filterwarnings('ignore')

import torch.utils.data as data
import torchvision.transforms as transforms
from torch import default_generator

from PIL import Image

from datasets import datasets
import datasets.transforms as ext_transforms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import argparse
import matplotlib.pyplot as plt

from conformal import helper
from conformal.fcn_model import make_fcn, FCN, enet_weighing
from conformal.icp import IcpRegressor, RegressorNc, FeatRegressorNc
from conformal.icp import AbsErrorErrFunc, FeatErrorErrFunc
from conformal.utils import compute_coverage, WeightedMSE, seed_torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def makedirs(path):
    if not os.path.exists(path):
        print('creating dir: {}'.format(path))
        os.makedirs(path)
    else:
        print(path, "already exist!")


def visualize(image, height, width, save_dir):
    fig, ax = plt.subplots()
    ax.imshow(image)
    plt.axis("off")
    fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(save_dir, pad_inches=0)
    plt.close()


def random_split(dataset, lengths, generator=default_generator, cal_test_generator=None):
    """
    Refactor the random_split function of pytorch.
    Randomly split a dataset into non-overlapping new datasets of given lengths.
    Optionally fix the generator for reproducible results, e.g.:

    >>> random_split(range(10), [3, 7], generator=torch.Generator().manual_seed(42))

    Arguments:
        dataset (Dataset): Dataset to be split
        lengths (sequence): lengths of splits to be produced
        generator (Generator): Generator used for the random permutation.
    """
    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):  # type: ignore
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = torch.randperm(sum(lengths), generator=generator)
    if cal_test_generator is None:
        indices = indices.tolist()
        return [data.Subset(dataset, indices[offset - length: offset]) for offset, length in zip(torch._utils._accumulate(lengths), lengths)]
    else:
        # only shuffle the calibration and test set
        assert len(lengths) == 4
        train_val_indices = indices[:sum(lengths[:2])]
        cal_test_indices = indices[-sum(lengths[-2:]):]
        assert len(train_val_indices) + len(cal_test_indices) == len(indices)
        cal_test_indices = cal_test_indices[torch.randperm(len(cal_test_indices), generator=cal_test_generator)]
        final_indices = train_val_indices.tolist() + cal_test_indices.tolist()
        assert len(final_indices) == len(indices)
        return [data.Subset(dataset, final_indices[offset - length: offset]) for offset, length in zip(torch._utils._accumulate(lengths), lengths)]


def load_dataset(dataset, seed):
    print("\nLoading dataset...\n")

    print("Selected dataset:", args.data)
    print("Dataset directory:", args.dataset_dir)

    image_transform = transforms.Compose(
        [transforms.Resize((args.height, args.width)),
         transforms.ToTensor()])

    label_transform = transforms.Compose([
        transforms.Resize((args.height, args.width), Image.NEAREST),
        ext_transforms.PILToLongTensor(),
        ext_transforms.ToOnehotGaussianBlur(kernel_size=7, num_classes=dataset.num_classes)
    ])

    whole_train_set = dataset(
        args.dataset_dir,
        transform=image_transform,
        label_transform=label_transform)

    if args.data_seed is None:
        cal_test_generator = torch.Generator().manual_seed(seed)
    else:
        cal_test_generator = torch.Generator().manual_seed(args.data_seed)
    train_set, val_set, cal_set, test_set = random_split(whole_train_set, [1800, 200, 600, len(whole_train_set) - 2600],
                                                         generator=torch.Generator().manual_seed(0),
                                                         cal_test_generator=cal_test_generator)

    train_loader = data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)
    cal_loader = data.DataLoader(
        cal_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    test_loader = data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    # Get encoding between pixel valus in label images and RGB colors
    class_encoding = whole_train_set.color_encoding

    # Get number of classes to predict
    num_classes = len(class_encoding)

    # Print information for debugging
    print("Number of classes to predict:", num_classes)
    print("Train dataset size:", len(train_set))
    print("Calibration dataset size:", len(cal_set))
    print("test dataset size:", len(test_set))

    # Get a batch of samples to display
    images, _, labels = iter(train_loader).next()
    print("Image size:", images.size())
    print("Label size:", labels.size())
    print("Class-color encoding:", class_encoding)

    print("Computing class weights...")
    print("(this can take a while depending on the dataset size)")
    class_weights = enet_weighing(train_loader, num_classes)
    if class_weights is not None:
        class_weights = torch.from_numpy(class_weights).float().to(device)
        # Set the weight of the unlabeled class to 0
        ignore_index = list(class_encoding).index('unlabeled')
        class_weights[ignore_index] = 0

    print("Class weights:", class_weights)

    return (train_loader, cal_loader, test_loader), class_weights, class_encoding


def main(train_loader, cal_loader, test_loader, args):
    dir = f"ckpt/{args.data}"
    if os.path.exists(os.path.join(dir, "model.pt")):
        model = make_fcn(pretrained=True, img_shape=image_shape, num_classes=20)
        print(f"==> Load model from {dir}")
        model.load_state_dict(torch.load(os.path.join(dir, "model.pt"), map_location=device)['state_dict'])
    else:
        raise ValueError("does not find the checkpoint in {}".format(dir))

    mean_estimator = helper.MSENet_RegressorAdapter(model=model, device=device, fit_params=None,
                                                    in_shape=image_shape, out_shape=args.height*args.width*num_classes,
                                                    hidden_size=args.hidden_size, learn_func=nn_learn_func, epochs=args.epochs,
                                                    batch_size=args.batch_size, dropout=args.dropout, lr=args.lr, wd=args.wd,
                                                    test_ratio=cv_test_ratio, random_state=cv_random_state, )

    if float(args.feat_norm) <= 0 or args.feat_norm == "inf":
        args.feat_norm = "inf"
        print("Use inf as feature norm")
    else:
        args.feat_norm = float(args.feat_norm)

    criterion = WeightedMSE(out_shape=(args.height, args.width), weight=class_weights)

    # FeatureCP
    nc = FeatRegressorNc(mean_estimator, criterion=criterion, inv_lr=args.feat_lr, inv_step=args.feat_step,
                         feat_norm=args.feat_norm, certification_method=args.cert_method,
                         g_out_process=partial(model.output_post_process, img_shape=image_shape))
    icp = IcpRegressor(nc)

    icp.calibrate_batch(cal_loader)

    # calculating the coverage of FCP
    in_coverage = icp.if_in_coverage_batch(test_loader, significance=alpha)
    coverage_fcp = np.sum(in_coverage) * 100 / len(in_coverage)

    test_intervals = []
    all_y_test = []
    img_idx = 0
    for x_test, discrete_y_test, y_test in tqdm(test_loader):
        intervals = icp.predict(x_test, significance=alpha)
        test_intervals.append(intervals)
        all_y_test.append(y_test.cpu().numpy())

        # this is used for visualization
        if args.visualize:
            loglog_interval = np.exp(-np.exp(intervals))
            show_interval = np.abs(loglog_interval[..., 1] - loglog_interval[..., 0])
            for img_interval in show_interval:
                img_interval = img_interval.reshape(19, args.height, args.width).mean(axis=0)
                img_name = os.path.basename(test_loader.dataset.dataset.train_data[test_loader.dataset.indices[img_idx]])
                visualize(img_interval, height=args.height, width=args.width, save_dir=os.path.join(f'visualization/seed{seed}', img_name))
                img_idx += 1

    test_intervals = np.concatenate(test_intervals, axis=0)
    all_y_test = np.concatenate(all_y_test, axis=0)
    # estimating the length of FCP
    y_lower, y_upper = test_intervals[..., 0], test_intervals[..., 1]
    _, length_fcp = compute_coverage(all_y_test, y_lower, y_upper, alpha, "FeatRegressorNc")

    # Vanilla CP
    icp2 = IcpRegressor(RegressorNc(mean_estimator))
    icp2.calibrate_batch(cal_loader)

    test_intervals = []
    all_y_test = []
    for x_test, _, y_test in tqdm(test_loader):
        intervals = icp2.predict(x_test, significance=alpha)
        test_intervals.append(intervals)
        all_y_test.append(y_test.cpu().numpy())
    test_intervals = np.concatenate(test_intervals, axis=0)
    all_y_test = np.concatenate(all_y_test, axis=0)

    y_lower, y_upper = test_intervals[..., 0], test_intervals[..., 1]
    coverage_cp, length_cp = compute_coverage(all_y_test, y_lower, y_upper, alpha, "RegressorNc")
    return coverage_fcp, length_fcp, coverage_cp, length_cp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "--d", default=-1, type=int)
    parser.add_argument('--seed', type=int, nargs='+', default=[0])
    parser.add_argument("--data", type=str, default="cityscapes", help="only support cityscapes now", choices=['cityscapes'])
    parser.add_argument("--data-seed", type=int, default=None, help="the random seed to split the calibration and test sets")
    parser.add_argument("--dataset-dir", type=str, default=None,
                        help="Path to the root directory of the selected dataset.")
    parser.add_argument("--workers", type=int, default=4, help="Number of subprocesses to use for data loading. Default: 4")

    parser.add_argument("--alpha", type=float, default=0.1, help="miscoverage error")

    parser.add_argument("--height", type=int, default=256, help="The image height. Default: 512")
    parser.add_argument("--width", type=int, default=512, help="The image width. Default: 1024")

    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", "--bs", type=int, default=64)
    parser.add_argument("--hidden_size", "--hs", type=int, default=64)
    parser.add_argument("--dropout", "--do", type=float, default=0.1)
    parser.add_argument("--wd", type=float, default=1e-6)
    parser.add_argument("--no-resume", action="store_true", default=False)

    parser.add_argument("--feat_opt", "--fo", type=str, default="sgd", choices=["sgd", "adam"])
    parser.add_argument("--feat_lr", "--fl", type=float, default=1e-2)
    parser.add_argument("--feat_step", "--fs", type=int, default=None)
    parser.add_argument("--feat_norm", "--fn", default=-1)
    parser.add_argument("--cert_method", "--cm", type=int, default=0, choices=[0, 1, 2, 3])

    parser.add_argument("--visualize", action="store_true", default=False, help="visualize the length in the image")
    args = parser.parse_args()

    fcp_coverage_list, fcp_length_list, cp_coverage_list, cp_length_list = [], [], [], []
    for seed in args.seed:
        seed_torch(seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = "{:}".format(args.device)
        device = torch.device("cpu") if args.device < 0 else torch.device("cuda")

        if args.visualize:
            makedirs(f"./visualization/seed{seed}")

        nn_learn_func = torch.optim.Adam

        # ratio of held-out data, used in cross-validation
        cv_test_ratio = 0.05
        # desired miscoverage error
        # alpha = 0.1
        alpha = args.alpha
        # used to determine the size of test set
        test_ratio = 0.2
        # seed for splitting the data in cross-validation.
        cv_random_state = 1

        if args.data.lower() == 'cityscapes':
            from datasets.cityscapes import Cityscapes as dataset
        else:
            # Should never happen...but just in case it does
            raise RuntimeError("\"{0}\" is not a supported dataset.".format(
                args.dataset))
        (train_loader, cal_loader, test_loader), class_weights, class_encoding = load_dataset(dataset, seed)
        # remove the first dimension "unlabeled"
        class_weights = class_weights[1:]

        image_shape = (args.height, args.width)
        n_train = len(train_loader.dataset)
        num_classes = len(class_encoding)

        print("Dataset: %s" % (args.data))

        coverage_fcp, length_fcp, coverage_cp, length_cp = \
            main(train_loader, cal_loader, test_loader, args)
        fcp_coverage_list.append(coverage_fcp)
        fcp_length_list.append(length_fcp)
        cp_coverage_list.append(coverage_cp)
        cp_length_list.append(length_cp)

    print(f'FeatureCP coverage: {np.mean(fcp_coverage_list)} \pm {np.std(fcp_coverage_list)}',
          f'FeatureCP estimated length: {np.mean(fcp_length_list)} \pm {np.std(fcp_length_list)}')
    print(f'VanillaCP coverage: {np.mean(cp_coverage_list)} \pm {np.std(cp_coverage_list)}',
          f'VanillaCP length: {np.mean(cp_length_list)} \pm {np.std(cp_length_list)}')
