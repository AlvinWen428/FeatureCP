import torch
from tqdm import tqdm
import os
import numpy as np
np.warnings.filterwarnings('ignore')

from datasets import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import argparse

from conformal import helper
from conformal.icp import IcpRegressor, RegressorNc, FeatRegressorNc
from conformal.icp import AbsErrorErrFunc, FeatErrorErrFunc
from conformal.utils import compute_coverage, seed_torch


def makedirs(path):
    if not os.path.exists(path):
        print('creating dir: {}'.format(path))
        os.makedirs(path)
    else:
        print(path, "already exist!")


def main(x_train, y_train, x_test, y_test, idx_train, idx_cal, args):
    dir = f"ckpt/{args.data}"
    if os.path.exists(os.path.join(dir, "model.pt")) and not args.no_resume:
        model = helper.mse_model(in_shape=in_shape, out_shape=out_shape, hidden_size=args.hidden_size,
                                 dropout=args.dropout)
        print(f"==> Load model from {dir}")
        model.load_state_dict(torch.load(os.path.join(dir, "model.pt"), map_location=device))
    else:
        model = None

    mean_estimator = helper.MSENet_RegressorAdapter(model=model, device=device, fit_params=None,
                                                    in_shape=in_shape, out_shape=out_shape,
                                                    hidden_size=args.hidden_size, learn_func=nn_learn_func, epochs=args.epochs,
                                                    batch_size=args.batch_size, dropout=args.dropout, lr=args.lr, wd=args.wd,
                                                    test_ratio=cv_test_ratio, random_state=cv_random_state, )

    if float(args.feat_norm) <= 0 or args.feat_norm == "inf":
        args.feat_norm = "inf"
        print("Use inf as feature norm")
    else:
        args.feat_norm = float(args.feat_norm)

    nc = FeatRegressorNc(mean_estimator, inv_lr=args.feat_lr, inv_step=args.feat_step,
                         feat_norm=args.feat_norm, certification_method=args.cert_method)
    icp = IcpRegressor(nc)

    if os.path.exists(os.path.join(dir, "model.pt")) and not args.no_resume:
        pass
    else:
        icp.fit(x_train[idx_train, :], y_train[idx_train])
        makedirs(dir)
        print(f"==> Saving model at {dir}/model.pt")
        torch.save(mean_estimator.model.state_dict(), os.path.join(dir, "model.pt"))

    icp.calibrate(x_train[idx_cal, :], y_train[idx_cal])
    predictions = icp.predict(x_test, significance=alpha)

    y_lower, y_upper = predictions[..., 0], predictions[..., 1]
    # The estimated length on the output space by lirpa
    _, length_fcp = compute_coverage(y_test, y_lower, y_upper, alpha, name="FeatRegressorNc")
    # The coverage calculated on the feature space
    in_coverage = icp.if_in_coverage(x_test, y_test, significance=alpha)
    coverage_fcp = np.sum(in_coverage) * 100 / len(in_coverage)

    icp2 = IcpRegressor(RegressorNc(mean_estimator))
    icp2.calibrate(x_train[idx_cal, :], y_train[idx_cal])
    predictions = icp2.predict(x_test, significance=alpha)
    y_lower, y_upper = predictions[..., 0], predictions[..., 1]
    # The coverage and length of vanilla CP is calculated on the output space
    coverage_cp, length_cp = compute_coverage(y_test, y_lower, y_upper, alpha, name="RegressorNc")
    return coverage_fcp, length_fcp, coverage_cp, length_cp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", "--d", default=-1, type=int)
    parser.add_argument('--seed', type=int, nargs='+', default=[0])
    parser.add_argument("--data", type=str, default="community", help="meps20 fb1 fb2 blog")

    parser.add_argument("--alpha", type=float, default=0.1, help="miscoverage error")

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
    args = parser.parse_args()

    fcp_coverage_list, fcp_length_list, cp_coverage_list, cp_length_list = [], [], [], []
    for seed in tqdm(args.seed):
        seed_torch(seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = "{:}".format(args.device)
        device = torch.device("cpu") if args.device < 0 else torch.device("cuda")

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

        dataset_base_path = "./datasets/"
        dataset_name = args.data
        X, y = datasets.GetDataset(dataset_name, dataset_base_path)
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=seed)

        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)

        n_train = x_train.shape[0]
        in_shape = x_train.shape[1]
        out_shape = y_train.shape[1] if len(y_train.shape) > 1 else 1

        print("Dataset: %s" % (dataset_name))
        print("Dimensions: train set (n=%d, p=%d) ; test set (n=%d, p=%d)" %
              (x_train.shape[0], x_train.shape[1], x_test.shape[0], x_test.shape[1]))

        # divide the data into proper training set and calibration set
        idx = np.random.permutation(n_train)
        n_half = int(np.floor(n_train / 2))
        idx_train, idx_cal = idx[:n_half], idx[n_half:2 * n_half]

        # zero mean and unit variance scaling
        scalerX = StandardScaler()
        scalerX = scalerX.fit(x_train[idx_train])
        x_train = scalerX.transform(x_train)
        x_test = scalerX.transform(x_test)

        # scale the labels by dividing each by the mean absolute response
        mean_y_train = np.mean(np.abs(y_train[idx_train]))
        y_train = np.squeeze(y_train) / mean_y_train
        y_test = np.squeeze(y_test) / mean_y_train

        coverage_fcp, length_fcp, coverage_cp, length_cp = \
            main(x_train, y_train, x_test, y_test, idx_train, idx_cal, args)
        fcp_coverage_list.append(coverage_fcp)
        fcp_length_list.append(length_fcp)
        cp_coverage_list.append(coverage_cp)
        cp_length_list.append(length_cp)

    print(f'FeatureCP coverage: {np.mean(fcp_coverage_list)} \pm {np.std(fcp_coverage_list)}',
          f'FeatureCP estimated length: {np.mean(fcp_length_list)} \pm {np.std(fcp_length_list)}')
    print(f'VanillaCP coverage: {np.mean(cp_coverage_list)} \pm {np.std(cp_coverage_list)}',
          f'VanillaCP length: {np.mean(cp_length_list)} \pm {np.std(cp_length_list)}')
