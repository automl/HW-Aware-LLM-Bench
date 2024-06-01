import torch
import pickle
import os
from hwgpt.predictors.hwmetric.net import Net
from lib.utils import (
    convert_str_to_arch,
    get_arch_feature_map,
    normalize_arch_feature_map,
    search_spaces,
)
import numpy as np
from hwgpt.predictors.hwmetric.deep_ensemble.ensemble import BaggingEnsemble
from hwgpt.predictors.hwmetric.conformal.surrogate.quantile_regression_model import (
    GradientBoostingQuantileRegressor,
)
from hwgpt.predictors.hwmetric.conformal.surrogate.symmetric_conformalized_quantile_regression_model import (
    SymmetricConformalizedGradientBoostingQuantileRegressor,
)
from typing import List, Tuple
from hwgpt.predictors.hwmetric.gaussian_mlp import GaussianNN
import argparse


def get_model_and_datasets(args: argparse.Namespace):
    if args.model == "gaussianmlp":
        train_dataset = HWDatasetMeanStd(
            mode="train",
            device_name=args.device,
            search_space=args.search_space,
            metric=args.metric,
            type=args.type,
        )
        test_dataset = HWDatasetMeanStd(
            mode="test",
            device_name=args.device,
            search_space=args.search_space,
            metric=args.metric,
            type=args.type,
        )
    else:
        train_dataset = HWDataset(
            mode="train",
            device_name=args.device,
            search_space=args.search_space,
            metric=args.metric,
            type=args.type,
            remove_outliesr=True
        )
        test_dataset = HWDataset(
            mode="test",
            device_name=args.device,
            search_space=args.search_space,
            metric=args.metric,
            type=args.type,
        )
    model = get_model(args)
    return model, train_dataset, test_dataset


def get_model(args: argparse.Namespace):
    if args.model == "conformal_quantile":
        model = SymmetricConformalizedGradientBoostingQuantileRegressor(
            quantiles=args.num_quantiles
        )
    elif args.model == "quantile":
        model = GradientBoostingQuantileRegressor(quantiles=args.num_quantiles)
    elif args.model == "mlp":
        search_space = search_spaces[args.search_space]
        max_layers = max(search_space["n_layer_choices"])
        model = Net(max_layers, False, 128, 128)
    elif args.model == "gaussianmlp":
        search_space = search_spaces[args.search_space]
        max_layers = max(search_space["n_layer_choices"])
        model = GaussianNN(max_layers)
    elif args.model == "ensemble":
        model = BaggingEnsemble(member_model_type="xgb", ensemble_size=10)
    else:
        raise ValueError("Model type not supported")
    return model


class HWDatasetMeanStd(torch.utils.data.Dataset):
    "Dataset to load the hardware metrics data for training and testing"

    def __init__(
        self,
        device_name: str = "a100",
        search_space: str = "s",
        metric: str = "latencies",
        type: str = "median",
        mode: str = "train",
    ):
        "Initialization"
        self.device_name = device_name
        self.search_space = search_space
        self.transform = False
        self.metric = metric
        self.type = type
        self.mode = mode
        self.archs_to_one_hot = []
        self.metric_obs_mean = []
        self.metric_obs_std = []
        arch_stats_path = (
            "data_collection/gpt_datasets/gpt_" + str(self.search_space) + "/stats.pkl"
        )
        with open(arch_stats_path, "rb") as f:
            self.arch_stats = pickle.load(f)
        self.archs_all = list(self.arch_stats.keys())
        self.archs_train = self.archs_all[0:8000]
        self.archs_test = self.archs_all[8000:]
        self.load_data()

    def process_arch_device(
        self, arch: str, metric_mean: List, metric_std: List, arch_features: List
    ) -> Tuple[List, List]:
        arch_config = convert_str_to_arch(arch)
        feature = get_arch_feature_map(arch_config, self.search_space)
        feature = normalize_arch_feature_map(feature, self.search_space)
        if self.metric == "latencies" or self.metric == "energies":
            metric_mean.append(
                self.arch_stats[arch][self.device_name][self.metric + "_mean"]
            )
            metric_std.append(
                self.arch_stats[arch][self.device_name][self.metric + "_std"]
            )
            arch_features.append(feature)
        else:
            raise ValueError("Invalid metric")

        return metric_mean, metric_std, arch_features

    def load_data(self):
        arch_features_train = []
        arch_features_test = []
        metric_train_mean = []
        metric_train_std = []
        metric_test_mean = []
        metric_test_std = []
        for arch in self.archs_train:
            metric_train_mean, metric_train_std, arch_features_train = (
                self.process_arch_device(
                    arch, metric_train_mean, metric_train_std, arch_features_train
                )
            )
        for arch in self.archs_test:
            metric_test_mean, metric_test_std, arch_features_test = (
                self.process_arch_device(
                    arch, metric_test_mean, metric_test_std, arch_features_test
                )
            )
        self.arch_features_train = torch.tensor(arch_features_train)
        self.latencies_train_mean = torch.tensor(metric_train_mean)
        self.latencies_train_std = torch.tensor(metric_train_std)
        self.arch_features_test = torch.tensor(arch_features_test)
        self.latencies_test_mean = torch.tensor(metric_test_mean)
        self.latencies_test_std = torch.tensor(metric_test_std)

    def __len__(self):
        "Denotes the total number of samples"
        if self.mode == "train":
            return self.arch_features_train.shape[0]
        else:
            return self.arch_features_test.shape[0]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        "Generates one sample of data"
        # Select sample
        if self.mode == "train":
            one_hot = self.arch_features_train[idx]
            metric_mean = self.latencies_train_mean[idx]
            metric_std = self.latencies_train_std[idx]
        else:
            one_hot = self.arch_features_test[idx]
            metric_mean = self.latencies_test_mean[idx]
            metric_std = self.latencies_test_std[idx]
        return one_hot, metric_mean, metric_std


class HWDatasetStratified(torch.utils.data.Dataset):
    "Dataset to load the hardware metrics data for training and testing"

    def __init__(
        self,
        device_name: str = "a100",
        search_space: str = "s",
        metric: str = "latencies",
        type: str = "median",
        mode: str = "train",
        remove_outliesr: bool = False,
    ):
        "Initialization"
        self.device_name = device_name
        self.search_space = search_space
        self.remove_outliers = remove_outliesr
        self.transform = False
        self.metric = metric
        self.type = type
        self.mode = mode
        self.archs_to_one_hot = []
        self.metric_obs = []
        arch_stats_path = (
            "data_collection/gpt_datasets/gpt_" + str(self.search_space) + "/stats.pkl"
        )
        with open(arch_stats_path, "rb") as f:
            self.arch_stats = pickle.load(f)
        self.archs_all = list(self.arch_stats.keys())
        self.archs_train = self.archs_all[0:8000]
        self.archs_test = self.archs_all[8000:]
        self.load_data()

    def remove_outliers_iqr(self, metric_list: List):
        # Convert data to a numpy array for convenience
        data = np.array(metric_list)

        # Calculate Q1 and Q3
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)

        # Calculate IQR
        IQR = Q3 - Q1

        # Determine the lower and upper bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter out outliers
        filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]

        return filtered_data.tolist()

    def remove_outliers_z_score(self, metric_list: List, threshold: float = 3):
        mean = np.mean(metric_list)
        std = np.std(metric_list)
        z_scores = [(x - mean) / std for x in metric_list]
        filtered_data = [x for x, z in zip(metric_list, z_scores) if abs(z) < threshold]
        return filtered_data

    def process_arch_device(
        self, arch: str, metric: dict, arch_features: dict
    ) -> Tuple[List, List]:
        arch_config = convert_str_to_arch(arch)
        feature = get_arch_feature_map(arch_config, self.search_space)
        feature = normalize_arch_feature_map(feature, self.search_space)
        if self.metric == "latencies" or self.metric == "energies":
            if self.type == "median":
                metric.append(
                    np.median(self.arch_stats[arch][self.device_name][self.metric])
                )
                arch_features.append(feature)
            else:
                if "cpu" in self.device_name and self.metric == "energies":
                    latencies_arch = [
                        self.arch_stats[arch][self.device_name][self.metric]
                    ]
                    # latencies_arch = self.remove_outliers_z_score(latencies_arch)
                else:
                    latencies_arch = self.arch_stats[arch][self.device_name][
                        self.metric
                    ]
                    if self.remove_outliers:
                        latencies_arch = self.remove_outliers_z_score(latencies_arch)
                latencies_arch = list(latencies_arch)
                for i,lat in enumerate(latencies_arch):
                    if i not in metric:
                        metric[i] = []
                        arch_features[i] = []
                    metric[i].append(lat)
                    arch_features[i].append(feature)
        elif "memory" in self.metric:
            metric.append(self.arch_stats[arch][self.metric])
            arch_features.append(feature)
        elif self.metric == "flops":
            metric.append(self.arch_stats[arch][self.metric] / 10**12)
            arch_features.append(feature)
        elif self.metric == "params":
            metric.append(self.arch_stats[arch][self.metric] / 10**6)
            arch_features.append(feature)
        else:
            raise ValueError("Invalid metric")

        return metric, arch_features

    def load_data(self):
        arch_features_train = {}
        arch_features_test = {}
        metric_train = {}
        metric_test = {}
        for arch in self.archs_train:
            metric_train, arch_features_train = self.process_arch_device(
                arch, metric_train, arch_features_train
            )
        for arch in self.archs_test:
            metric_test, arch_features_test = self.process_arch_device(
                arch, metric_test, arch_features_test
            )
        self.arch_features_train = torch.tensor(arch_features_train)
        self.latencies_train = torch.tensor(metric_train)
        self.arch_features_test = torch.tensor(arch_features_test)
        self.latencies_test = torch.tensor(metric_test)

    def __len__(self):
        "Denotes the total number of samples"
        if self.mode == "train":
            return self.arch_features_train.shape[0]
        else:
            return self.arch_features_test.shape[0]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        "Generates one sample of data"
        # Select sample
        if self.mode == "train":
            one_hot = self.arch_features_train[idx]
            metric = self.latencies_train[idx]
        else:
            one_hot = self.arch_features_test[idx]
            metric = self.latencies_test[idx]
        return one_hot, metric
    
class HWDataset(torch.utils.data.Dataset):
    "Dataset to load the hardware metrics data for training and testing"

    def __init__(
        self,
        device_name: str = "a100",
        search_space: str = "s",
        metric: str = "latencies",
        type: str = "median",
        mode: str = "train",
        remove_outliesr: bool = False,
    ):
        "Initialization"
        self.device_name = device_name
        self.search_space = search_space
        self.remove_outliers = remove_outliesr
        self.transform = False
        self.metric = metric
        self.type = type
        self.mode = mode
        self.archs_to_one_hot = []
        self.metric_obs = []
        arch_stats_path = (
            "data_collection/gpt_datasets/gpt_" + str(self.search_space) + "/stats.pkl"
        )
        with open(arch_stats_path, "rb") as f:
            self.arch_stats = pickle.load(f)
        self.archs_all = list(self.arch_stats.keys())
        self.archs_train = self.archs_all[0:8000]
        self.archs_test = self.archs_all[8000:]
        self.load_data()

    def remove_outliers_iqr(self, metric_list: List):
        # Convert data to a numpy array for convenience
        data = np.array(metric_list)

        # Calculate Q1 and Q3
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)

        # Calculate IQR
        IQR = Q3 - Q1

        # Determine the lower and upper bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter out outliers
        filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]

        return filtered_data.tolist()

    def remove_outliers_z_score(self, metric_list: List, threshold: float = 3):
        mean = np.mean(metric_list)
        std = np.std(metric_list)
        z_scores = [(x - mean) / std for x in metric_list]
        filtered_data = [x for x, z in zip(metric_list, z_scores) if abs(z) < threshold]
        return filtered_data

    def process_arch_device(
        self, arch: str, metric: str, arch_features: List
    ) -> Tuple[List, List]:
        arch_config = convert_str_to_arch(arch)
        feature = get_arch_feature_map(arch_config, self.search_space)
        feature = normalize_arch_feature_map(feature, self.search_space)
        if self.metric == "latencies" or self.metric == "energies":
            if self.type == "median":
                metric.append(
                    np.median(self.arch_stats[arch][self.device_name][self.metric])
                )
                arch_features.append(feature)
            else:
                if "cpu" in self.device_name and self.metric == "energies":
                    latencies_arch = [
                        self.arch_stats[arch][self.device_name][self.metric]
                    ]
                    # latencies_arch = self.remove_outliers_z_score(latencies_arch)
                else:
                    latencies_arch = self.arch_stats[arch][self.device_name][
                        self.metric
                    ]
                    if self.remove_outliers:
                        latencies_arch = self.remove_outliers_z_score(latencies_arch)
                latencies_arch = list(latencies_arch)
                for lat in latencies_arch:
                    metric.append(lat)
                    arch_features.append(feature)
        elif "memory" in self.metric:
            metric.append(self.arch_stats[arch][self.metric])
            arch_features.append(feature)
        elif self.metric == "flops":
            metric.append(self.arch_stats[arch][self.metric] / 10**12)
            arch_features.append(feature)
        elif self.metric == "params":
            metric.append(self.arch_stats[arch][self.metric] / 10**6)
            arch_features.append(feature)
        else:
            raise ValueError("Invalid metric")

        return metric, arch_features

    def load_data(self):
        arch_features_train = []
        arch_features_test = []
        metric_train = []
        metric_test = []
        for arch in self.archs_train:
            metric_train, arch_features_train = self.process_arch_device(
                arch, metric_train, arch_features_train
            )
        for arch in self.archs_test:
            metric_test, arch_features_test = self.process_arch_device(
                arch, metric_test, arch_features_test
            )
        self.arch_features_train = torch.tensor(arch_features_train)
        self.latencies_train = torch.tensor(metric_train)
        self.arch_features_test = torch.tensor(arch_features_test)
        self.latencies_test = torch.tensor(metric_test)

    def __len__(self):
        "Denotes the total number of samples"
        if self.mode == "train":
            return self.arch_features_train.shape[0]
        else:
            return self.arch_features_test.shape[0]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        "Generates one sample of data"
        # Select sample
        if self.mode == "train":
            one_hot = self.arch_features_train[idx]
            metric = self.latencies_train[idx]
        else:
            one_hot = self.arch_features_test[idx]
            metric = self.latencies_test[idx]
        return one_hot, metric


if __name__ == "__main__":
    devices_all = [
        "a100",
        "a40",
        "h100",
        "rtx2080",
        "rtx3080",
        "a6000",
        "v100",
        "P100",
        "cpu_xeon_silver",
        "cpu_xeon_gold",
        "cpu_amd_7502",
        "cpu_amd_7513",
        "cpu_amd_7452",
    ]
    models = ["s", "m", "l"]
    metrics = [
        "energies",
        "latencies",
        "float16_memory",
        "bfloat16_memory",
        "flops",
        "params",
    ]
    type = "quantile"
    for device in devices_all:
        for model in models:
            for metric in metrics:
                dset = HWDataset(
                    mode="train",
                    device_name=device,
                    search_space=model,
                    metric=metric,
                    type="quantile",
                )
                print(len(dset.arch_features_train))
                print(len(dset.arch_features_test))
                print(device)
                print(model)
                assert len(dset.latencies_test) == len(dset.arch_features_test)
                assert len(dset.latencies_train) == len(dset.latencies_train)
                # if metric == "energies" and "cpu" not in device:
                #    assert len(dset.latencies_train) == 400000
                #    assert len(dset.latencies_test) == 100000
                # elif metric == "latencies":
                #    assert len(dset.latencies_train) == 80000
                #    assert len(dset.latencies_test) == 20000
                # else:
                #    assert len(dset.latencies_train) == 8000
                #    assert len(dset.latencies_test) == 2000
