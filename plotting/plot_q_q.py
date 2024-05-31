import torch
from hwgpt.api import HWGPTBenchAPI
import pickle
import numpy as np
import matplotlib.pyplot as plt
from lib.utils import convert_str_to_arch
import random
import statsmodels.api as sm
import scipy.stats as stats
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.linestyle"] = "dotted"
plt.rcParams["font.size"] = 16
# plt tight layout
plt.rcParams["figure.autolayout"] = True


class HWQQPlot:
    def __init__(self, search_space: str):
        self.search_space = search_space
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.api = HWGPTBenchAPI(search_space=self.search_space)
        stats_path = (
            "data_collection/gpt_datasets/gpt_" + self.search_space + "/stats.pkl"
        )
        with open(stats_path, "rb") as f:
            self.stats = pickle.load(f)
        print("API initialized")
        self.sample_arch()

    def sample_arch(self):
        self.sampled_arch = random.choice(list(self.stats.keys()))
        self.sampled_config = convert_str_to_arch(self.sampled_arch)

    def get_hw_metrics(
        self, hw_metric: str, device: str, surrogate_type: str, data_type: str
    ):
        self.api.set_arch(self.sampled_config)
        return self.api.compute_predictions_hw(
            hw_metric=hw_metric,
            device=device,
            surrogate_type=surrogate_type,
            data_type=data_type,
            return_all_quantiles=True,
        )

    def get_actual_quantiles(self, hw_metric: str, device: str, quantiles: list):
        # print(list(self.stats.keys())[0])
        return np.quantile(self.stats[self.sampled_arch][device][hw_metric], quantiles)

    def plot_q_q(
        self, hw_metric: str, device: str, surrogate_type: str, data_type: str
    ):
        hw_quantiles = self.get_hw_metrics(hw_metric, device, surrogate_type, data_type)
        actual_quantiles = self.get_actual_quantiles(
            hw_metric, device, hw_quantiles.quantiles
        )*1000
        hw_quantiles = hw_quantiles.results_stacked[0]*1000
        print(hw_quantiles, actual_quantiles)
        plt.plot(hw_quantiles, actual_quantiles, marker="o", ls="")
        x = np.linspace(
            np.min((hw_quantiles.min(), actual_quantiles.min())),
            np.max((hw_quantiles.max(), actual_quantiles.max())),
        )
        plt.plot(x, x, linestyle="--", color="black")
        plt.xlabel("HW Quantiles")
        plt.ylabel("Actual Quantiles")
        plt.title(f"{hw_metric} Q-Q Plot")
        plt.savefig(f"{hw_metric}_{device}_{surrogate_type}_{data_type}_qq_plot.pdf")
        plt.clf()
        # q-q against gaussian
        print(np.mean(actual_quantiles))
        print(np.std(actual_quantiles))
        sm.qqplot(actual_quantiles,line="45",loc=np.mean(actual_quantiles),scale=np.std(actual_quantiles))
        plt.savefig(f"{hw_metric}_{device}_{surrogate_type}_{data_type}_qq_plot_gaussian.pdf")
        plt.clf()
        return hw_quantiles, actual_quantiles


if __name__ == "__main__":
    plot = HWQQPlot("s")
    hw_quantiles, actual_quantiles = plot.plot_q_q(
        "energies", "rtx3080", "conformal_quantile", "quantile"
    )
    hw_quantiles, actual_quantiles = plot.plot_q_q(
        "energies", "rtx2080", "conformal_quantile", "quantile"
    )
    print(hw_quantiles, actual_quantiles)
    hw_quantiles, actual_quantiles = plot.plot_q_q(
        "energies", "a40", "conformal_quantile", "quantile"
    )
    hw_quantiles, actual_quantiles = plot.plot_q_q(
        "energies", "a100", "conformal_quantile", "quantile"
    )
    hw_quantiles, actual_quantiles = plot.plot_q_q(
        "energies", "a40", "conformal_quantile", "quantile"
    )
    print(hw_quantiles, actual_quantiles)
    hw_quantiles, actual_quantiles = plot.plot_q_q(
        "energies", "P100", "conformal_quantile", "quantile"
    )
    hw_quantiles, actual_quantiles = plot.plot_q_q(
        "energies", "h100", "conformal_quantile", "quantile"
    )
    hw_quantiles, actual_quantiles = plot.plot_q_q(
        "energies", "a6000", "conformal_quantile", "quantile"
    )
    print(hw_quantiles, actual_quantiles)