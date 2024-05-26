from predictors.metric.net import Net
from hwgpt.model.gpt.utils import sample_config_max, sample_config_min
from lib.utils import search_spaces, convert_config_to_one_hot
import torch
import pickle
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute max min stats of ppl/acc predictor"
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="perplexity",
    )
    parser.add_argument("--search_space", type=str, default="s", help="search space")
    args = parser.parse_args()
    search_space = search_spaces[args.search_space]
    num_layers = max(search_space["n_layer_choices"])
    net = Net(num_layers, 128).cuda()
    net.load_state_dict(
        torch.load(
            "data_collection/gpt_datasets/predictor_ckpts/metric/"
            + str(args.metric)
            + "_"
            + str(args.search_space)
            + ".pt"
        )
    )
    net.eval()
    out_list = []
    max_arch = sample_config_max(search_spaces["s"])
    max_arch_feature_map = convert_config_to_one_hot(max_arch, args.search_space)
    max_arch_feature_map = max_arch_feature_map.unsqueeze(0).cuda()
    out = net(max_arch_feature_map)
    out_list.append(out.item())
    min_arch = sample_config_min(search_spaces["s"])
    min_arch_feature_map = convert_config_to_one_hot(min_arch, search_spaces["s"])
    min_arch_feature_map = min_arch_feature_map.unsqueeze(0).cuda()
    out = net(min_arch_feature_map)
    out_list.append(out.item())
    max_min_stats = {"max": max(out_list), "min": min(out_list)}
    max_min_save_path = (
        "data_collection/gpt_datasets/predictor_ckpts/metric/max_min_stats_"
        + str(args.metric)
        + "_"
        + str(args.search_space)
        + ".pkl"
    )
    max_min_stats = {"max": max(out_list), "min": min(out_list)}
    with open(max_min_save_path, "wb") as f:
        pickle.dump(max_min_stats, f)
    print(max_min_stats)