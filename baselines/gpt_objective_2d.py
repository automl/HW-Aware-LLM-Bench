from syne_tune import Reporter
import time
from syne_tune import Reporter
from lib.utils import (
    get_arch_feature_map,
    convert_config_to_one_hot,
    normalize_arch_feature_map,
    get_ppl_predictor_surrogate,
    get_hw_predictor_surrogate,
    get_max_min_stats,
    predict_hw_surrogate,
    normalize_energy,
    normalize_latency,
    normalize_ppl,
    search_spaces,
)
from typing import Dict, Any

report = Reporter()


def objective(
    sampled_config: Dict[str, Any],
    device: str,
    search_space: str,
    surrogate_type: str,
    objective: str,
) -> Reporter:
    max_layers = get_max_min_stats(search_space)["max_layers"]
    arch_feature_map = get_arch_feature_map(sampled_config, search_space)
    arch_feature_map_ppl_predictor = convert_config_to_one_hot(
        sampled_config, search_space=search_space
    )
    arch_feature_map_predictor = normalize_arch_feature_map(
        arch_feature_map, search_space
    )

    ppl_predictor = get_ppl_predictor_surrogate(search_space)
    perplexity = ppl_predictor(arch_feature_map_ppl_predictor.cuda().unsqueeze(0))
    hw_predictor = get_hw_predictor_surrogate(
        max_layers, search_space, device, surrogate_type, objective
    )
    hw_metric = predict_hw_surrogate(
        [arch_feature_map_predictor], hw_predictor, surrogate_type
    )
    ppl = perplexity.item()
    ppl_norm = normalize_ppl(ppl, search_space)
    if objective == "energy":
        hw_metric_norm = normalize_energy(hw_metric, device, search_space)
    elif objective == "latency":
        hw_metric_norm = normalize_latency(hw_metric, device, search_space)
    report(perplexity=ppl_norm, hw_metric=hw_metric_norm)


if __name__ == "__main__":
    import logging
    import argparse

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--surrogate_type", type=str, default="conformal_quantile")
    parser.add_argument("--search_space", type=str, default="s")
    parser.add_argument("--device", type=str, default="P100")
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--bias", type=bool, default=True)
    parser.add_argument("--objective", type=str, default="energy")
    args = parser.parse_known_args()[0]
    search_space = search_spaces[args.search_space]
    max_layers = max(search_spaces["n_layer_choices"])
    for i in range(max_layers):
        parser.add_argument(f"--num_heads_{i}", type=int, default=12)
        parser.add_argument(f"--mlp_ratio_{i}", type=int, default=4)

    args, _ = parser.parse_known_args()
    # Evaluate objective and report results to Syne Tune
    print(vars(args))
    sample_config = {}
    sample_config["sample_n_layer"] = args.num_layers
    sample_config["sample_embed_dim"] = args.embed_dim
    sample_config["sample_bias"] = args.bias
    sample_config["sample_n_head"] = []
    sample_config["sample_mlp_ratio"] = []
    for i in range(max_layers):
        sample_config["sample_n_head"].append(getattr(args, f"num_heads_{i}"))
        sample_config["sample_mlp_ratio"].append(getattr(args, f"mlp_ratio_{i}"))
    objective(
        sampled_config=sample_config,
        search_space=args.search_space,
        surrogate_type=args.surrogate_type,
        device=args.device,
        objective=args.objective,
    )
