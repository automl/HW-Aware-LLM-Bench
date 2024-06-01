from hwgpt.predictors.hwmetric.net import Net
from hwgpt.predictors.hwmetric.utils import HWDataset, search_spaces
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import scipy
from hwgpt.predictors.hwmetric.utils import get_model_and_datasets
import pickle


def train(
    model: torch.nn.Module,
    device: str,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Adam,
    epoch: int,
    log_interval: int = 10,
):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.float()
        # print(data.shape)
        # print(target.shape)
        target = target.float()
        optimizer.zero_grad()
        output = model(data)
        target = torch.squeeze(target)
        output = torch.squeeze(output)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(model: torch.nn.Module, device: str, test_loader: torch.utils.data.DataLoader):
    model.eval()
    test_loss = 0
    out_test = []
    test_target = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data, target = data.float(), target.float()
            target = torch.squeeze(target)
            test_target.append(torch.squeeze(target))
            output = model(data)
            output = torch.squeeze(output)
            out_test.append(torch.squeeze(output))
            test_loss += nn.MSELoss()(output, target).item()  # sum up batch loss
    final_target = torch.cat(test_target, dim=-1)
    final_out = torch.cat(out_test, dim=-1)
    test_loss /= len(test_loader.dataset)
    # print(final_target.shape)
    # print(final_out.shape)
    print("\nTest set: Average loss: {:.4f}\n".format(test_loss))
    print(
        "Corr",
        scipy.stats.kendalltau(final_target.cpu().numpy(), final_out.cpu().numpy())[0],
    )


def gaussian_nll_loss(mean, logvar, target):
    var = torch.exp(logvar)
    nll = 0.5 * (logvar + (target - mean) ** 2 / var)
    return nll.mean()


def sample_from_gaussian(mean, logvar):
    std = torch.sqrt(torch.exp(logvar))
    output = []
    for i in range(mean.shape[0]):
        normal = torch.distributions.Normal(mean[i], std[i])
        sample = normal.sample((1,))
        output.append(sample)
    return torch.tensor(output)


def train_gaussian(
    model: torch.nn.Module,
    device: str,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Adam,
    epoch: int,
    log_interval: int = 10,
):
    model.train()
    mse = nn.MSELoss()
    for batch_idx, (data, target_mean, target_std) in enumerate(train_loader):
        data, target_mean, target_std = (
            data.to(device),
            target_mean.to(device),
            target_std.to(device),
        )
        data = data.float()
        # print(data.shape)
        # print(target.shape)
        target_mean = target_mean.float()
        target_std = target_std.float()
        optimizer.zero_grad()
        mean, logvar = model(data)
        mean = torch.squeeze(mean)
        logvar = torch.squeeze(logvar)
        target_mean = torch.squeeze(target_mean) * 1000
        target_std = torch.squeeze(target_std) * 1000

        loss = mse(mean, target_mean) + mse(torch.exp(logvar), target_std**2)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test_gaussian(
    model: torch.nn.Module, device: str, test_loader: torch.utils.data.DataLoader
):
    model.eval()
    test_loss = 0
    mse = nn.MSELoss()
    with torch.no_grad():
        for data, target_mean, target_std in test_loader:
            data, target_mean, target_std = (
                data.to(device),
                target_mean.to(device),
                target_std.to(device),
            )
            data, target_mean, target_std = (
                data.float(),
                target_mean.float(),
                target_std.float(),
            )
            target_mean = torch.squeeze(target_mean) * 1000
            target_std = torch.squeeze(target_std) * 1000
            mean, logvar = model(data)
            mean = torch.squeeze(mean)
            logvar = torch.squeeze(logvar)
            # output = sample_from_gaussian(mean, logvar)
            # print(output)
            # out_test.append(torch.squeeze(output))
            test_loss += mse(mean, target_mean) + mse(torch.exp(logvar), target_std**2)
            # break
    # final_target = torch.cat(test_target, dim=-1)
    # final_out = torch.cat(out_test, dim=-1)
    test_loss /= len(test_loader.dataset)
    print(torch.sqrt(torch.exp(logvar)))
    print(mean)
    # print(final_target.shape)
    # print(final_out.shape)
    print("\nTest set: Average loss: {:.4f}\n".format(test_loss))
    # print(
    #    "Corr",
    #    scipy.stats.kendalltau(final_target.cpu().numpy(), final_out.cpu().numpy())[0],
    # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch HW Metric Predictor")
    parser.add_argument("--device", type=str, default="a100", help="device name")
    parser.add_argument(
        "--metric",
        type=str,
        default="energies",
    )
    parser.add_argument("--search_space", type=str, default="s")
    parser.add_argument("--model", type=str, default="conformal_quantile")
    parser.add_argument("--type", type=str, default="quantile")
    parser.add_argument("--num_quantiles", type=str, default=9)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4000,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed ((default: 1)"
    )
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    print(args)
    model, train_dataset, test_dataset = get_model_and_datasets(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1024, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1024, shuffle=False
    )
    if args.model == "conformal_quantile" or args.model == "quantile":
        X_train = train_dataset.arch_features_train.data.numpy()
        X_test = test_dataset.arch_features_test.data.numpy()
        Y_train = train_dataset.latencies_train.data.numpy()
        Y_test = test_dataset.latencies_test.data.numpy()
        model.fit(X_train, Y_train)
        base_path = (
            "data_collection/gpt_datasets/predictor_ckpts/hwmetric/"
            + str(args.model)
            + "/"
        )
        model_path = (
            base_path
            + args.metric
            + "_"
            + args.type
            + "_"
            + args.search_space
            + "_"
            + args.device
            + ".pkl"
        )
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    elif args.model == "ensemble":

        X_train = train_dataset.arch_features_train.data.numpy()
        X_test = test_dataset.arch_features_test.data.numpy()
        print(X_train.shape)
        print(X_test.shape)
        Y_train = train_dataset.latencies_train.data.numpy() * 1000
        Y_test = test_dataset.latencies_test.data.numpy() * 1000
        model.train(X_train, Y_train)
        mse_mean, mse_std = model.validate(X_test, Y_test)
        print("Mean mse", mse_mean)
        print("Std mse", mse_std)
        base_path = (
            "data_collection/gpt_datasets/predictor_ckpts/hwmetric/"
            + str(args.model)
            + "/"
        )
        model_path = (
            base_path
            + args.metric
            + "_"
            + args.type
            + "_"
            + args.search_space
            + "_"
            + args.device
            + ".pkl"
        )
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    elif args.model == "mlp":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        model = model.to(device)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_path = (
            "data_collection/gpt_datasets/predictor_ckpts/hwmetric/"
            + str(args.model)
            + "/"
        )
        if "memory" in args.metric or "flops" in args.metric or "params" in args.metric:
            model_path = base_path + args.metric + "_" + args.search_space + ".pth"
        else:
            model_path = (
                base_path
                + args.metric
                + "_"
                + args.type
                + "_"
                + args.search_space
                + "_"
                + args.device
                + ".pth"
            )
        for epoch in range(1, args.epochs + 1):
            train(model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader)
        torch.save(model.state_dict(), model_path)
    elif args.model == "gaussianmlp":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        model = model.to(device)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_path = (
            "data_collection/gpt_datasets/predictor_ckpts/hwmetric/"
            + str(args.model)
            + "/"
        )
        if "memory" in args.metric or "flops" in args.metric or "params" in args.metric:
            model_path = base_path + args.metric + "_" + args.search_space + ".pth"
        else:
            model_path = (
                base_path
                + args.metric
                + "_"
                + args.type
                + "_"
                + args.search_space
                + "_"
                + args.device
                + ".pth"
            )
        for epoch in range(1, args.epochs + 1):
            train_gaussian(model, device, train_loader, optimizer, epoch)
            if (epoch + 1) % 10 == 0:
                test_gaussian(model, device, test_loader)
        torch.save(model.state_dict(), model_path)
