from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import numpy as np



def main(args: Namespace) -> None:
    # generate mobility data for input with array
    data = np.load(args.data_dir, allow_pickle=True)
    
    # load data
    train_loader, test_loader = load_data(args, data)

    # build model
    model = build_model(args)

    # train model
    train(args, model, train_loader)

    # test model
    test(args, model, test_loader)
    
    
def load_data(args: Namespace) -> Tuple[DataLoader, DataLoader]:
    pass


def build_model(args: Namespace) -> nn.Module:
    pass


def train(args: Namespace, model: nn.Module, train_loader: DataLoader) -> None:
    pass


def test(args: Namespace, model: nn.Module, test_loader: DataLoader) -> None:
    pass


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset(1node or 100node).",
        default="./data/5min_1node_data.npy",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
