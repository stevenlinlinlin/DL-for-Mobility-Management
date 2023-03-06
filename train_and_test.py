from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import get_dataloader
from model import *



def main(args: Namespace) -> None:
    print(f"------Model name: {args.model}-------")
    # generate mobility data for input with array
    data = np.load(args.data_dir, allow_pickle=True)
    #print(data.shape)
    
    # load data
    train_loader, test_loader = load_data(args, data)

    # build model
    model = build_model(args)

    
    # train model
    if args.train==1:
        train(args, model, train_loader, args.model)

    # test model
    if args.train==0:
        model.load_state_dict(torch.load(args.save_dir))
        
    test(args, model, test_loader)
    
    
def load_data(args, data) -> Tuple[DataLoader, DataLoader]:
    x = []
    y = []
    for i in range(0, len(data)-5):
        x.append([data[j] for j in range(i, i+5)])
        y.append(data[i+5])
    x = torch.tensor(np.array(x), dtype= torch.float32)
    y = torch.tensor(np.array(y), dtype = torch.float32)
    return get_dataloader(args.batch_size, x, y)


def build_model(args) -> nn.Module:
    if args.model == 'LSTM':
        model = LSTM(2,64,3).to(args.device)
    elif args.model == 'GRU':
        model = GRU(2,64,3).to(args.device)
    elif args.model == 'TCN':
        model = TCN().to(args.device)
    return model


def train(args: Namespace, model: nn.Module, train_loader: DataLoader, model_name: str) -> None:
    learning_rate = args.lr
    loss = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum = 0.9)
    num_epoch = args.num_epoch
    
    model.train()
    train_loss_array = []
    for epoch in tqdm(range(num_epoch)):
        train_loss = 0
        for (inputs, labels) in train_loader:
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            #print(inputs.shape)

            y_pred = model(inputs)
            l = loss(y_pred, labels)
            train_loss += l.item()

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        
        train_loss_array.append(train_loss/len(train_loader))
        # print("Epoch: ", epoch, "Loss: ", train_loss/len(train_loader))

    torch.save(model.state_dict(), model_name + '.pt')
    # plt.plot(train_loss_array)
    # plt.savefig('train_loss.png')


def test(args: Namespace, model: nn.Module, test_loader: DataLoader) -> None:
    model.eval()
    mae = 0
    for (inputs, labels) in test_loader:
        #print(len(test_loader))
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        #print(inputs.shape)
        
        prediction = model(inputs)
        mae += torch.mean(torch.abs(prediction - labels))
    
    print("MAE: ", (mae/len(test_loader)).item())


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset(1node or 100node).",
        default="./data/5min_1node_data.npy",
    )
    # parser.add_argument(
    #     "--save_dir",
    #     type=Path,
    #     help="Directory to save the model.",
    #     default="./models/",
    # )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda:0", default="cuda:0"
    )
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--model", type=str, default="LSTM")
    parser.add_argument("--train", type=int, default=1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
