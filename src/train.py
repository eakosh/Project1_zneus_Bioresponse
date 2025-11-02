import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from datamodule import DataModule
from model import MultiLayerPerceptron
from config import *


from utils import decide_device

def main():
    device = torch.device(decide_device())

    datamodule = DataModule()
    datamodule.setup()

    model = MultiLayerPerceptron(
        nin=datamodule.num_x,
        nhidden=56,
        nout=1,
    )
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(100):
        with tqdm(datamodule.dataloader_train, desc="Train:") as progress:
            for x, y in progress:

                x = x.to(device)
                y = y.to(device)

                y_hat = model(x)
                l = loss_fn(y_hat, y)

                optimizer.zero_grad()
                l.backward()
                optimizer.step()

                progress.set_postfix({"loss": l.item()})

        with tqdm(datamodule.dataloader_val, desc="Val:") as progress:
            for x, y in progress:

                x = x.to(device)
                y = y.to(device)

                y_hat = model(x)
                l = loss_fn(y_hat, y)

                progress.set_postfix({"loss": l.item()})

    with tqdm(datamodule.dataloader_test, desc="Test:") as progress:
        for x, y in progress:

            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)
            l = loss_fn(y_hat, y)

            progress.set_postfix({"loss": l.item()})


if __name__ == "__main__":
    main()
