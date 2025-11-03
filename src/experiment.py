import torch.nn as nn
import torch.optim as optim

from datamodule import DataModule
from model import MultiLayerPerceptron
from config import *
from trainer import Trainer


def main():

    print("Loading and preparing data...")
    datamodule = DataModule()
    datamodule.setup()

    model = MultiLayerPerceptron(
        nin=datamodule.num_x,
        nhidden=[128, 64, 32, 16, 8],
        nout=1,
    )

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()

    trainer = Trainer(epochs=EPOCHS, model=model, loss_fn=loss_fn, optimizer=optimizer)
    trainer.setup(datamodule)
    trainer.fit()
    trainer.test()

    if USE_WANDB:
        trainer.finish()


if __name__ == "__main__":
    main()
