"""Main training script — runs full pipeline for model training, validation, and testing."""

import torch.nn as nn
import torch.optim as optim

from datamodule import DataModule
from model import MultiLayerPerceptron
from config import *
from trainer import Trainer


def main():
    """Execute full training workflow."""
    print("Loading and preparing data")

    datamodule = DataModule()
    datamodule.setup()

    model = MultiLayerPerceptron(
        nin=datamodule.num_x,
        nhidden=[128, 64, 32],
        nout=1,
    )

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCELoss()

    trainer = Trainer(
        epochs=EPOCHS,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        apply_early_stopping=APPLY_EARLY_STOPPING_PATIENCE,
        early_stopping_patience=EARLY_STOPPING_PATIENCE
    )

    trainer.setup(datamodule)
    trainer.fit()
    trainer.test()

    if USE_WANDB:
        trainer.finish()


if __name__ == "__main__":
    main()