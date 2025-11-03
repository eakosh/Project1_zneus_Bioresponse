import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

from datamodule import DataModule
from model import MultiLayerPerceptron
from config import *
from utils import decide_device


class Trainer:
    def __init__(self, model, optimizer, loss_fn, epochs):
        self.device = torch.device(decide_device())
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs

        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        self.best_val_loss = float('inf')
        self.best_model_state = None

    def setup(self, datamodule):
        self.datamodule = datamodule
        self.datamodule.setup()

    def fit(self):

        for epoch in range(self.epochs):

            self.train_epoch(epoch)
            self.validate_epoch(epoch)


    def train_epoch(self, epoch):
        self.model.train()

        with tqdm(self.datamodule.dataloader_train, desc=f"Train: {epoch}") as progress:
            for x, y in progress:
                x = x.to(self.device)
                y = y.to(self.device)

                y_hat = self.model(x)
                loss = self.loss_fn(y_hat, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                progress.set_postfix({"loss": f"{loss.item():.4f}"})


    def validate_epoch(self, epoch):
        self.model.eval()

        with torch.no_grad():
            with tqdm(self.datamodule.dataloader_val, desc=f"Val: {epoch}") as progress:
                for x, y in progress:
                    x = x.to(self.device)
                    y = y.to(self.device)

                    y_hat = self.model(x)
                    loss = self.loss_fn(y_hat, y)

                    progress.set_postfix({"loss": f"{loss.item():.4f}"})


    def test(self):
        self.model.eval()

        with torch.no_grad():
            with tqdm(self.datamodule.dataloader_test, desc="Test") as progress:
                for x, y in progress:
                    x = x.to(self.device)
                    y = y.to(self.device)

                    y_hat = self.model(x)
                    loss = self.loss_fn(y_hat, y)

                    progress.set_postfix({"loss": f"{loss.item():.4f}"})

