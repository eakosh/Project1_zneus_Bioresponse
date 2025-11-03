import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import os

from datamodule import DataModule
from model import MultiLayerPerceptron
from config import *
from utils import decide_device

if USE_WANDB:
    import warnings

    UnsupportedFieldAttributeWarning = None
    for module_path in (
            "pydantic._internal.generate_schema",
            "pydantic._internal._generate_schema",
    ):
        try:
            mod = __import__(module_path, fromlist=["UnsupportedFieldAttributeWarning"])
            UnsupportedFieldAttributeWarning = getattr(mod, "UnsupportedFieldAttributeWarning")
            break
        except Exception:
            pass

    if UnsupportedFieldAttributeWarning is not None:
        warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
    else:
        warnings.filterwarnings(
            "ignore",
            message=r".*The 'repr' attribute with value.*provided to the `Field\(\)` function.*"
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*The 'frozen' attribute with value.*provided to the `Field\(\)` function.*"
        )

    import wandb


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
        self.best_val_f1 = 0.0
        self.best_model_state = None
        self.patience_counter = 0

        os.makedirs(EXPERIMENT_PATH, exist_ok=True)

    def setup(self, datamodule):
        self.datamodule = datamodule
        self.datamodule.setup()

        if USE_WANDB:
            config = {
                "learning_rate": LEARNING_RATE,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,

                "architecture": "MultiLayerPerceptron",
                "model_string": str(self.model),
                "num_features": self.datamodule.num_x,
                "total_params": sum(p.numel() for p in self.model.parameters()),
                "trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad),

                "optimizer": type(self.optimizer).__name__,
                "loss_function": type(self.loss_fn).__name__,

                "train_size": len(self.datamodule.train_dataset),
                "val_size": len(self.datamodule.val_dataset),
                "test_size": len(self.datamodule.test_dataset),
                "test_split": TEST_SIZE,
                "val_split": VAL_SIZE,
                "random_state": RANDOM_STATE,
                "stratify": STRATIFY,

                "preprocess": PREPROCESS,
                "remove_duplicates": REMOVE_DUPLICATES,
                "remove_zero_columns": REMOVE_ZERO_COLUMNS,
                "remove_constant_columns": REMOVE_CONSTANT_COLUMNS,
                "remove_low_variance_columns": REMOVE_LOW_VARIANCE_COLUMNS,
                "variance_threshold": VARIANCE_THRESHOLD,

                "feature_selection": APPLY_FEATURE_SELECTION,
                "feature_selection_methods": FEATURE_SELECTION_METHOD if APPLY_FEATURE_SELECTION else None,
                "top_n_features": TOP_N_FEATURES if APPLY_FEATURE_SELECTION else self.datamodule.num_x,
                "selected_features": self.datamodule.num_x,

                "normalization": NORMALIZATION_METHOD,

                "device": str(self.device),
                "experiment_name": EXPERIMENT_NAME,
                "save_best_model": SAVE_BEST_MODEL,
            }

            self.run = wandb.init(
                project=WANDB_PROJECT,
                entity=WANDB_ENTITY,
                name=EXPERIMENT_NAME,
                config=config
            )


    def fit(self):

        for epoch in range(self.epochs):

            train_loss = self.train_epoch(epoch)
            val_loss, val_metrics = self.validate_epoch(epoch)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)

            if USE_WANDB:
                self.run.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": val_metrics['accuracy'],
                    "val_precision": val_metrics['precision'],
                    "val_recall": val_metrics['recall'],
                    "val_f1": val_metrics['f1'],
                    "val_roc_auc": val_metrics['roc_auc']
                })

            if SAVE_BEST_MODEL and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_val_f1 = val_metrics['f1']
                self.best_model_state = self.model.state_dict().copy()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_metrics': val_metrics,
                }, MODEL_CHECKPOINT_PATH)
                print(f"Best model saved: (Val Loss: {val_loss:.4f}, F1: {val_metrics['f1']:.4f})")
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if APPLY_EARLY_STOPPING_PATIENCE:
                if self.patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break

        if SAVE_BEST_MODEL and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\nBest model loaded: (Val Loss: {self.best_val_loss:.4f}, F1: {self.best_val_f1:.4f})")

        # Сохраняем графики
        self.plot_training_history()


    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        with tqdm(self.datamodule.dataloader_train, desc=f"Train: {epoch}") as progress:
            for x, y in progress:
                x = x.to(self.device)
                y = y.to(self.device)

                y_hat = self.model(x)
                loss = self.loss_fn(y_hat, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1
                progress.set_postfix({"loss": f"{loss.item():.4f}"})

        return epoch_loss / num_batches


    def validate_epoch(self, epoch):
        self.model.eval()
        epoch_loss = 0.0
        num_batches = 0
        all_preds = []
        all_probs = []
        all_targets = []

        with torch.no_grad():
            with tqdm(self.datamodule.dataloader_val, desc=f"Val: {epoch}") as progress:
                for x, y in progress:
                    x = x.to(self.device)
                    y = y.to(self.device)

                    y_hat = self.model(x)
                    loss = self.loss_fn(y_hat, y)

                    epoch_loss += loss.item()
                    num_batches += 1

                    probs = torch.sigmoid(y_hat)
                    preds = (probs > 0.5).float()

                    all_preds.extend(preds.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                    all_targets.extend(y.cpu().numpy())

                    progress.set_postfix({"loss": f"{loss.item():.4f}"})

        metrics = self.compute_metrics(all_targets, all_preds, all_probs)

        return epoch_loss / num_batches, metrics


    def test(self):
        self.model.eval()
        test_loss = 0.0
        num_batches = 0
        all_preds = []
        all_probs = []
        all_targets = []

        with torch.no_grad():
            with tqdm(self.datamodule.dataloader_test, desc="Test") as progress:
                for x, y in progress:
                    x = x.to(self.device)
                    y = y.to(self.device)

                    y_hat = self.model(x)
                    loss = self.loss_fn(y_hat, y)

                    test_loss += loss.item()
                    num_batches += 1

                    probs = torch.sigmoid(y_hat)
                    preds = (probs > 0.5).float()

                    all_preds.extend(preds.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                    all_targets.extend(y.cpu().numpy())

                    progress.set_postfix({"loss": f"{loss.item():.4f}"})

        test_metrics = self.compute_metrics(all_targets, all_preds, all_probs)
        avg_test_loss = test_loss / num_batches

        self.run.summary["test_loss"] = avg_test_loss
        self.run.summary["test_metrics"] = test_metrics

        return test_metrics


    def compute_metrics(self, targets, preds, probs):

        targets = np.array(targets).flatten()
        preds = np.array(preds).flatten()
        probs = np.array(probs).flatten()

        metrics = {
            'accuracy': accuracy_score(targets, preds),
            'precision': precision_score(targets, preds, zero_division=0),
            'recall': recall_score(targets, preds, zero_division=0),
            'f1': f1_score(targets, preds, zero_division=0),
            'roc_auc': roc_auc_score(targets, probs)
        }

        return metrics


    def plot_training_history(self):

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16)

        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        accuracies = [m['accuracy'] for m in self.val_metrics]
        axes[0, 1].plot(accuracies, label='Val Accuracy', color='green')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        precisions = [m['precision'] for m in self.val_metrics]
        axes[0, 2].plot(precisions, label='Val Precision', color='blue')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].set_title('Precision')
        axes[0, 2].legend()
        axes[0, 2].grid(True)

        recalls = [m['recall'] for m in self.val_metrics]
        axes[1, 0].plot(recalls, label='Val Recall', color='orange')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].set_title('Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        f1_scores = [m['f1'] for m in self.val_metrics]
        axes[1, 1].plot(f1_scores, label='Val F1', color='red')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].set_title('F1 Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        roc_aucs = [m['roc_auc'] for m in self.val_metrics]
        axes[1, 2].plot(roc_aucs, label='Val ROC-AUC', color='purple')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('ROC-AUC')
        axes[1, 2].set_title('ROC-AUC')
        axes[1, 2].legend()
        axes[1, 2].grid(True)

        plt.tight_layout()
        plot_path = os.path.join(EXPERIMENT_PATH, 'training_history.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nTraining plots saved to: {plot_path}")

        if USE_WANDB:
            self.run.log({"training_history": wandb.Image(fig)})

        plt.close()

    def finish(self):
        if USE_WANDB:
            summary_dict = {
                "best_val_loss": self.best_val_loss,
                "best_val_f1": self.best_val_f1,
                "total_epochs_trained": len(self.train_losses),
                "early_stopped": len(self.train_losses) < EPOCHS,
            }

            for key, value in summary_dict.items():
                self.run.summary[key] = value

            self.run.finish()

