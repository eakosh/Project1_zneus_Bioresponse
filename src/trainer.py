import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import os
from config import *


# for warning from wandb
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
        warnings.filterwarnings("ignore", message=r".*The 'repr' attribute with value.*")
        warnings.filterwarnings("ignore", message=r".*The 'frozen' attribute with value.*")

    import wandb


def decide_device():
    """Select available device: CUDA, MPS, or CPU."""
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


class Trainer:
    """Handles model training, validation, testing, and logging."""

    def __init__(self, model, optimizer, loss_fn, epochs, apply_early_stopping, early_stopping_patience):
        """Initialize trainer with model, optimizer, loss, and settings."""
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
        self.apply_early_stopping = apply_early_stopping
        self.early_stopping_patience = early_stopping_patience

        os.makedirs(EXPERIMENT_PATH, exist_ok=True)

    def setup(self, datamodule):
        """Initialize datamodule and setup experiment tracking"""
        self.datamodule = datamodule
        self.datamodule.setup()

        if USE_WANDB:
            config = {
                "learning_rate": LEARNING_RATE,
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "early_stopping_patience": self.early_stopping_patience,
                "architecture": "MultiLayerPerceptron",
                "model_string": str(self.model),
                "num_features": self.datamodule.num_x,
                "total_params": sum(p.numel() for p in self.model.parameters()),
                "trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                "optimizer": type(self.optimizer).__name__,
                "loss_function": type(self.loss_fn).__name__,
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
        """Run full training and validation loop."""
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

            if self.apply_early_stopping and self.patience_counter >= self.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break

        if SAVE_BEST_MODEL and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"\nBest model loaded: (Val Loss: {self.best_val_loss:.4f}, F1: {self.best_val_f1:.4f})")

        self.plot_training_history()

    def train_epoch(self, epoch):
        """Train model for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        with tqdm(self.datamodule.dataloader_train, desc=f"Train: {epoch}") as progress:
            for x, y in progress:
                x, y = x.to(self.device), y.to(self.device)
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
        """Validate model and compute metrics."""
        self.model.eval()
        epoch_loss = 0.0
        num_batches = 0
        all_preds, all_probs, all_targets = [], [], []

        with torch.no_grad():
            with tqdm(self.datamodule.dataloader_val, desc=f"Val: {epoch}") as progress:
                for x, y in progress:
                    x, y = x.to(self.device), y.to(self.device)
                    y_hat = self.model(x)
                    loss = self.loss_fn(y_hat, y)

                    epoch_loss += loss.item()
                    num_batches += 1
                    preds = (y_hat > 0.5).float()

                    all_preds.extend(preds.cpu().numpy())
                    all_probs.extend(y_hat.cpu().numpy())
                    all_targets.extend(y.cpu().numpy())

                    progress.set_postfix({"loss": f"{loss.item():.4f}"})

        metrics = self.compute_metrics(all_targets, all_preds, all_probs)
        return epoch_loss / num_batches, metrics

    def test(self):
        """Evaluate the model on test data."""
        self.model.eval()
        test_loss = 0.0
        num_batches = 0
        all_preds, all_probs, all_targets = [], [], []

        with torch.no_grad():
            with tqdm(self.datamodule.dataloader_test, desc="Test") as progress:
                for x, y in progress:
                    x, y = x.to(self.device), y.to(self.device)
                    y_hat = self.model(x)
                    loss = self.loss_fn(y_hat, y)

                    test_loss += loss.item()
                    num_batches += 1
                    preds = (y_hat > 0.5).float()

                    all_preds.extend(preds.cpu().numpy())
                    all_probs.extend(y_hat.cpu().numpy())
                    all_targets.extend(y.cpu().numpy())
                    progress.set_postfix({"loss": f"{loss.item():.4f}"})

        test_metrics = self.compute_metrics(all_targets, all_preds, all_probs)
        avg_test_loss = test_loss / num_batches

        self.run.summary["test_loss"] = avg_test_loss
        self.run.summary["test_metrics"] = test_metrics

        return test_metrics

    def compute_metrics(self, targets, preds, probs):
        """Compute accuracy, precision, recall, F1, and ROC-AUC."""
        targets, preds, probs = map(lambda x: np.array(x).flatten(), [targets, preds, probs])
        return {
            'accuracy': accuracy_score(targets, preds),
            'precision': precision_score(targets, preds, zero_division=0),
            'recall': recall_score(targets, preds, zero_division=0),
            'f1': f1_score(targets, preds, zero_division=0),
            'roc_auc': roc_auc_score(targets, probs)
        }

    def plot_training_history(self):
        """Plot loss and metrics per epoch."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16)

        metrics_map = {
            (0, 0): ('Loss', self.train_losses, self.val_losses),
            (0, 1): ('Accuracy', [m['accuracy'] for m in self.val_metrics]),
            (0, 2): ('Precision', [m['precision'] for m in self.val_metrics]),
            (1, 0): ('Recall', [m['recall'] for m in self.val_metrics]),
            (1, 1): ('F1 Score', [m['f1'] for m in self.val_metrics]),
            (1, 2): ('ROC-AUC', [m['roc_auc'] for m in self.val_metrics]),
        }

        for (i, j), (title, *data) in metrics_map.items():
            ax = axes[i, j]
            if len(data) == 2:
                ax.plot(data[0], label='Train Loss')
                ax.plot(data[1], label='Val Loss')
            else:
                ax.plot(data[0], label=f'Val {title}', color='C1')
            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        path = os.path.join(EXPERIMENT_PATH, 'training_history.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"\nTraining plots saved to: {path}")
        if USE_WANDB:
            self.run.log({"training_history": wandb.Image(fig)})
        plt.close()

    def finish(self):
        """Finalize W&B run with summary metrics."""
        if USE_WANDB:
            summary = {
                "best_val_loss": self.best_val_loss,
                "best_val_f1": self.best_val_f1,
                "epochs_trained": len(self.train_losses),
                "early_stopped": len(self.train_losses) < EPOCHS,
            }
            for k, v in summary.items():
                self.run.summary[k] = v
            self.run.finish()