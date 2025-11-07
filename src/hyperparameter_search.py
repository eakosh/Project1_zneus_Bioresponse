import torch.nn as nn
import torch.optim as optim
import wandb
import pandas as pd

from datamodule import DataModule
from model import MultiLayerPerceptron
from trainer import Trainer
from config import WANDB_PROJECT, WANDB_ENTITY


sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'architecture': {
            'values': [
                [64],
                [128],
                [128, 64],
                [256, 128],
                [128, 64, 32],
                [64, 32, 16],
                [64, 32]
            ]
        },

        'dropout': {
            'distribution': 'uniform',
            'min': 0.1,
            'max': 0.5
        },

        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-4,
            'max': 1e-2
        },

        'weight_decay': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-2
        },

        'batch_size': {
            'values': [64, 128, 256, 512]
        },

        'top_n_features': {
            'values': [64, 128, 256]
        },

        'feature_selection_method': {
            'values': [
                ['corr', 'mi', 'rf'],
                ['corr'],
                ['mi'],
                ['rf'],
                ['mi', 'rf'],
                ['corr', 'mi'],
                ['corr', 'rf'],
            ]
        },

        'variance_threshold': {
            'distribution': 'uniform',
            'min': 0.0001,
            'max': 0.1
        },

        'early_stopping_patience': {
            'distribution': 'int_uniform',
            'min': 5,
            'max': 50
        }
    }
}


all_results = []


def train_sweep():

    run = wandb.init()
    config = wandb.config

    print(f"\n\nStarting experiment: {run.name}\n")
    print(f"Architecture: {config.architecture}")
    print(f"Dropout: {config.dropout:.3f}")
    print(f"Learning Rate: {config.learning_rate:.6f}")
    print(f"Weight Decay: {config.weight_decay:.6f}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Top N Features: {config.top_n_features}")
    print(f"Feature Selection: {config.feature_selection_method}")
    print(f"Variance Threshold: {config.variance_threshold:.6f}")
    print(f"Early Stopping Patience: {config.early_stopping_patience}")
    print(f"\n\n")


    datamodule = DataModule(
        batch_size=config.batch_size,
        num_features=config.top_n_features,
        feature_selection_method=list(config.feature_selection_method),
        variance_threshold=config.variance_threshold
    )
    datamodule.setup()

    model = MultiLayerPerceptron(
        nin=datamodule.num_x,
        nhidden=config.architecture,
        nout=1,
        dropout=config.dropout
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    loss_fn = nn.BCEWithLogitsLoss()

    trainer = Trainer(
        epochs=100,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        apply_early_stopping=True,
        early_stopping_patience=config.early_stopping_patience,
    )

    trainer.setup(datamodule)

    trainer.fit()

    test_metrics = trainer.test()

    wandb.log({
        'final_test_f1': test_metrics['f1'],
        'final_test_accuracy': test_metrics['accuracy'],
        'final_test_roc_auc': test_metrics['roc_auc'],
        'final_test_precision': test_metrics['precision'],
        'final_test_recall': test_metrics['recall']
    })

    result = {
        'run_name': run.name,
        'run_id': run.id,

        'architecture': str(config.architecture),
        'dropout': config.dropout,
        'learning_rate': config.learning_rate,
        'weight_decay': config.weight_decay,
        'batch_size': config.batch_size,
        'top_n_features': config.top_n_features,
        'feature_selection_method': str(config.feature_selection_method),
        'variance_threshold': config.variance_threshold,
        'early_stopping_patience': config.early_stopping_patience,

        'best_val_loss': trainer.best_val_loss,
        'best_val_f1': trainer.best_val_f1,

        'test_f1': test_metrics['f1'],
        'test_accuracy': test_metrics['accuracy'],
        'test_roc_auc': test_metrics['roc_auc'],
        'test_precision': test_metrics['precision'],
        'test_recall': test_metrics['recall'],

        'num_features_used': datamodule.num_x,
        'total_epochs': len(trainer.train_losses),
    }

    all_results.append(result)


    print(f"\nExperiment finished:\n")
    print(f"Val Loss: {trainer.best_val_loss:.4f}, Val F1: {trainer.best_val_f1:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}, Test ROC-AUC: {test_metrics['roc_auc']:.4f}")



def main():

    sweep_id = wandb.sweep(
        sweep_config,
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY
    )

    print(f"Sweep ID: {sweep_id}")
    print(f"Results: https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}")

    wandb.agent(
        sweep_id,
        function=train_sweep,
        count=100)

    print("\nSweep finished")

    if all_results:

        df = pd.DataFrame(all_results).sort_values('best_val_loss', ascending=True)

        print("\nTop 5 beast models:\n")

        for i, row in df.head(5).iterrows():
            print(f"\n#{df.index.get_loc(i) + 1}. {row['run_name']}")
            print(f"   Val Loss: {row['best_val_loss']:.4f} | Val F1: {row['best_val_f1']:.4f}")
            print(f"   Test F1: {row['test_f1']:.4f} | Test ROC-AUC: {row['test_roc_auc']:.4f}")
            print(f"   Architecture: {row['architecture']}")
            print(f"   LR: {row['learning_rate']:.6f} | Dropout: {row['dropout']:.3f}")
            print(f"   Features: {row['top_n_features']} | Method: {row['feature_selection_method']}")

        print(f"\nResults: https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/sweeps/{sweep_id}")


if __name__ == "__main__":
    main()