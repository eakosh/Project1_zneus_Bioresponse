import torch.nn as nn
from config import *


class MultiLayerPerceptron(nn.Module):
    """Fully connected feedforward network with LayerNorm, ReLU, and Dropout."""

    def __init__(self, nin, nhidden, nout, dropout=DROPOUT):
        """Initialize network structure with given input, hidden, and output sizes."""
        super().__init__()

        if isinstance(nhidden, int):
            nhidden = [nhidden]

        layers = []
        prev_dim = nin

        for hidden_dim in nhidden:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim

        layers += [nn.Linear(prev_dim, nout), nn.Sigmoid()]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        """Forward propagation through the network."""
        return self.main(x)