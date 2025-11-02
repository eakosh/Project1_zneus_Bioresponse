import torch.nn as nn

class MultiLayerPerceptron(nn.Module):
    def __init__(self,
                 nin,
                 nhidden,
                 nout):
        super().__init__()

        if isinstance(nhidden, int):
            nhidden = [nhidden]

        layers = []
        prev_dim = nin

        for hidden_dim in nhidden:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, nout))

        self.main = nn.Sequential(*layers)



    def forward(self, x):
        return self.main(x)
