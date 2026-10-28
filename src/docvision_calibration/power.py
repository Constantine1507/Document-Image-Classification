"""Low-dimensional neural ensemble pooling, fitted with bounded grid optimization."""
import numpy as np
import torch
from torch import nn


class PowerMeanPool(nn.Module):
    """Pool two frozen neural classifiers using a weighted generalized mean."""
    def __init__(self, weight=.5, power=0.):
        """Store the ConvNeXt weight and the pooling exponent as fixed buffers."""
        super().__init__();self.register_buffer('weights',torch.tensor([weight,1-weight],dtype=torch.float64))
        self.power=float(power)

    def forward(self, probability):
        """Return normalized class probabilities; zero exponent is the geometric mean."""
        logs=probability.clamp_min(1e-12).log()
        if abs(self.power)<1e-12:pooled=(logs*self.weights[None,:,None]).sum(1)
        else:pooled=torch.logsumexp(self.power*logs+self.weights.log()[None,:,None],dim=1)/self.power
        return pooled.softmax(1)


@torch.inference_mode()
def grid_predictions(probability, grid):
    """Evaluate a predeclared parameter grid without using any target labels."""
    torch.set_num_threads(2)
    tensor=torch.as_tensor(probability,dtype=torch.float64)
    return np.stack([PowerMeanPool(**cfg)(tensor).numpy() for cfg in grid])


def select_grid(predictions, labels, grid, tolerance_rows):
    """Fit the two pooling parameters on training rows only, preferring simple ties."""
    correct=np.sum(predictions.argmax(2)==np.asarray(labels)[None],axis=1)
    eligible=np.where(correct>=correct.max()-tolerance_rows)[0]
    if tolerance_rows==0:
        index=min(eligible,key=lambda i:(abs(grid[i]['weight']-.5),abs(grid[i]['power'])))
        return [int(index)]
    return eligible.astype(int).tolist()
