from typing import Any
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm

class Config():
    def __init__(self, in_feature_size, out_feature_size, bias, **kwargs):
        self.in_feature_size = in_feature_size
        self.out_feature_size = out_feature_size
        self.bias = bias

class Processor():
    def __init__(self,  **kwargs):
        pass
    def __call__(self, input, **kwargs):
        return {'input': input}

class LinearRegressionModel(nn.Module):
    def __init__(self, config : Config):
        super(LinearRegressionModel, self).__init__()
        self.config = config
        self.loss = nn.MSELoss()
        self.layer = nn.Linear(config.in_feature_size, config.out_feature_size, bias=config.bias)
        self.device = next(self.parameters()).device

    def forward(self, input, **kwargs):
        self.device = next(self.parameters()).device
        B = input.shape[0]

        input = input.to(self.device)
        out = self.layer(input)
        return out