import torch
import os
import argparse
import csv
import math
import json
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

class CustomDataset(Dataset):
    def __init__(self):
        self.data = torch.randn(128, 1)
        self.label = 3 * self.data + 2
        
    def __getitem__(self, index):
        return self.data[index], self.label[index]
        
    def __len__(self):
        return len(self.data)
