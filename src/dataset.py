import cv2
from pathlib import Path
from typing import Callable
import urllib.request
import zipfile
from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn.functional as tnf


class BioresponseDataset(Dataset):
    def __init__(self):
        self.path = Path("./data/phpSSK7iA.csv")