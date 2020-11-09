"""
Load dataset

"""

import os
import json
import functools
from collections import Counter

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


from data.vocab import Vocab
from data.dataset import CXRDataset
from models.cxrbert import CXRBERT, CXRBertEncoder
from models.train import CXRBERT_Trainer

from transformers import BertTokenizer


def get_transforms():
    return transforms.Compose(
        [
            transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2)),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),

        ]
    )

