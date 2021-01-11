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


def get_transforms(args):
    if args.num_image_embeds < 100:
            return transforms.Compose([
                # transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2)),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    else:
        return transforms.Compose([
                # transforms.ColorJitter(brightness=(0.8, 1.2), contrast=(0.8, 1.2)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

