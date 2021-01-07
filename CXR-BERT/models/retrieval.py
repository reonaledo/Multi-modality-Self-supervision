"""
Downstream task ; Image Retrieval
"""

import os
import copy
import json
import logging
logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
from torch.nn import LayerNorm as FusedLayerNorm

from models.cxrbert import CXRBertEncoder, CXRBERT

from transformers import BertConfig, BertModel
from transformers import AutoConfig, AutoModel

class CXRBertForImageRetrieval(nn.Module):
    """
    Downstream tasks : Image Retrieval
    """
    def __init__(self, args):
        super().__init__()
        self.enc = CXRBertEncoder(args)  # CXRBertEncoder(args)

        # self.enc = BertModel.from_pretrained(config=BertConfig.from_pretrained('bert-base-uncased'), state_dict=<torch.load된 state_dict>)
        self.model = CXRBERT()
        self.ir = ImageRetrieval(args.hidden_size)

        """
        # bert = BertModel.from_pretrained(args.bert_model)
        if args.init_model == 'bert-base-uncased':
            bert = BertModel.from_pretrained(args.bert_model)
        elif args.init_model == 'BlueBERT':
            config = BertConfig.from_pretrained('bluebert')
            model_state_dict = torch.load('bluebert/pytorch_model.bin')
            bert = BertModel.from_pretrained('bluebert', state_dict=model_state_dict, config=config)

        self.txt_embeddings = bert.embeddings
        self.img_embeddings = ImageBertEmbeddings(args, self.txt_embeddings)

        self.img_encoder = ImageEncoder(args)
        self.encoder = bert.encoder

        self.pooler = bert.pooler
        
        """

    def forward(self, input_txt, attn_mask, segment, input_img):
        x = self.enc(input_txt, attn_mask, segment, input_img)
        return self.ir(x)

class ImageRetrieval(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x[:, 0]))
