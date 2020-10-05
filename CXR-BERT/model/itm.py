"""
Image Text Matching for MIMIC-CXR BERT
"""

import os
import sys
from collections import defaultdict

import torch
import torch.nn as nn

from .modelling import CXR_PreTrainedModel, CXR_Model

class CXRForImageTextRetrieval(CXR_PreTrainedModel):
    """
    Fine-tune CXR-Bert for image text retrieval
    """
    def __init__(self, config, img_dim, margin=0.2):
        super().__init__(config)
        self.cxr = CXR_Model(config, img_dim)
        self.itm_output = nn.Linear(config.hidden_size, 2)
        self.rank_output = nn.Linear(config.hidden_size, 1)
        self.margin = margin
        self.apply(self.init_weights)

    def init_output(self):
        """
        Need to be called after from pretrained
        """
        self.rank_output.weight.data = self.itm_output.weight.data[1:, :]
        self.rank_output.bias.data = self.itm_output.bias.data[1:]

    def forward(self, batch, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        attention_mask = batch['attn_mask']
        gather_index = batch['gather_index']

        sequence_output = self.cxr(input_ids, position_ids, img_feat,
                                   attention_mask, gather_index, output_all_encoded_layers=False)
        pooled_output = self.cxr.pooler(sequence_output)
        rank_scores = self.rank_output(pooled_output)

        # TODO: chekchekhckehckechkechkecheck ! ! !
        if compute_loss:
            #triplet loss
            rank_scores_sigmoid = torch.sigmoid(rank_scores)
            sample_size = batch['sample_size']
            scores = rank_scores_sigmoid.contiguous().view(-1, sample_size)
            pos = scores[:, :1]
            neg = scores[:, 1:]
            rank_loss = torch.clamp(self.margin + neg - pos, 0)
            return rank_loss
        else:
            return rank_scores

# TODO: option
class CXRForImageTextRetrievalHardNeg(CXR_PreTrainedModel):
    pass
