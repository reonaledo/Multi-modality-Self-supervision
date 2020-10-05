# Pre-training: Masked Language Modelling, Image Text Matching

from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layer import GELU, BertOnlyMLMHead
from .modelling import CXR_Model, CXR_PreTrainedModel

class CXRForPretraining(CXR_PreTrainedModel):
    """ CXR pretraining """
    def __init__(self, config, img_dim):
        super().__init__(config)
        self.cxr = CXR_Model(config, img_dim)
        self.cls = BertOnlyMLMHead(config, self.cxr.embeddings.word_embeddings.weight)
        self.itm_output = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_weights)

    def forward(self, batch, task, compute_loss=True):
        batch = defaultdict(lambda: None, batch)
        input_ids = batch['input_ids']
        position_ids = batch['position_ids']
        img_feat = batch['img_feat']
        attention_mask = batch['attn_masks']
        gather_index = batch['gather_index']

        if task == 'mlm':
            txt_labels = batch['txt_labels']
            return self.forward_mlm(input_ids, position_ids, img_feat,
                                    attention_mask, gather_index,
                                    txt_labels, compute_loss)

        elif task == 'itm':
            targets = batch['targets']
            return self.forward_itm(input_ids, position_ids, img_feat,
                                    attention_mask, gather_index,
                                    targets, compute_loss)

        else:
            raise ValueError('Invalid task, MLM or ITM')

    def forward_mlm(self, input_ids, position_ids, img_feat,
                    attention_mask, gather_index, txt_labels, compute_loss=True):
        sequence_output = self.cxr(input_ids, position_ids, img_feat,
                                   attention_mask, gather_index,
                                   output_all_encoded_layers=False)
        # get only the text part
        sequence_output = sequence_output[:, :input_ids.size(1), :]
        # only compute masked tokens for better efficiency
        masked_output = self._compute_masked_hidden(sequence_output, txt_labels != -1)
        prediction_scores = self.cls(masked_output)

        if compute_loss:
            masked_lm_loss = F.cross_entropy(prediction_scores, txt_labels[txt_labels != -1], reduction='none')
            return masked_lm_loss
        else:
            return prediction_scores

    def _compute_masked_hidden(self, hidden, mask):
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked

    def forward_itm(self, input_ids, position_ids, img_feat,
                    attention_mask, gather_index, targets, compute_loss=True):
        sequence_output = self.cxr(input_ids, position_ids, img_feat,
                                   attention_mask, gather_index,
                                   output_all_encoded_layers=False)
        pooled_output = self.cxr.pooler(sequence_output)
        itm_scores = self.itm_output(pooled_output)

        if compute_loss:
            itm_loss = F.cross_entropy(itm_scores, targets, reduction='none')
            return itm_loss
        else:
            return itm_scores
