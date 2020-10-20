#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn

from models.bert import BertEncoder
from models.image import ImageEncoder

" concatenating img and txt's bert result from each single modality"
" -> We concatenate the outputs of the Bert and the Img baselines. In this case," 
" the input to the classifier is 2048+768-dimensions. This is a"
" highly competitive baseline, since it combines the best encoder"
" for each modality such that the classifier has direct"
" access to the encoder outputs."

class MultimodalConcatBertClf(nn.Module):
    def __init__(self, args):
        super(MultimodalConcatBertClf, self).__init__()
        self.args = args
        self.txtenc = BertEncoder(args)
        self.imgenc = ImageEncoder(args)

        #Just concate the embedding vectors and directly connec to linear classifier

        last_size = args.hidden_sz + (args.img_hidden_sz * args.num_image_embeds)
        self.clf = nn.ModuleList()

        for hidden in args.hidden:
            self.clf.append(nn.Linear(last_size, hidden))
            if args.include_bn:
                self.clf.append(nn.BatchNorm1d(hidden))
            self.clf.append(nn.ReLU())
            self.clf.append(nn.Dropout(args.dropout))
            last_size = hidden

        self.clf.append(nn.Linear(last_size, args.n_classes))

    def forward(self, txt, mask, segment, img):
        txt = self.txtenc(txt, mask, segment)
        img = self.imgenc(img)
        img = torch.flatten(img, start_dim=1)
        out = torch.cat([txt, img], -1)
        for layer in self.clf:
            out = layer(out)
        return out
