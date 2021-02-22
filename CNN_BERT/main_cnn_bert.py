"""
CNN + BERT
    1. CNN(resnet50), GAP
    2. BERT Encoder, CLS token
"""
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
import wandb
import random
import argparse
import numpy as np
from PIL import Image
from datetime import datetime


import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizer, AutoTokenizer
from transformers.modeling_auto import AutoConfig, AutoModel
from transformers.modeling_bert import BertConfig, BertModel, BertPreTrainedModel

from data.helper import get_transforms

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def truncate_txt(txt_tokens, max_seq_len):
    while True:
        if len(txt_tokens) <= max_seq_len:
            break
        else:
            txt_tokens.pop()

class CNN_BERT_Dataset(Dataset):
    def __init__(self, data_path, tokenizer, transforms, args):
        self.args = args
        self.data_dir = os.path.dirname(data_path)
        self.data = [json.loads(line) for line in open(data_path)]

        self.num_image_embeds = args.num_image_embeds
        self.seq_len = args.seq_len
        self.transforms = transforms

        self.tokenizer = tokenizer  # tokenizer = BertTokenizer.from_pretrained('bert-based-uncased').tokenize

        if args.bert_model == "emilyalsentzer/Bio_ClinicalBERT":
            self.BertTokenizer = AutoTokenizer.from_pretrained(args.bert_model)
            self.vocab_stoi = self.BertTokenizer.vocab
            self.vocab_len = len(self.vocab_stoi)  # 28996

        elif args.bert_model == "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12":
            self.BertTokenizer = AutoTokenizer.from_pretrained(args.bert_model)
            self.vocab_stoi = self.BertTokenizer.vocab
            self.vocab_len = len(self.vocab_stoi)  # 30522

        elif args.bert_model == "bert-small-scratch":
            self.BertTokenizer = BertTokenizer.from_pretrained("google/bert_uncased_L-4_H-512_A-8")
            self.vocab_stoi = self.BertTokenizer.vocab
            self.vocab_len = len(self.vocab_stoi)  # 30522

        elif args.bert_model == "bert-base-scratch":
            self.BertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.vocab_stoi = self.BertTokenizer.vocab
            self.vocab_len = len(self.vocab_stoi)  # 30522

        else:  # BERT-base, small, tiny
            self.BertTokenizer = BertTokenizer.from_pretrained(args.bert_model)
            self.vocab_stoi = self.BertTokenizer.vocab
            self.vocab_len = len(self.vocab_stoi)  # 30522

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        _, _, label, txt, img = self.data[idx].keys()
        origin_txt = self.data[idx][txt]
        img_path = self.data[idx][img]

        tokenized_sentence = self.tokenizer(origin_txt)  # ['i','ate','an','apple'], no special token

        truncate_txt(tokenized_sentence, self.seq_len)

        encoded_sentence = [self.vocab_stoi[w] if w in self.vocab_stoi else self.vocab_stoi["[UNK]"]
                            for w in tokenized_sentence]  # [178, 8756, 1126, 12075]

        input_ids = [self.vocab_stoi["[CLS]"]] + encoded_sentence + [self.vocab_stoi["[SEP]"]]

        attn_masks = [1] * len(input_ids)

        padding = [self.vocab_stoi["[PAD]"] for _ in range(self.seq_len - len(input_ids) + 2)]

        input_ids.extend(padding)
        attn_masks.extend(padding)

        segment = [1 for _ in range(self.seq_len + 2)]

        input_ids = torch.tensor(input_ids)
        attn_masks = torch.tensor(attn_masks)
        segment = torch.tensor(segment)

        if self.args.img_channel == 3:
            image = Image.open(os.path.join(self.data_dir, img_path))
        elif self.args.img_channel == 1:
            image = Image.open(os.path.join(self.data_dir, img_path)).convert("RGB")

        image = self.transforms(image)

        return input_ids, attn_masks, segment, image

class IMG_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        model = torchvision.models.resnet50(pretrained=True)
        modules = list(model.children())[:-2]
        self.model = nn.Sequential(*modules)
        self.pool = F.adaptive_avg_pool2d

    def forward(self, x):
        out = self.model(x)  # 224x224: torch.Size([16, 2048, 7, 7])
        # out = self.pool(out, (1, 1))  # torch.Size([16, 2048, 1, 1])
        out = torch.mean(out.view(out.size(0), out.size(1), -1), dim=2)
        return out

class TXT_Encoder(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.args = args

        if args.bert_model == "emilyalsentzer/Bio_ClinicalBERT":
            bert = AutoModel.from_pretrained(args.bert_model)
        elif args.bert_model == "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12":
            bert = AutoModel.from_pretrained(args.bert_model)
        elif args.bert_model == "bert-small-scratch":
            config = BertConfig.from_pretrained("google/bert_uncased_L-4_H-512_A-8")
            bert = BertModel(config)
        elif args.bert_model == "bert-base-scratch":
            config = BertConfig.from_pretrained("bert-base-uncased")
            bert = BertModel(config)
        else:
            bert = BertModel.from_pretrained(args.bert_model)  # bert-base-uncased, small, tiny

        self.txt_embeddings = bert.embeddings

        self.encoder = bert.encoder
        self.pooler = bert.pooler

    def get_extended_attn_mask(self, attn_mask):
        if attn_mask.dim() == 2:
            extended_attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
        elif attn_mask.dim() == 3:
            extended_attn_mask = attn_mask.unsqueeze(1)
        else:
            raise NotImplementedError
        extended_attn_mask = extended_attn_mask.to(dtype=torch.float16)
        extended_attn_mask = (1.0 - extended_attn_mask) * - 10000.0

        return extended_attn_mask

    def forward(self, input_txt, attn_mask, segment):
        extended_attn_mask = self.get_extended_attn_mask(attn_mask)
        txt_embed_out = self.txt_embeddings(input_txt, segment)  # bsz, seq_len, hsz. inputs: bsz, seq_len
        encoded_layers = self.encoder(txt_embed_out, extended_attn_mask, output_hidden_states=False)
        return self.pooler(encoded_layers[-1])

class CNN_BERT(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)

        img_txt_hiddens = args.img_hidden_sz + args.hidden_size

        self.txt_enc = TXT_Encoder(config, args)
        self.img_enc = IMG_Encoder()
        self.linear = nn.Linear(img_txt_hiddens, 2)

    def forward(self, input_txt, attn_mask, segment, input_img):
        txt_cls = self.txt_enc(input_txt, attn_mask, segment)
        img_cls = self.img_enc(input_img)

        cls = torch.cat([img_cls, txt_cls], 1)

        return self.linear(cls)
