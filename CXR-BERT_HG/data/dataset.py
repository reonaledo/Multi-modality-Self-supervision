"""
Dataset based on mmbt
"""
import os
import json
import random
import numpy as np
from PIL import Image
# from fuzzywuzzy import fuzz

import torch
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizer
from transformers.tokenization_albert import AlbertTokenizer


def truncate_img_txt(num_image_embeds, txt_tokens, max_seq_len):
    while True:
        total_length = num_image_embeds + len(txt_tokens) + 3  # for special tokens [CLS],[SEP],[SEP]
        if total_length <= max_seq_len:
            break
        else:
            txt_tokens.pop()


class CXRDataset(Dataset):  # for both MLM and ITM
    def __init__(self, data_path, tokenizer, transforms, args):
        self.args = args
        self.data_dir = os.path.dirname(data_path)
        self.data = [json.loads(l) for l in open(data_path)]

        self.max_seq_len = args.max_seq_len  # 512
        self.max_seq_len -= args.num_image_embeds  # 512 - 100(#img_embeds)
        self.transforms = transforms

        self.tokenizer = tokenizer  # tokenizer = BertTokenizer.from_pretrained('bert-based-uncased').tokenize

        if args.bert_model == "albert-base-v2":
            self.albert_tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
            self.vocab_stoi = self.albert_tokenizer.get_vocab()  # <unk>, <pad>
            self.vocab_len = len(self.vocab_stoi)  # 30000

        elif self.args.bert_model == 'bert-base-uncased':
            self.BertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.vocab_stoi = self.BertTokenizer.vocab
            self.vocab_len = len(self.vocab_stoi)  # 30522

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # MLM
        tokenized_sentence = self.tokenizer(self.data[idx]['text'])  # ['i','ate','an','apple'], no special token

        truncate_img_txt(self.args.num_image_embeds, tokenized_sentence, self.args.max_seq_len)

        if self.args.bert_model == "albert-base-v2":
            encoded_sentence = [self.vocab_stoi[w] if w in self.vocab_stoi else self.vocab_stoi["<unk>"]
                                for w in tokenized_sentence]
        elif self.args.bert_model == 'bert-base-uncased':
            encoded_sentence = [self.vocab_stoi[w] if w in self.vocab_stoi else self.vocab_stoi["[UNK]"]
                                for w in tokenized_sentence]  # [178, 8756, 1126, 12075]

        input_ids, txt_labels = self.random_word(encoded_sentence)

        input_ids = [self.vocab_stoi["[SEP]"]] + input_ids + [self.vocab_stoi["[SEP]"]]
        txt_labels_t = [-100] + txt_labels + [-100]  # [SEP], txt, [SEP]  # 0
        txt_labels_i = [-100] * (self.args.num_image_embeds + 1)  # 0

        attn_masks_t = [1] * len(input_ids)
        attn_masks_i = [1] * (self.args.num_image_embeds + 1)  # [CLS]

        if self.args.bert_model == "albert-base-v2":
            padding = [self.vocab_stoi["<pad>"] for _ in range(self.max_seq_len - len(input_ids) - 1)]  # [CLS]
        elif self.args.bert_model == 'bert-base-uncased':
            padding = [self.vocab_stoi["[PAD]"] for _ in range(self.max_seq_len - len(input_ids) - 1)]  # [CLS]
            label_padding = [-100 for _ in range(self.max_seq_len - len(input_ids) - 1)]  # [CLS]
        # TODO: padding set to 0(origin) or -100(for ignored in loss computing)
        input_ids.extend(padding)
        attn_masks_t.extend(padding)
        txt_labels_t.extend(label_padding)

        txt_labels = txt_labels_i + txt_labels_t
        attn_masks = attn_masks_i + attn_masks_t  # attn_masks [1, 1, 1, 1, 1, 1, 1, 1, 0, 0] -> Img_feat, Token, Pad

        segment = [1 for _ in range(self.max_seq_len - 1)]

        cls_tok = [self.vocab_stoi["[CLS]"]]
        cls_tok = torch.tensor(cls_tok)
        input_ids = torch.tensor(input_ids)
        txt_labels = torch.tensor(txt_labels)
        attn_masks = torch.tensor(attn_masks)
        segment = torch.tensor(segment)

        image = Image.open(os.path.join(self.data_dir, self.data[idx]['img']))  # .convert("RGB")
        image = self.transforms(image)

        # ITM
        # TODO: ITM negative sample
        txt_itm, _, is_aligned = self.random_pair_sampling(idx)
        # input_ids_ITM = self.BertTokenizer(txt_itm, padding='max_length', max_length=self.max_seq_len)['input_ids']
        input_ids_ITM = [self.vocab_stoi["[SEP]"]] + encoded_sentence + [self.vocab_stoi["[SEP]"]]
        input_ids_ITM.extend(padding)

        is_aligned = torch.tensor(is_aligned)
        input_ids_ITM = torch.tensor(input_ids_ITM)

        # through the collate_fn, mmbt return: txt, segment, mask, img, tgt = batch
        """
        Let max_seq = 8, num_img_embeds = 4
        sequence = 'I ate an apple'
        encoded_sequence = [CLS] I ate an apple [SEP] [PAD] [PAD]
                            -> 100, 3, 7, 101, 45, 105 , 0, 0 

        through the random_word(), return input_ids, txt_labels

        input_ids : for MLM and also only for TXT not IMG(sample random features), 
                    ex) 100, 3, 104[M], 101, 45, 105, 0, 0 //
        txt_labels : only for MLM
                    ex) -1, -1, 7, -1 -1 -1, -1(0), -1(0) // -1, -1, -1, -1
        attn_masks : 1, 1, 1, 1, 1, 1, 0, 0 // 1, 1, 1, 1
        image : full_img, _.jpg
        segment : 0, 0, 0, 0, 0, 0, 0, 0 // ( 1, 1, 1, 1) -> implemented in cxrbert.py img_tok, need to check
        is_aligned : Aligned(1) / Not aligned(0)
        input_ids_ITM : 100, 3, 7, 101, 45, 105, 0, 0 -> not masked, just encoded_sequence
        """
        # print('input_id_size:', input_ids.size())
        # print('txt_labels:', txt_labels.size())
        # print('attn_masks:', attn_masks.size())
        # print('segment:', segment.size())
        # print('is_aligned:', is_aligned)
        # print('input_ids_ITM:', input_ids_ITM.size())

        return cls_tok, input_ids, txt_labels, attn_masks, image, segment, is_aligned, input_ids_ITM

    def random_word(self, tokens):
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.vocab_stoi["[MASK]"]

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.randrange(self.vocab_len)

                # 10% randomly change token to current token
                # else:
                #     tokens[i] = token
                output_label.append(token)
            else:
                tokens[i] = token
                output_label.append(-100)  # 0

        if all(o == -100 for o in output_label):  # 0
            # at least one mask
            output_label[0] = tokens[0]
            tokens[0] = self.vocab_stoi["[MASK]"]

        return tokens, output_label

    def random_pair_sampling(self, idx):
        _, txt, img = self.data[idx].keys()
        # _, _, _, txt, img = self.data[idx].keys()

        d_txt = self.data[idx][txt]
        d_img = self.data[idx][img]

        if random.random() > 0.5:
            return d_txt, d_img, 1
        else:
            return self.get_random_line(), d_img, 0

    def get_random_line(self):
        rand_num = random.randint(0, len(self.data) - 1)
        txt = self.data[rand_num]['text']
        return txt
# ----------ITM for txt, img and labels---------------------------------------------
# def random_pair_sampling(self, idx):
#     _, _, label, txt, img = self.data[idx].keys()
#     d_label = self.data[idx][label]
#     d_txt = self.data[idx][txt]
#     d_img = self.data[idx][img]
#     if random.random() > 0.5:
#         return d_txt, d_img, 1
#     else:
#         for itr in range(10):
#             random_label = self.get_random_line()[1]
#             random_txt = self.get_random_line()[0]
#             if fuzz.token_sort_ratio(d_label, random_label) != 100:
#                 return random_txt, d_img, 0
#                 break
#             else:
#                 pass
#
# def get_random_line(self):
#     rand_num = random.randint(0, len(self.data) - 1)
#     txt = self.data[rand_num]['text']
#     label = self.data[rand_num]['label']
#     return txt, label
