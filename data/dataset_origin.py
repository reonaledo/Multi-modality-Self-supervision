"""
generate dataset
"""
import os
import json
import random
import numpy as np
from PIL import Image
from fuzzywuzzy import fuzz

import torch
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizer, AutoTokenizer
from transformers.tokenization_albert import AlbertTokenizer


def truncate_txt(txt_tokens, max_seq_len):
    while True:
        if len(txt_tokens) <= max_seq_len:
            break
        else:
            txt_tokens.pop()


class CXRDataset(Dataset):
    def __init__(self, data_path, tokenizer, transforms, args):
        self.args = args
        self.data_dir = os.path.dirname(data_path)
        self.data = [json.loads(l) for l in open(data_path)]

        self.max_seq_len = args.max_seq_len  # 512
        self.max_seq_len -= args.num_image_embeds  # 512 - #img_embeds

        self.seq_len = args.seq_len
        self.transforms = transforms

        self.total_len = self.seq_len + self.args.num_image_embeds + 3
        self._tril_matrix = torch.tril(torch.ones((self.total_len, self.total_len), dtype=torch.long))

        self.tokenizer = tokenizer  # tokenizer = BertTokenizer.from_pretrained('bert-based-uncased').tokenize

        if args.bert_model == "albert-base-v2":
            self.albert_tokenizer = AlbertTokenizer.from_pretrained(args.bert_model)
            self.vocab_stoi = self.albert_tokenizer.get_vocab()  # <unk>, <pad>
            self.vocab_len = len(self.vocab_stoi)  # 30000

        elif args.bert_model == "emilyalsentzer/Bio_ClinicalBERT":
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

        # elif args.bert_model == "load_pretrained_model":
        #     self.BertTokenizer = BertTokenizer.from_pretrained(args.init_model)
        #     self.vocab_stoi = self.BertTokenizer.vocab
        #     self.vocab_len = len(self.vocab_stoi)  # 30522

        else:  # BERT-base, small, tiny
            self.BertTokenizer = BertTokenizer.from_pretrained(args.bert_model)
            self.vocab_stoi = self.BertTokenizer.vocab
            self.vocab_len = len(self.vocab_stoi)  # 30522

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # MLM
        origin_txt, img_path, is_aligned, itm_prob = self.random_pair_sampling(idx)

        if self.args.img_channel == 3:
            image = Image.open(os.path.join(self.data_dir, img_path)).convert("RGB")
        elif self.args.img_channel == 1:
            image = Image.open(os.path.join(self.data_dir, img_path)).convert("RGB")

        image = self.transforms(image)

        tokenized_sentence = self.tokenizer(origin_txt)  # ['i','ate','an','apple'], no special token

        truncate_txt(tokenized_sentence, self.seq_len)

        if self.args.bert_model == "albert-base-v2":
            encoded_sentence = [self.vocab_stoi[w] if w in self.vocab_stoi else self.vocab_stoi["<unk>"]
                                for w in tokenized_sentence]
        else:
            encoded_sentence = [self.vocab_stoi[w] if w in self.vocab_stoi else self.vocab_stoi["[UNK]"]
                                for w in tokenized_sentence]  # [178, 8756, 1126, 12075]

        input_ids, txt_labels = self.random_word(encoded_sentence)

        if self.disturbing_mask:
            input_ids = input_ids + [self.vocab_stoi["[SEP]"]]
            txt_labels_t = [-100] + txt_labels + [-100]
            txt_labels_i = [-100] * (self.args.num_image_embeds + 2)
        else:
            input_ids = input_ids + [self.vocab_stoi["[SEP]"]]
            txt_labels_t = txt_labels + [-100]
            txt_labels_i = [-100] * (self.args.num_image_embeds + 2)

        attn_masks_t = [1] * len(input_ids)
        attn_masks_i = [1] * (self.args.num_image_embeds + 2)

        if self.args.bert_model == "albert-base-v2":
            padding = [self.vocab_stoi["<pad>"] for _ in range(self.seq_len - len(input_ids) + 1)]  # [SEP]
        else:
            padding = [self.vocab_stoi["[PAD]"] for _ in range(self.seq_len - len(input_ids) + 1)]  # [SEP]
            label_padding = [-100 for _ in range(self.seq_len - len(input_ids) + 1)]  # [SEP]

        input_ids.extend(padding)
        attn_masks_t.extend(padding)
        txt_labels_t.extend(label_padding)

        txt_labels = txt_labels_i + txt_labels_t
        attn_masks = attn_masks_i + attn_masks_t  # attn_masks [1, 1, 1, 1, 1, 1, 1, 1, 0, 0] -> Img_feat, Token, Pad

        segment = [1 for _ in range(self.seq_len + 1)]  # 2 [SEP]

        cls_tok = [self.vocab_stoi["[CLS]"]]
        cls_tok = torch.tensor(cls_tok)
        input_ids_tensor = torch.tensor(input_ids)
        txt_labels = torch.tensor(txt_labels)
        segment = torch.tensor(segment)
        is_aligned = torch.tensor(is_aligned)

        attn_1d = torch.tensor(attn_masks)

        full_attn = torch.tensor((attn_masks_i + attn_masks_t),
                                 dtype=torch.long).unsqueeze(0).expand(self.total_len, self.total_len).clone()

        extended_attn_masks = torch.zeros(self.total_len, self.total_len, dtype=torch.long)
        second_st, second_end = self.args.num_image_embeds + 2, self.args.num_image_embeds + 2 + len(input_ids)
        extended_attn_masks[:, :self.args.num_image_embeds + 2].fill_(1)
        extended_attn_masks[second_st:second_end, second_st:second_end].copy_(
            self._tril_matrix[:second_end - second_st, :second_end - second_st])
        s2s_attn = extended_attn_masks

        mixed_lst = [full_attn, s2s_attn]

        if self.args.Mixed:
            # print('Mixed attn mask')
            assert (self.args.s2s_prob + self.args.bi_prob) == 1.0
            attn_masks_tensor = random.choices(mixed_lst, weights=[self.args.bi_prob, self.args.s2s_prob])[0]
            # print(f'S2S {self.args.s2s_prob} vs Bi {self.args.bi_prob}')

        elif self.args.BAR_attn:
            # print('BAR_attn attn mask')
            extended_attn_masks[:self.args.num_image_embeds+2, :].fill_(1)
            attn_masks_tensor = extended_attn_masks

        elif self.args.disturbing_mask:
            baseline_attn = torch.zeros(self.total_len, self.total_len, dtype=torch.long)
            baseline_attn[:self.args.num_image_embeds + 2, :self.args.num_image_embeds + 2].fill_(1)
            baseline_attn[self.args.num_image_embeds + 2:, self.args.num_image_embeds + 2:].fill_(1)
            attn_masks_tensor = baseline_attn

        else:
            if self.args.attn_1d:
                # print('1d_bidirectional attn mask')
                attn_masks_tensor = attn_1d  # '1d attention mask'

            else:
                # print('full_bidirecitonal attn mask')
                attn_masks_tensor = full_attn  # 'Full attention mask'

        sep_tok = [self.vocab_stoi["[SEP]"]]
        sep_tok = torch.tensor(sep_tok)

        return cls_tok, input_ids_tensor, txt_labels, attn_masks_tensor, image, segment, is_aligned, sep_tok, itm_prob

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
        _, _, label, txt, img = self.data[idx].keys()  # id, txt, img

        d_label = self.data[idx][label]
        d_txt = self.data[idx][txt]
        d_img = self.data[idx][img]

        itm_prob = random.random()

        if itm_prob > 0.5:
            return d_txt, d_img, 1, itm_prob
        else:
            for itr in range(300):
                random_txt, random_label = self.get_random_line()
                if fuzz.token_sort_ratio(d_label, random_label) != 100:
                    return random_txt, d_img, 0, itm_prob
                    break
                else:
                    pass

    def get_random_line(self):
        rand_num = random.randint(0, len(self.data) - 1)
        txt = self.data[rand_num]['text']
        label = self.data[rand_num]['label']
        return txt, label
