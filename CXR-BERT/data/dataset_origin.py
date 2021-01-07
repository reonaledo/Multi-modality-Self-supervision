"""
Dataset based on mmbt
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
# from transformers.tokenization_albert import AlbertTokenizer


def truncate_txt(txt_tokens, max_seq_len):
    while True:
        if len(txt_tokens) <= max_seq_len:
            break
        else:
            txt_tokens.pop()

class CXRDataset_origin(Dataset):
    def __init__(self, data_path, tokenizer, transforms, args):
        self.args = args
        self.data_dir = os.path.dirname(data_path)
        self.data = [json.loads(l) for l in open(data_path)]
        self.mode = args.mode
        self.attn_1d = args.attn_1d
        self.masked_attnt_dropout = args.masked_attnt_dropout
    
        # self.max_seq_len = args.max_seq_len  # 512
        # self.max_seq_len -= args.num_image_embeds  # 512 - #img_embeds

        self.seq_len = args.seq_len
        self._tril_matrix = torch.tril(torch.ones((self.seq_len + args.num_image_embeds+3, self.seq_len + args.num_image_embeds+3), dtype=torch.long))
        self.random_matrix = torch.randint(0,2,(self.seq_len + args.num_image_embeds+3, self.seq_len + args.num_image_embeds+3), dtype=torch.long)

        self.transforms = transforms

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
        origin_txt, img_path, is_aligned = self.random_pair_sampling(idx)

        # if self.args.img_channel == 3:
        image = Image.open(os.path.join(self.data_dir, img_path))
        # elif self.args.img_channel == 1:
        #     image = Image.open(os.path.join(self.data_dir, img_path)).convert("RGB")

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

        # input_ids = [self.vocab_stoi["[SEP]"]] + input_ids + [self.vocab_stoi["[SEP]"]]
        # txt_labels_t = [-100] + txt_labels + [-100]  # [SEP], txt, [SEP]  # 0
        # txt_labels_i = [-100] * (self.args.num_image_embeds + 1)  # 0

        input_ids = input_ids + [self.vocab_stoi["[SEP]"]]
        txt_labels_t = txt_labels + [-100]  # [SEP], txt, [SEP]  # 0
        txt_labels_i = [-100] * (self.args.num_image_embeds + 2)  # 0 [CLS] img [SEP]

        # attn_masks_t = [1] * len(input_ids)
        # attn_masks_i = [1] * (self.args.num_image_embeds + 1)  # [CLS]


        # if self.args.bert_model == "albert-base-v2":
        #     padding = [self.vocab_stoi["<pad>"] for _ in range(self.max_seq_len - len(input_ids) - 1)]  # [CLS]
        # else:
        #     padding = [self.vocab_stoi["[PAD]"] for _ in range(self.max_seq_len - len(input_ids) - 1)]  # 0, [CLS]
        #     label_padding = [-100 for _ in range(self.max_seq_len - len(input_ids) - 1)]  # [CLS]
        if self.args.bert_model == "albert-base-v2":
            padding = [self.vocab_stoi["<pad>"] for _ in range(self.seq_len - len(input_ids) + 1)]  # [SEP]
        else:
            padding = [self.vocab_stoi["[PAD]"] for _ in range(self.seq_len - len(input_ids) + 1)]  # [SEP]
            label_padding = [-100 for _ in range(self.seq_len - len(input_ids) + 1)]  # [SEP]

        # """ ###self-attention mask###
        extended_attn_masks = torch.zeros(self.args.num_image_embeds+self.seq_len+3, self.args.num_image_embeds+self.seq_len+3, dtype=torch.long)
        second_st, second_end = self.args.num_image_embeds+2, self.args.num_image_embeds+2+len(input_ids) #CLS, SEP + input_ids
        
        
        if self.mode == "s2s":
            extended_attn_masks[:, :self.args.num_image_embeds+2].fill_(1)
            extended_attn_masks[second_st:second_end, second_st:second_end].copy_(
                self._tril_matrix[:second_end-second_st, :second_end-second_st])
            attn_masks = extended_attn_masks

        elif self.mode == "bi" and self.attn_1d == False and self.masked_attnt_dropout==False:
            extended_attn_masks = torch.tensor([1] * (self.args.num_image_embeds+len(input_ids)+2) + [0] * len(padding), dtype=torch.long) \
               .unsqueeze(0).expand(self.args.num_image_embeds+self.seq_len+3, self.args.num_image_embeds+self.seq_len+3).clone()
            attn_masks = extended_attn_masks
        
        elif self.mode == "bi" and self.attn_1d == False and self.masked_attnt_dropout:
            extended_attn_masks[:, :self.args.num_image_embeds+2].fill_(1)
            ################################### if text key에 대해서만 random matrix (50%) 적용 할때,##################################
            # extended_attn_masks[second_st:second_end, second_st:second_end].copy_(
            #     self.random_matrix[:second_end-second_st, :second_end-second_st])
            ################################### if 전체 key에 대해서 random matrix (50%) 적용 할때,##################################
            # extended_attn_masks[:second_end, :second_end].copy_(
            #     self.random_matrix[:second_end, :second_end])
            ##################################text key에도 attention이 걸리도록 새로 추가해준 부분.#################################
            extended_attn_masks[second_st:second_end, second_st:second_end].copy_(
                self._tril_matrix[:second_end-second_st, :second_end-second_st])
            extended_attn_masks[:self.args.num_image_embeds+2, :].fill_(1)
            ##############################################################################################################
            attn_masks = extended_attn_masks

        elif self.mode == "bi" and self.attn_1d == True:
            attn_masks_t = [1] * len(input_ids)
            attn_masks_i = [1] * (self.args.num_image_embeds + 2)  # [CLS], [SEP]
            # attn_masks_t = [1] * len(input_ids)
            # attn_masks_i = [1] * (self.num_image_embeds + 2)  # [CLS], [SEP]
            attn_masks_t.extend(padding)
            attn_masks = attn_masks_i + attn_masks_t  # attn_masks [1, 1, 1, 1, 1, 1, 1, 1, 0, 0] -> Img_feat, Token, Pad
            attn_masks = torch.tensor(attn_masks)



        input_ids.extend(padding)
        # attn_masks_t.extend(padding)
        txt_labels_t.extend(label_padding)

        txt_labels = txt_labels_i + txt_labels_t
        # attn_masks = attn_masks_i + attn_masks_t  # attn_masks [1, 1, 1, 1, 1, 1, 1, 1, 0, 0] -> Img_feat, Token, Pad

        # segment = [1 for _ in range(self.max_seq_len - 1)]
        segment = [1 for _ in range(self.seq_len + 1)]  # 2 [SEP]

        cls_tok = [self.vocab_stoi["[CLS]"]]
        cls_tok = torch.tensor(cls_tok)
        input_ids = torch.tensor(input_ids)
        txt_labels = torch.tensor(txt_labels)
        segment = torch.tensor(segment)
        is_aligned = torch.tensor(is_aligned)

        sep_tok = [self.vocab_stoi["[SEP]"]]
        sep_tok = torch.tensor(sep_tok)

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

        return cls_tok, input_ids, txt_labels, attn_masks, image, segment, is_aligned, sep_tok

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
        # _, _, txt, img = self.data[idx].keys()  # id, label, txt, img

        d_label = self.data[idx][label]
        d_txt = self.data[idx][txt]
        d_img = self.data[idx][img]

        if random.random() > 0.5:
            return d_txt, d_img, 1
        else:
            # random_txt, random_label = self.get_random_line()
            # return random_txt, d_img, 0

            for itr in range(100):
                random_txt, random_label = self.get_random_line()
                if fuzz.token_sort_ratio(d_label, random_label) != 100:
                    return random_txt, d_img, 0
                    break
                else:
                    pass

    def get_random_line(self):
        rand_num = random.randint(0, len(self.data) - 1)
        txt = self.data[rand_num]['text']
        label = self.data[rand_num]['label']
        return txt, label
# ----------ITM for txt, img and labels---------------------------------------------
# def random_pair_sampling(self, idx):
#     _, label, txt, img = self.data[idx].keys()
#     d_label = self.data[idx][label]
#     d_txt = self.data[idx][txt]
#     d_img = self.data[idx][img]
#     if random.random() > 0.5:
#         return d_txt, d_img, 1
#     else:
#         for itr in range(10):
#             random_label = self.get_random_line()[1]
#             random_txt = self.get_random_line()[0]
#             if fuzz.token_sort_ratio(d_label, random_label) != 100:  # order not matter, ignore punctuation
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
