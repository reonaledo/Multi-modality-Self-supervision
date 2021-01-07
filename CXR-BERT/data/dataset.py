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
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel, BertConfig
# from transformers.tokenization_albert import AlbertTokenizer
from random import randint, shuffle, choices
from random import random as rand
import pickle
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers.modeling_utils import PreTrainedModel#, PreTrainedBertModel
from models.image import random_sample, Img_patch_embedding, fully_use_cnn
from utils.loader_utils import get_random_word, batch_list_to_batch_tensors, Pipeline


def truncate_txt(txt_tokens, max_seq_len):
    while True:
        if len(txt_tokens) <= max_seq_len:
            break
        else:
            txt_tokens.pop()

class CXRDataset(Dataset):
    """ Load image-sentence pairs """
    def random_pair_sampling(paired_img, paired_txt, tgt_label):
            if rand() > 0.5:
                return paired_img, paired_txt, tgt_label, 1
            else:
                for itr in range(len(img_dat)):
                    random_txt, random_label = get_random_line()
                    if fuzz.token_sort_ratio(tgt_label, random_label) != 100:
                        return paired_img, random_txt, random_label, 0
                        break
                    else:pass

    def __init__(self, data_path, tokenizer, batch_size, bi_uni_pipeline=[],  s2s_prob=0, bi_prob=1):
        super().__init__()
        self.tokenizer = tokenizer  # tokenize function
        self.bi_uni_pipeline = bi_uni_pipeline
        self.batch_size = batch_size
        self.s2s_prob = s2s_prob
        self.bi_prob = bi_prob
        print(' seq2seq {} vs bidirectional {}'.format(self.s2s_prob, self.bi_prob))
        assert(self.s2s_prob + self.bi_prob == 1)

        # read the file into memory
        self.ex_list = []
        img_dat = [json.loads(l) for l in open(data_path)]
        print('Loading {0} valid JPG IDs!'.format(len(img_dat)))
                                    
        def get_random_line():
            rand_num = randint(0, len(img_dat) - 1)
            txt = img_dat[rand_num]['text']
            label = img_dat[rand_num]['label']
            return txt, label

        for idx, src in enumerate(tqdm(img_dat)): # load each img path & txt
            src_tk = src['img']
            tgt_label = src['label']
            tgt_tk = src['text']
            
            src_tk, ran_sampled_txt, random_label, random_itm_label  = random_pair_sampling(src_tk, tgt_tk, tgt_label)
            self.ex_list.append((src_tk, ran_sampled_txt, random_label, random_itm_label))                        

        print('Load {0} documents'.format(len(self.ex_list)))
    

    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):
        print("len(self.ex_list)",len(self.ex_list))
        print("len(self.ex_list[idx] ",len(self.ex_list[idx]))
        input("STOP!!!")
        instance = self.ex_list[idx]        
        proc = choices(self.bi_uni_pipeline, weights=[self.s2s_prob, self.bi_prob])[0]
        instance = proc(instance)
        return instance

    def __iter__(self):  # iterator to load data
        print("math.ceil(len(self.ex_list) / float(self.batch_size))",math.ceil(len(self.ex_list) / float(self.batch_size)))
        for __ in range(math.ceil(len(self.ex_list) / float(self.batch_size))):
            batch = []
            for __ in range(self.batch_size):
                idx = randint(0, len(self.ex_list)-1)
                print("idx",idx)
                batch.append(self.__getitem__(idx))
                print("batch",batch)
                input("STOP!!!")

            # To Tensor
            yield batch_list_to_batch_tensors(batch)


# For encoder seq2seq model
class Preprocess4Seq2seq(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, tokenizer, transforms, mode, seq_len, num_image_embeds, new_segment_ids, bert_model, mask_prob, attn_1d=False, masked_attnt_dropout=0):
        super().__init__()
        self.mode = mode
        # self.max_seq_len = max_seq_len  # 512
        self.seq_len = seq_len  # 253
        self.max_seq_len = seq_len + num_image_embeds  # 512 - 100(#img_embeds)
        self.transforms = transforms
        self.new_segment_ids = new_segment_ids
        self._tril_matrix = torch.tril(torch.ones((self.max_seq_len+3, self.max_seq_len+3), dtype=torch.long))
        self.random_matrix = torch.randint(0,2,(self.max_seq_len+3, self.max_seq_len+3), dtype=torch.long)


        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self.num_image_embeds = num_image_embeds
        self.attn_1d = attn_1d
        self.masked_attnt_dropout = masked_attnt_dropout
        self.mask_prob = mask_prob

        self.new_segment_ids = new_segment_ids
        assert mode in ("s2s", "bi")

        if self.mode == 's2s': 
            self.task_idx = 3   # relax projection layer for different tasks
        else: 
            self.task_idx = 0
            

        if self.bert_model == 'bert-base-uncased':
            self.BertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.vocab_stoi = self.BertTokenizer.vocab
            self.vocab_len = len(self.vocab_stoi)  # 30522

        elif self.bert_model == 'ClinicalBERT':
            self.BertTokenizer =   AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            self.vocab_stoi = self.BertTokenizer.vocab
            self.vocab_len = len(self.vocab_stoi)  # 30522

        elif self.bert_model == 'google/bert_uncased_L-4_H-512_A-8':
            self.BertTokenizer =   AutoTokenizer.from_pretrained('google/bert_uncased_L-4_H-512_A-8')
            self.vocab_stoi = self.BertTokenizer.vocab
            self.vocab_len = len(self.vocab_stoi)  # 30522

    def random_word(self, tokens):
        output_label = []
        for i, token in enumerate(tokens):
            prob = random.random()
            if prob < self.mask_prob: #0.15:
                prob /= self.mask_prob #0.15
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

    def __call__(self, instance):
        
        # for text
        img_path, origin_txt, label, is_aligned = instance[:4]

        # print("we will load text!")

        # for label
        # img_path, _, tokenized_sentence, is_aligned = instance[:4]
        # print("we will load label!")

        
        image = Image.open(img_path)
        image = self.transforms(image)
        tokenized_sentence = self.tokenizer(origin_txt)  # ['i','ate','an','apple'], no special token

        truncate_txt(tokenized_sentence, self.seq_len)


        if self.bert_model == "albert-base-v2":
            encoded_sentence = [self.vocab_stoi[w] if w in self.vocab_stoi else self.vocab_stoi["<unk>"]
                                for w in tokenized_sentence]
        else:
            encoded_sentence = [self.vocab_stoi[w] if w in self.vocab_stoi else self.vocab_stoi["[UNK]"]
                                for w in tokenized_sentence]  # [178, 8756, 1126, 12075]
    
        input_ids, txt_labels = self.random_word(encoded_sentence)

        # converted id from report 

        # input_ids = [self.vocab_stoi["[SEP]"]] + input_ids + [self.vocab_stoi["[SEP]"]]
        # txt_labels_t = [-100] + txt_labels + [-100]  # [SEP], txt, [SEP]  # 0
        # txt_labels_i = [-100] * (self.num_image_embeds + 1)  # 0

        input_ids = input_ids + [self.vocab_stoi["[SEP]"]]
        txt_labels_t = txt_labels + [-100]  # [SEP], txt, [SEP]  # 0
        txt_labels_i = [-100] * (self.num_image_embeds + 2)  # [CLS] img [SEP]

        if self.bert_model == "albert-base-v2":
            padding = [self.vocab_stoi["<pad>"] for _ in range(self.seq_len - len(input_ids)+1)]  # 1 [SEP]
        elif self.bert_model == 'bert-base-uncased':
            padding = [self.vocab_stoi["[PAD]"] for _ in range(self.seq_len - len(input_ids)+1)]  # 1 [SEP]
            label_padding = [-100 for _ in range(self.seq_len - len(input_ids)+1)]  # 1 [SEP]


        # TODO: padding set to 0(origin) or -100(for ignored in loss computing)

        # """ ###self-attention mask###
        extended_attn_masks = torch.zeros(self.max_seq_len+3, self.max_seq_len+3, dtype=torch.long)
        second_st, second_end = self.num_image_embeds+2, self.num_image_embeds+2+len(input_ids) #CLS, SEP + input_ids
        
        if self.mode == "s2s":
            extended_attn_masks[:, :self.num_image_embeds+2].fill_(1)
            extended_attn_masks[second_st:second_end, second_st:second_end].copy_(
                self._tril_matrix[:second_end-second_st, :second_end-second_st])
            attn_masks = extended_attn_masks

        elif self.mode == "bi" and self.attn_1d == False and self.masked_attnt_dropout==False:
            extended_attn_masks = torch.tensor([1] * (self.num_image_embeds+len(input_ids)+2) + [0] * len(padding), dtype=torch.long) \
                .unsqueeze(0).expand(self.max_seq_len+3, self.max_seq_len+3).clone()
            attn_masks = extended_attn_masks
        
        elif self.mode == "bi" and self.attn_1d == False and self.masked_attnt_dropout:
            extended_attn_masks[:, :self.num_image_embeds+2].fill_(1)
            ################################### if text key에 대해서만 random matrix (50%) 적용 할때,##################################
            # extended_attn_masks[second_st:second_end, second_st:second_end].copy_(
            #     self.random_matrix[:second_end-second_st, :second_end-second_st])
            ################################### if 전체 key에 대해서 random matrix (50%) 적용 할때,##################################
            # extended_attn_masks[:second_end, :second_end].copy_(
            #     self.random_matrix[:second_end, :second_end])
            ##################################text key에도 attention이 걸리도록 새로 추가해준 부분.#################################
            extended_attn_masks[second_st:second_end, second_st:second_end].copy_(
                self._tril_matrix[:second_end-second_st, :second_end-second_st])
            extended_attn_masks[:self.num_image_embeds+2, :].fill_(1)
            ##############################################################################################################
            attn_masks = extended_attn_masks

        elif self.mode == "bi" and self.attn_1d == True:
            attn_masks_t = [1] * len(input_ids)
            attn_masks_i = [1] * (self.num_image_embeds + 2)  # [CLS], [SEP]
            attn_masks_t.extend(padding)
            attn_masks = attn_masks_i + attn_masks_t  # attn_masks [1, 1, 1, 1, 1, 1, 1, 1, 0, 0] -> Img_feat, Token, Pad
            attn_masks = torch.tensor(attn_masks)        

        ###############"""

        input_ids.extend(padding)
        txt_labels_t.extend(label_padding)
        txt_labels = txt_labels_i + txt_labels_t # for Masked Language Modeling
        

        if self.new_segment_ids:
            if self.mode == 's2s':
                # segment_ids = [4] * (len(tokens_a)+2) + [5] * (len(tokens_b)+1)
                segment = [5 for _ in range(self.seq_len+1)] # 2 [SEP] 
            elif self.mode == 'bi':
                segment = [1 for _ in range(self.seq_len+1)] # 2 [SEP]
        else:
            segment = [1 for _ in range(self.seq_len+1)] # 2 [SEP]

        cls_tok = [self.vocab_stoi["[CLS]"]]
        cls_tok = torch.tensor(cls_tok)
        input_ids = torch.tensor(input_ids)
        txt_labels = torch.tensor(txt_labels)
        segment = torch.tensor(segment)        
        is_aligned = torch.tensor(is_aligned)

        sep_tok = [self.vocab_stoi["[SEP]"]]
        sep_tok = torch.tensor(sep_tok)

        return (cls_tok, input_ids, txt_labels, attn_masks, image, segment, is_aligned, sep_tok)