
"""
Dataset based on mmbt
"""
import os
import json
import random
import numpy as np
from PIL import Image
from fuzzywuzzy import fuzz
from collections import Counter

import math
import torch
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel, BertConfig
# from transformers.tokenization_albert import AlbertTokenizer
from random import randint, shuffle, choices
from random import random as rand
import pickle
import json
from collections import namedtuple
import torch
import torch.nn as nn
import unicodedata
from multiprocessing import Lock
from tqdm import tqdm
from transformers.modeling_utils import PreTrainedModel, ModuleUtilsMixin#, PreTrainedBertModel
from models.image import random_sample, Img_patch_embedding, fully_use_cnn
from sklearn import preprocessing
import re
from constants import *

def get_random_word(vocab_words):
    i = randint(0, len(vocab_words)-1)
    return vocab_words[i]

def batch_list_to_batch_tensors(batch):
    batch_tensors = []
    for x in zip(*batch):
        if isinstance(x[0], torch.Tensor):
            batch_tensors.append(torch.stack(x))
        else:
            batch_tensors.append(torch.tensor(x, dtype=torch.long))
    return batch_tensors

def truncate_img_txt(num_image_embeds, txt_tokens, max_seq_len):
    while True:
        if len(txt_tokens)  <= max_seq_len:
            break
        else:
            txt_tokens.pop()

class Pipeline():
    """ Pre-process Pipeline Class : callable """
    def __init__(self):
        super().__init__()
        self.mask_same_word = None
        self.skipgram_prb = None
        self.skipgram_size = None
    def __call__(self, instance):
        raise NotImplementedError


class CXRDataset(Dataset):
    """ Load image-sentence pairs """
    def __init__(self, args, data_path, tokenizer, batch_size, bi_uni_pipeline=[],  s2s_prob=0, bi_prob=1):
        super().__init__()
        self.args = args
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

        def random_pair_sampling(paired_img, paired_txt, tgt_label):
            if rand() > 0.5:
                paired_txt = self.tokenizer(paired_txt)
                return paired_img, paired_txt, tgt_label, 1
            else:
                for itr in range(10):
                    random_txt, random_label = get_random_line()
                    if fuzz.token_sort_ratio(tgt_label, random_label) != 100:
                        tokenized_random_txt = self.tokenizer(random_txt)
                        return paired_img, tokenized_random_txt, tgt_label, 0
                        break
                    else:
                        pass

        def sample_for_classification(paired_img, paired_txt, tgt_label):
            paired_txt = self.tokenizer(paired_txt)
            return paired_img, paired_txt, tgt_label, 1
        
        def get_random_line():
            rand_num = randint(0, len(img_dat) - 1)
            txt = img_dat[rand_num]['text']
            label = img_dat[rand_num]['label']
            return txt, label


        def get_labels_and_frequencies(data_labels):
            label_freqs = Counter()
            if type(data_labels[0]) == list:
                for label_row in data_labels:
                    label_freqs.update(label_row)
            else:
                label_freqs.update(data_labels)

            return list(label_freqs.keys()), label_freqs


        total_label = []
        for idx, src in enumerate(tqdm(img_dat)): # load each img path & txt
            src_tk = src['img']
            tgt_label = src['label']
            tgt_tk = src['text']
            if tgt_label == '':
                tgt_label = 'Others'
            else: pass

            if self.args.task =='classification':
                src_tk, ran_sampled_txt, random_label, random_itm_label  = sample_for_classification(src_tk, tgt_tk, tgt_label)
            else:
                src_tk, ran_sampled_txt, random_label, random_itm_label  = random_pair_sampling(src_tk, tgt_tk, tgt_label)

            random_label=random_label.split(', ')
            label_replace = [re.sub(r"'", "", itr) for itr in random_label]

            label = torch.zeros(15)
            label[[label_replace.index(tgt) for tgt in CONDITIONS]] = 1


            y_onehot = nn.functional.one_hot(torch.LongTensor(labels), num_classes=15)
            print("y_onehot",y_onehot)
            
            # encoder = preprocessing.LabelEncoder()
            # encoder.fit(label_replace)
            # label = encoder.transform(label_replace)
            # label = torch.LongTensor(label)
            # print("label",label)

            # input("STOP!")
            # input("STOP!")
            # le = preprocessing.OneHotEncoder()
            # print("random_label",random_label)
            # label = le.fit_transform(random_label)
            # print("one hot random_label",random_label)
            # label = torch.as_tensor(label)
            # print("torch random_label",random_label)

            # input("SOP!!!!!")
            total_label.append(label_replace)
            self.ex_list.append((src_tk, ran_sampled_txt, label_replace, random_itm_label))   

        labels, label_freqs = get_labels_and_frequencies(total_label)
        

        print('Load {0} documents'.format(len(self.ex_list)))

    def __len__(self):
        return len(self.ex_list)-1

    def __getitem__(self, idx):
        instance = self.ex_list[idx]
        proc = choices(self.bi_uni_pipeline, weights=[self.s2s_prob, self.bi_prob])[0]
        instance = proc(instance) # for img2txt tasks the answer is replace by dummy.
        return instance

    def __iter__(self):  # iterator to load data
        for __ in range(math.ceil(len(self.ex_list) / float(self.batch_size))):
            batch = []
            for __ in range(self.batch_size):
                idx = randint(0, len(self.ex_list)-1)
                batch.append(self.__getitem__(idx))
            # To Tensor
            total_label = self.ex_list[-1]
            
            yield batch_list_to_batch_tensors(batch)


# For encoder seq2seq model
class Preprocess4Seq2seq(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, tokenizer, transforms, mode, seq_len, num_image_embeds, new_segment_ids, bert_model, attn_1d=False):
        super().__init__()
        self.mode = mode
        # self.max_seq_len = max_seq_len  # 512
        self.seq_len = seq_len  # 253
        self.max_seq_len = seq_len + num_image_embeds  # 512 - 100(#img_embeds)
        self.transforms = transforms
        self.new_segment_ids = new_segment_ids
        self._tril_matrix = torch.tril(torch.ones((self.max_seq_len+3, self.max_seq_len+3), dtype=torch.long))
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self.num_image_embeds = num_image_embeds
        self.attn_1d = attn_1d

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


    def to_onehot(labels, n_categories=14, dtype=torch.float32):
        print("label",labels)
        # 먼저 숫자값으로 변환을 위해 LabelEncoder로 변환합니다. 
        encoder = preprocessing.LabelEncoder()
        encoder.fit(labels)
        labels = encoder.transform(labels)
        print("label",labels)

        labels = labels.reshape(-1,1)
        # 원-핫 인코딩을 적용합니다. 
        oh_encoder = preprocessing.OneHotEncoder()
        oh_encoder.fit(labels)
        oh_labels = oh_encoder.transform(labels)
        print('원-핫 인코딩 데이터')
        print(oh_labels.toarray())
        print('원-핫 인코딩 데이터 차원')
        print(oh_labels.shape)
        input("지금 여기임")

        one_hot_labels = torch.zeros(size=(n_categories), dtype=dtype)
        for i, label in enumerate(labels):
            # Subtract 1 from each LongTensor because your
            # indexing starts at 1 and tensor indexing starts at 0
            label = torch.LongTensor(label) - 1
            one_hot_labels[i] = one_hot_labels[i].scatter_(dim=0, index=label, value=1.)
        return one_hot_labels
        
    def __call__(self, instance):
        
        img_path, tokenized_sentence, label, is_aligned = instance[:4]


        image = Image.open(img_path)
        image = self.transforms(image)
        # print("is this tokenized already? txt", txt)
        # input("STOP!")
        
        # tokenized_sentence = self.tokenizer(txt)  # ['i','ate','an','apple'], no special token
        truncate_img_txt(self.num_image_embeds, tokenized_sentence, self.seq_len)

        if self.bert_model == "albert-base-v2":
            encoded_sentence = [self.vocab_stoi[w] if w in self.vocab_stoi else self.vocab_stoi["<unk>"]
                                for w in tokenized_sentence]
        elif self.bert_model == 'bert-base-uncased':
            encoded_sentence = [self.vocab_stoi[w] if w in self.vocab_stoi else self.vocab_stoi["[UNK]"]
                                for w in tokenized_sentence]  # [178, 8756, 1126, 12075]
        elif self.bert_model == 'ClinicalBERT':
            encoded_sentence = [self.vocab_stoi[w] if w in self.vocab_stoi else self.vocab_stoi["[UNK]"]
                                for w in tokenized_sentence]  # [178, 8756, 1126, 12075]
        elif self.bert_model == 'bert_small':
            encoded_sentence = [self.vocab_stoi[w] if w in self.vocab_stoi else self.vocab_stoi["[UNK]"]
                                for w in tokenized_sentence]  # [178, 8756, 1126, 12075]

        input_ids, txt_labels = self.random_word(encoded_sentence)

        # converted id from report 

        # input_ids = [self.vocab_stoi["[SEP]"]] + input_ids + [self.vocab_stoi["[SEP]"]]
        # txt_labels_t = [-100] + txt_labels + [-100]  # [SEP], txt, [SEP]  # 0
        # txt_labels_i = [-100] * (self.num_image_embeds + 1)  # 0

        # txt + [SEP]
        input_ids = input_ids + [self.vocab_stoi["[SEP]"]]
        txt_labels_t = txt_labels + [-100]  # [SEP], txt, [SEP]  # 0
        txt_labels_i = [-100] * (self.num_image_embeds + 2)  # [CLS] img [SEP]

        if self.bert_model == "albert-base-v2":
            padding = [self.vocab_stoi["<pad>"] for _ in range(self.seq_len - len(input_ids)+1)]  # 1 [SEP]
        elif self.bert_model == 'bert-base-uncased':
            padding = [self.vocab_stoi["[PAD]"] for _ in range(self.seq_len - len(input_ids)+1)]  # 1 [SEP]
            label_padding = [-100 for _ in range(self.seq_len - len(input_ids)+1)]  # 1 [SEP]
        elif self.bert_model == 'ClinicalBERT':
            padding = [self.vocab_stoi["[PAD]"] for _ in range(self.seq_len - len(input_ids)+1)] # 1 [SEP]
        elif self.bert_model == 'bert_small':
            padding = [self.vocab_stoi["[PAD]"] for _ in range(self.seq_len - len(input_ids)+1)]  # 1 [SEP]
            label_padding = [-100 for _ in range(self.seq_len - len(input_ids)+1)] # 1 [SEP]

        # TODO: padding set to 0(origin) or -100(for ignored in loss computing)

        # """ ###self-attention mask###
        extended_attn_masks = torch.zeros(self.max_seq_len+3, self.max_seq_len+3, dtype=torch.long)
        second_st, second_end = self.num_image_embeds+2, self.num_image_embeds+2+len(input_ids) #CLS, SEP + input_ids
        
        if self.mode == "s2s":
            # print("MODE", self.mode)
            extended_attn_masks[:, :self.num_image_embeds+2].fill_(1)
            extended_attn_masks[second_st:second_end, second_st:second_end].copy_(
                self._tril_matrix[:second_end-second_st, :second_end-second_st])
            # print("extended_attn_masks", extended_attn_masks)
            # print("size of extended_attn_masks", extended_attn_masks.size())
            attn_masks = extended_attn_masks

        elif self.mode == "bi" and self.attn_1d == False:
            # print("img + txt + special token :",self.max_seq_len+3) # -> 512가 max가 아님. 253+100+3 이 맥스임
            extended_attn_masks = torch.tensor([1] * (self.num_image_embeds+len(input_ids)+2) + [0] * len(padding), dtype=torch.long) \
                .unsqueeze(0).expand(self.max_seq_len+3, self.max_seq_len+3).clone()
            # print("extended_attn_masks", extended_attn_masks)
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
        # )
        
        segment = torch.tensor(segment)        

        
        sep_tok = [self.vocab_stoi["[SEP]"]]
        sep_tok = torch.tensor(sep_tok)

        # ITM
        # TODO: ITM negative sample
        # txt_itm, _, is_aligned = self.random_pair_sampling(idx)
        # input_ids_ITM = self.BertTokenizer(txt_itm, padding='max_length', max_length=self.max_seq_len)['input_ids']
        input_ids_ITM = encoded_sentence + [self.vocab_stoi["[SEP]"]]
        input_ids_ITM.extend(padding)

        is_aligned = torch.tensor(is_aligned)
        input_ids_ITM = torch.tensor(input_ids_ITM)

        # label = self.to_onehot(label)
        # label = np.float32(label)
        # label = torch.tensor(label)
        # print("label",label)
        # print("label",label.size())

        # print("input_ids",input_ids.size())
        # print("input_ids",label.size())
        # input("STOP!")

        return (cls_tok, input_ids, txt_labels, attn_masks, image, segment, is_aligned, input_ids_ITM, sep_tok, label)