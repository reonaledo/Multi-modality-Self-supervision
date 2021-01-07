from random import randint, shuffle, choices
from random import random as rand
import pickle
import math
import json
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from vlp.loader_utils import get_random_word, batch_list_to_batch_tensors, Pipeline

import torchvision.transforms as transforms
from PIL import Image
# https://stackoverflow.com/questions/12984426/python-pil-ioerror-image-file-truncated-with-big-images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import imghdr
import numpy as np
import h5py
from tqdm import tqdm
from fuzzywuzzy import fuzz

# print("args.cnn",args.cnn)


def get_random_word(vocab_words):
    i = randint(0, len(vocab_words)-1)
    return vocab_words[i]


def batch_list_to_batch_tensors(batch):
    batch_tensors = []
    # print("batch", batch)
    # input("STOP!!")
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

        def random_pair_sampling(paired_img, paired_txt, tgt_label):
            if rand() > 0.5:
                paired_txt = self.tokenizer(paired_txt)
                return paired_img, paired_txt, tgt_label, 1
            else:
                for itr in range(10):
                    random_txt, random_label = get_random_line()
                    if fuzz.token_sort_ratio(tgt_label, random_label) != 100:
                        tokenized_random_txt = self.tokenizer(random_txt)
                        return paired_img, tokenized_random_txt, random_label, 0
                        break
                    else:
                        pass
                
        def get_random_line():
            rand_num = randint(0, len(img_dat) - 1)
            txt = img_dat[rand_num]['text']
            label = img_dat[rand_num]['label']
            return txt, label

        for idx, src in enumerate(tqdm(img_dat)): # load each img path & txt
            src_tk = src['img']
            tgt_label = src['label']
            tgt_tk = src['text']
            if tgt_label == []:
                tgt_label = 'Others'
            else: pass
            
            src_tk, ran_sampled_txt, random_label, random_itm_label  = random_pair_sampling(src_tk, tgt_tk, tgt_label)
            self.ex_list.append((src_tk, ran_sampled_txt, random_label, random_itm_label))                        

        print('Load {0} documents'.format(len(self.ex_list)))

    def __len__(self):
        return len(self.ex_list)

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
            yield batch_list_to_batch_tensors(batch)



class Preprocess4Seq2seqDecoder(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, tokenizer, transforms, max_len, max_txt_length, new_segment_ids=False, mode="s2s", len_vis_input=None):
        super().__init__()
        self.tokenizer = tokenizer  # function from token to token index 
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.new_segment_ids = new_segment_ids
        self.task_idx = 3   # relax projection layer for different tasks
        self.mode = mode
        if self.mode != "s2s":
            raise ValueError("Invalid mode for seq2seq decode: %s" % self.mode)
        self.max_txt_length = max_txt_length
        self.len_vis_input = len_vis_input
        self.transforms = transforms
        self.res_Normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __call__(self, instance):
        img_path, max_a_len = instance[:2]
        
        tokens_a = ['[UNK]'] * self.len_vis_input
        
        # Add Special Tokens
        padded_tokens_a = ['[CLS]'] + tokens_a + ['[SEP]']

        assert len(padded_tokens_a) <= max_a_len + 2
        if max_a_len + 2 > len(padded_tokens_a):
            padded_tokens_a += ['[PAD]'] * \
                (max_a_len + 2 - len(padded_tokens_a))
        assert len(padded_tokens_a) == max_a_len + 2
        max_len_in_batch = min(self.max_txt_length +
                               max_a_len + 2, self.max_len)

        tokens = padded_tokens_a

        if self.new_segment_ids:
            segment_ids = [4]*(len(padded_tokens_a)) \
                + [5]*(max_len_in_batch - len(padded_tokens_a))
        else:
            segment_ids = [0]*(len(padded_tokens_a)) \
                + [1]*(max_len_in_batch - len(padded_tokens_a))

        position_ids = []
        for i in range(len(tokens_a) + 2):
            position_ids.append(i)
        for i in range(len(tokens_a) + 2, max_a_len + 2):
            position_ids.append(0)
        for i in range(max_a_len + 2, max_len_in_batch):
            position_ids.append(i - (max_a_len + 2) + len(tokens_a) + 2)
        # Token Indexing
        input_ids = self.tokenizer(tokens)

        # Zero Padding
        input_mask = torch.zeros(
            max_len_in_batch, max_len_in_batch, dtype=torch.long)

        input_mask[:, :len(tokens_a)+2].fill_(1)
        second_st, second_end = len(padded_tokens_a), max_len_in_batch

        input_mask[second_st:second_end, second_st:second_end].copy_(
            self._tril_matrix[:second_end-second_st, :second_end-second_st])

        img = Image.open(img_path)
        img = self.transforms(img)

            
        return (input_ids, segment_ids, position_ids, input_mask, self.task_idx, img)