"""
Dataset based on mmbt
"""
import os
import json
import random
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


from transformers import BertModel, BertTokenizer

def random_word(tokens, vocab_range, mask):
    """
        Masking some random tokens for Language Model task with probabilities as in
            the original BERT paper.
        tokens: list of int, tokenized sentence.
        vocab_range: for choosing a random word
        return : (list of int, list of int), masked tokens and related labels for
            LM prediction
        """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = mask

            # 10% randomly change token to random token
            # TODO: check, if or elif
            elif prob < 0.9:
                tokens[i] = random.choice(list(range(*vocab_range)))

            # rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            output_label.append(token)
        else:
            # no masking token(will be ignored by loss function later)
            output_label.append(-1)

    if all(o == -1 for o in output_label):
        # at least mask 1
        output_label[0] = tokens[0]
        tokens[0] = mask

    return tokens, output_label

class CXRDataset(Dataset):  # for both MLM and ITM
    def __init__(self, data_path, tokenizer, transforms, vocab, args):
        self.data = [json.loads(l) for l in open(data_path)]
        self.data_dir = os.path.dirname(data_path)
        self.tokenizer = tokenizer
        self.args = args
        self.vocab = vocab
        # TODO: Check max_len (cuz, img + txt)
        self.max_seq_len = args.max_seq_len  # 512
        self.max_seq_len -= args.num_image_embeds # 512 - 100(#img_embeds)
        self.transforms = transforms

        """
        tokenizer = BertTokenizer.from_pretrained('bert-based-uncased').tokenize
        vocab_stoi = tokenizer.vocab
        vocab_itos = tokenizer.ids_to_tokens
        vocab_len = len(vocab_itos)
        """
        self.tokenizer = tokenizer  # tokenizer = BertTokenizer.from_pretrained('bert-based-uncased').tokenize
        #self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased').tokenize

        # TODO(Done): stoi, itos tokenizer need to change
        self.BertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.vocab_stoi = self.BertTokenizer.vocab  #tokenizer = BertTokenizer.from_pretrained('bert-based-uncased')
        self.vocab_itos = self.BertTokenizer.ids_to_tokens
        self.vocab_len = len(self.vocab_itos)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        study_id = self.data[idx]['study_id']

        # MLM
        tokenized_sentence = self.tokenize(self.data[idx]['text'])  # ['i','ate','an','apple'], no special token
        encoded_sentence = [self.vocab_stoi[w] if w in self.vocab_stoi else self.vocab_stoi["[UNK]"]  # [178, 8756, 1126, 12075]
                            for w in tokenized_sentence]

        input_ids, txt_labels = random_word(encoded_sentence, self.vocab_len, self.vocab_stoi["[MASK]"])
        input_ids = [self.vocab_stoi["[CLS]"]] + input_ids + [self.vocab_stoi["[SEP]"]]


        txt_labels_t = [-1] + txt_labels + [-1]  # [CLS], txt, [SEP]
        txt_labels_p_i = [-1] * (self.max_seq_len - len(input_ids) + self.args.num_image_embeds) # [PAD]s, IMGs
        txt_labels = txt_labels_t + txt_labels_p_i

        # TODO(Done): Attention_mask(to distinguish padded or not), also applied to special tokens
        attn_masks_t = [1] * len(input_ids)
        attn_masks_i = [1] * len(self.args.num_image_embeds)

        padding = [self.vocab_stoi["[PAD]"] for _ in range(self.max_seq_len - len(input_ids))]
        input_ids.extend(padding)

        attn_masks_t.extend(padding)
        attn_masks = attn_masks_t + attn_masks_i  # attn_masks [1, 1, 1, 1, 0, 0, 1, 1, 1] -> Token, Pad, Img_feat

        # TODO: to distinguish txt or img
        #segment_label = ([0 for _ in range(self.max_seq_len)] + [1 for _ in range(self.args.num_image_embeds)])
        # segment only for txt. cuz in cxrbert.py img_tok is segment for img
        segment_label = [1 for _ in range(self.max_seq_len)]

        input_ids = torch.tensor(input_ids)
        txt_labels = torch.tensor(txt_labels)
        attn_masks = torch.tensor(attn_masks)
        segment_label = torch.tensor(segment_label)

        # TODO: Image input format, 1) whole and process at next step or 2) sample fiber at first.... 1)
        # TODO: Image to ToTensor()
        image = Image.open(os.path.join(self.data_dir, self.data[idx]['image]']))
        image = self.transforms(image)

        # ITM
        # TODO: ITM negative sample
        txt_itm, _, is_aligned = self.random_pair_sampling(idx)
        input_ids_ITM = self.BertTokenizer(txt_itm)['input_ids']

        return input_ids, txt_labels, attn_masks, image, segment_label, is_aligned, input_ids_ITM

    def random_pair_sampling(self, idx):
        _, txt, img = self.data[idx].keys()

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

# class ITMDataset(Dataset):
#     def __init__(self, data_path, tokenizer, transforms, vocab, args):
#         self.data = [json.loads(l) for l in open(data_path)]
#         self.data_dir = os.path.dirname(data_path)
#         self.tokenizer = tokenizer
#         self.args = args
#         self.vocab = vocab
#         # TODO: Check max_len (cuz, img + txt)
#         self.max_seq_len = args.max_seq_len
#         self.max_seq_len -= args.num_image_embeds
#         self.transforms = transforms
#
#         """
#         tokenizer = BertTokenizer.from_pretrained('bert-based-uncased').tokenize
#         vocab_stoi = tokenizer.vocab
#         vocab_itos = tokenizer.ids_to_tokens
#         vocab_len = len(vocab_itos)
#         """
#         self.tokenizer = tokenizer  # tokenizer = BertTokenizer.from_pretrained('bert-based-uncased').tokenize
#         # TODO: stoi, itos tokenizer need to change
#         self.vocab_stoi = self.tokenizer.vocab  # tokenizer = BertTokenizer.from_pretrained('bert-based-uncased')
#         self.vocab_itos = self.tokenizer.ids_to_tokens
#         self.vocab_len = len(self.vocab_itos)
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         # TODO: randomly generate negative samples (Aligned / Not aligned)
#         text = None
#         img_feat = None
#         label = None  # (Aligned or Not aligned)
#
#         # TODO: Image input format, whole and process at next step or sampled fiber at first(duplicated.)
#         image = Image.open(os.path.join(self.data_dir, self.data[idx]['image]']))
#         image = self.transforms(image)
#
#         return text, image, label
