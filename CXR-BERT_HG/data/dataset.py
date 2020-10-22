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
                tokens[i] = random.choice(list(range(vocab_range)))

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
        self.args = args
        self.data_dir = os.path.dirname(data_path)
        self.data = [json.loads(l) for l in open(data_path)]
        #self.vocab = vocab  # not in used, cuz self.BerTokenizer.vocab, .ids_to_tokens ~~~
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
        study_id = self.data[idx]['id']  # study_id

        # MLM
        tokenized_sentence = self.tokenizer(self.data[idx]['text'])  # ['i','ate','an','apple'], no special token
        encoded_sentence = [self.vocab_stoi[w] if w in self.vocab_stoi else self.vocab_stoi["[UNK]"]  # [178, 8756, 1126, 12075]
                            for w in tokenized_sentence]

        input_ids, txt_labels = random_word(encoded_sentence, self.vocab_len, self.vocab_stoi["[MASK]"])

        input_ids = [self.vocab_stoi["[CLS]"]] + input_ids + [self.vocab_stoi["[SEP]"]]
        txt_labels_t = [-1] + txt_labels + [-1]  # [CLS], txt, [SEP]

        txt_labels_p_i = [-1] * (self.max_seq_len - len(input_ids) + self.args.num_image_embeds) # [PAD]s, IMGs


        # TODO(Done): Attention_mask(to distinguish padded or not), also applied to special tokens
        attn_masks_t = [1] * len(input_ids)
        attn_masks_i = [1] * self.args.num_image_embeds

        padding = [self.vocab_stoi["[PAD]"] for _ in range(self.max_seq_len - len(input_ids))]
        input_ids.extend(padding)
        #txt_labels_t.extend(padding)
        attn_masks_t.extend(padding)

        txt_labels = txt_labels_t + txt_labels_p_i
        attn_masks = attn_masks_t + attn_masks_i  # attn_masks [1, 1, 1, 1, 0, 0, 1, 1, 1] -> Token, Pad, Img_feat

        # TODO: to distinguish txt or img
        #segment_label = ([0 for _ in range(self.max_seq_len)] + [1 for _ in range(self.args.num_image_embeds)])
        # segment only for txt. cuz in cxrbert.py img_tok is segment for img
        segment = [1 for _ in range(self.max_seq_len)]

        input_ids = torch.tensor(input_ids)
        txt_labels = torch.tensor(txt_labels)
        attn_masks = torch.tensor(attn_masks)
        segment = torch.tensor(segment)

        image = Image.open(os.path.join(self.data_dir, self.data[idx]['img']))
        image = self.transforms(image)

        # ITM
        # TODO: ITM negative sample
        txt_itm, _, is_aligned = self.random_pair_sampling(idx)
        input_ids_ITM = self.BertTokenizer(txt_itm, padding='max_length', max_length=self.max_seq_len)['input_ids']
        #input_ids_ITM.extend(padding)

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

        return input_ids, txt_labels, attn_masks, image, segment, is_aligned, input_ids_ITM


    def random_pair_sampling(self, idx):
        _, _, txt, img = self.data[idx].keys()

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
