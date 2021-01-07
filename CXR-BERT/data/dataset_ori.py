
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
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel
# from transformers.tokenization_albert import AlbertTokenizer


def truncate_img_txt(num_image_embeds, txt_tokens, max_seq_len):
    while True:
        if len(txt_tokens)  <= max_seq_len:
            break
        else:
            txt_tokens.pop()
            
class CXRDataset(Dataset):  # for both MLM and ITM
    def __init__(self, data_path, tokenizer, transforms, args):
        self.args = args
        self.data_dir = os.path.dirname(data_path)
        self.data = [json.loads(l) for l in open(data_path)]
        self.mode = args.mode
        # self.max_seq_len = args.max_seq_len  # 512
        self.seq_len = args.seq_len  # 253
        self.max_seq_len = args.seq_len + args.num_image_embeds  # 512 - 100(#img_embeds)
        self.transforms = transforms
        self.new_segment_ids = args.new_segment_ids

        self._tril_matrix = torch.tril(torch.ones((self.max_seq_len+3, self.max_seq_len+3), dtype=torch.long))

        self.tokenizer = tokenizer  # tokenizer = BertTokenizer.from_pretrained('bert-based-uncased').tokenize

        # if args.bert_model == "albert-base-v2":
        #     self.albert_tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        #     self.vocab_stoi = self.albert_tokenizer.get_vocab()  # <unk>, <pad>
        #     self.vocab_len = len(self.vocab_stoi)  # 30000

        if self.args.bert_model == 'bert-base-uncased':
            self.BertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.vocab_stoi = self.BertTokenizer.vocab
            self.vocab_len = len(self.vocab_stoi)  # 30522

        elif self.args.bert_model == 'ClinicalBERT':
            self.BertTokenizer =   AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            self.vocab_stoi = self.BertTokenizer.vocab
            self.vocab_len = len(self.vocab_stoi)  # 30522

        elif self.args.bert_model == 'google/bert_uncased_L-4_H-512_A-8':
            self.BertTokenizer =   AutoTokenizer.from_pretrained('google/bert_uncased_L-4_H-512_A-8')
            self.vocab_stoi = self.BertTokenizer.vocab
            self.vocab_len = len(self.vocab_stoi)  # 30522

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # MLM
        origin_txt, img_path, label, is_aligned = self.random_pair_sampling(idx)

        # image = Image.open(os.path.join(self.data_dir, self.data[idx]['img']))  #.convert("RGB")
        image = Image.open(os.path.join(self.data_dir, img_path))  # .convert("RGB")
        image = self.transforms(image)

        # tokenized_sentence = self.tokenizer(self.data[idx]['text'])  # ['i','ate','an','apple'], no special token
        # tokenized_sentence = self.tokenizer(origin_txt)  # ['i','ate','an','apple'], no special token
        tokenized_sentence = self.tokenizer(origin_txt)  # ['i','ate','an','apple'], no special token
        # print("tokenized_sentence",tokenized_sentence)
        # input("STOP!!")

        truncate_img_txt(self.args.num_image_embeds, tokenized_sentence, self.seq_len)

        if self.args.bert_model == "albert-base-v2":
            encoded_sentence = [self.vocab_stoi[w] if w in self.vocab_stoi else self.vocab_stoi["<unk>"]
                                for w in tokenized_sentence]
        elif self.args.bert_model == 'bert-base-uncased':
            encoded_sentence = [self.vocab_stoi[w] if w in self.vocab_stoi else self.vocab_stoi["[UNK]"]
                                for w in tokenized_sentence]  # [178, 8756, 1126, 12075]
        elif self.args.bert_model == 'ClinicalBERT':
            encoded_sentence = [self.vocab_stoi[w] if w in self.vocab_stoi else self.vocab_stoi["[UNK]"]
                                for w in tokenized_sentence]  # [178, 8756, 1126, 12075]
        elif self.args.bert_model == 'bert_small':
            encoded_sentence = [self.vocab_stoi[w] if w in self.vocab_stoi else self.vocab_stoi["[UNK]"]
                                for w in tokenized_sentence]  # [178, 8756, 1126, 12075]

        print("encoded_sentence",encoded_sentence)
        input("STop!!")

        input_ids, txt_labels = self.random_word(encoded_sentence)

        input_ids = [self.vocab_stoi["[SEP]"]] + input_ids + [self.vocab_stoi["[SEP]"]]
        txt_labels_t = [-100] + txt_labels + [-100]  # [SEP], txt, [SEP]  # 0
        txt_labels_i = [-100] * (self.args.num_image_embeds + 1)  # 0

        #### 아래랑 중복여부 체크 해야 함 중복 되어야 함 같아야함 ####
        attn_masks_t = [1] * len(input_ids)
        attn_masks_i = [1] * (self.args.num_image_embeds + 1)  # [CLS]
        ##########################################################

        if self.args.bert_model == "albert-base-v2":
            padding = [self.vocab_stoi["<pad>"] for _ in range(self.seq_len - len(input_ids)+2)]  # 2 [SEP]
        elif self.args.bert_model == 'bert-base-uncased':
            padding = [self.vocab_stoi["[PAD]"] for _ in range(self.seq_len - len(input_ids)+2)]  # 2 [SEP]
            label_padding = [-100 for _ in range(self.seq_len - len(input_ids)+2)]  # 2 [SEP]
        elif self.args.bert_model == 'ClinicalBERT':
            padding = [self.vocab_stoi["[PAD]"] for _ in range(self.seq_len - len(input_ids)+2)] # 2 [SEP]
        elif self.args.bert_model == 'bert_small':
            padding = [self.vocab_stoi["[PAD]"] for _ in range(self.seq_len - len(input_ids)+2)]  # 2 [SEP]
            label_padding = [-100 for _ in range(self.seq_len - len(input_ids)+2)] # 2 [SEP]

        # TODO: padding set to 0(origin) or -100(for ignored in loss computing)

        # """ ###self-attention mask###
        extended_attn_masks = torch.zeros(self.max_seq_len+3, self.max_seq_len+3, dtype=torch.long)
        second_st, second_end = self.args.num_image_embeds+2, self.args.num_image_embeds+len(input_ids)+1 #CLS, SEP,  #CLS
        
        if self.mode == "s2s":
            # print("MODE", self.mode)
            extended_attn_masks[:, :self.args.num_image_embeds+2].fill_(1)
            extended_attn_masks[second_st:second_end, second_st:second_end].copy_(
                self._tril_matrix[:second_end-second_st, :second_end-second_st])
            # print("extended_attn_masks", extended_attn_masks)
            # print("size of extended_attn_masks", extended_attn_masks.size())
            attn_masks = extended_attn_masks
            
        elif self.mode == "bi":
            # print("img + txt + special token :",self.max_seq_len+3) # -> 512가 max가 아님. 253+100+3 이 맥스임
            extended_attn_masks = torch.tensor([1] * (self.args.num_image_embeds+len(input_ids)+1) + [0] * len(padding), dtype=torch.long) \
                .unsqueeze(0).expand(self.max_seq_len+3, self.max_seq_len+3).clone()
            # print("extended_attn_masks", extended_attn_masks)
            # print("size of extended_attn_masks", extended_attn_masks.size())
            attn_masks = extended_attn_masks
        else:
            attn_masks_t.extend(padding)
            attn_masks = attn_masks_i + attn_masks_t  # attn_masks [1, 1, 1, 1, 1, 1, 1, 1, 0, 0] -> Img_feat, Token, Pad
            attn_masks = torch.tensor(attn_masks)
        ###############"""
        
        input_ids.extend(padding)
        txt_labels_t.extend(label_padding)

        txt_labels = txt_labels_i + txt_labels_t
        # 


        if self.new_segment_ids:
            if self.mode == 's2s':
                # segment_ids = [4] * (len(tokens_a)+2) + [5] * (len(tokens_b)+1)
                segment = [5 for _ in range(self.seq_len+2)] # 2 [SEP] 
            elif self.mode == 'bi':
                segment = [1 for _ in range(self.seq_len+2)] # 2 [SEP]
        else:
            segment = [1 for _ in range(self.seq_len+2)] # 2 [SEP]

        cls_tok = [self.vocab_stoi["[CLS]"]]
        cls_tok = torch.tensor(cls_tok)
        input_ids = torch.tensor(input_ids)
        txt_labels = torch.tensor(txt_labels)
        # )
        
        segment = torch.tensor(segment)        

        # ITM
        # TODO: ITM negative sample
        # txt_itm, _, is_aligned = self.random_pair_sampling(idx)
        # input_ids_ITM = self.BertTokenizer(txt_itm, padding='max_length', max_length=self.max_seq_len)['input_ids']
        input_ids_ITM = [self.vocab_stoi["[SEP]"]] + encoded_sentence + [self.vocab_stoi["[SEP]"]]
        input_ids_ITM.extend(padding)

        is_aligned = torch.tensor(is_aligned)
        input_ids_ITM = torch.tensor(input_ids_ITM)

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
        _, _, label, txt, img = self.data[idx].keys()
        d_label = self.data[idx][label]
        d_txt = self.data[idx][txt]
        d_img = self.data[idx][img]
        if random.random() > 0.5:
            return d_txt, d_img, d_label, 1
        else:
            for itr in range(10):
                random_label = self.get_random_line()[1]
                random_txt = self.get_random_line()[0]
                if fuzz.token_sort_ratio(d_label, random_label) != 100:
                    return random_txt, d_img, random_label, 0
                    break
                else:
                    pass
    
    def get_random_line(self):
        rand_num = random.randint(0, len(self.data) - 1)
        txt = self.data[rand_num]['text']
        label = self.data[rand_num]['label']
        return txt, label