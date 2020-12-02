"""
Dataset based on mmbt
"""
import math
import json
import random
from tqdm import tqdm
from PIL import Image
from fuzzywuzzy import fuzz
from random import random as rand
from random import randint, choices

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, AutoTokenizer
from transformers.tokenization_albert import AlbertTokenizer


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


def truncate_img_txt(txt_tokens, seq_len):
    while True:
        if len(txt_tokens) <= seq_len:
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
    def __init__(self, data_path, tokenizer, batch_size, bi_uni_pipeline=[], s2s_prob=0, bi_prob=1):
        super().__init__()
        self.tokenizer = tokenizer  # BertTokenizer.from_pretrained().tokenize
        self.batch_size = batch_size
        self.bi_uni_pipeline = bi_uni_pipeline
        self.s2s_prob = s2s_prob
        self.bi_prob = bi_prob
        print(f'seq2seq {self.s2s_prob} vs bidirectional {self.bi_prob}')
        assert (self.s2s_prob + self.bi_prob == 1)

        # read the file into memory
        self.ex_list = []
        img_dat = [json.loads(l) for l in open(data_path)]
        print('Loading {0} valid JPG IDs!'.format(len(img_dat)))

        # def random_pair_sampling(paired_img, paired_txt, tgt_label):
        #     if rand() > 0.5:
        #         paired_txt = self.tokenizer(paired_txt)
        #         return paired_img, paired_txt, tgt_label, 1
        #     else:
        #         for itr in range(10):
        #             random_txt, random_label = get_random_line()
        #             if fuzz.token_sort_ratio(tgt_label, random_label) != 100:
        #                 tokenized_random_txt = self.tokenizer(random_txt)
        #                 return paired_img, tokenized_random_txt, random_label, 0
        #                 break
        #             else:
        #                 pass
        #
        # def get_random_line():
        #     rand_num = randint(0, len(img_dat) - 1)
        #     txt = img_dat[rand_num]['text']
        #     label = img_dat[rand_num]['label']
        #     return txt, label
        #
        # for idx, src in enumerate(tqdm(img_dat)):  # load each img path & txt
        #     src_tk = src['img']
        #     tgt_label = src['label']
        #     tgt_tk = src['text']
        #     if tgt_label == []:
        #         tgt_label = 'Others'
        #     else:
        #         pass
        #     # src_tk: img
        #     src_tk, ran_sampled_txt, random_label, random_itm_label = random_pair_sampling(src_tk, tgt_tk, tgt_label)
        #     self.ex_list.append((src_tk, ran_sampled_txt, random_label, random_itm_label))

        def random_pair_sampling_origin(paired_img, paired_txt):
            if random.random() > 0.5:
                paired_txt = self.tokenizer(paired_txt)
                return paired_img, paired_txt, 1
            else:
                paired_txt = self.tokenizer(get_random_line_origin())
                return paired_img, paired_txt, 0

        def get_random_line_origin():
            rand_num = random.randint(0, len(img_dat) - 1)
            txt = img_dat[rand_num]['text']
            return txt

        for idx, src in enumerate(tqdm(img_dat)):  # load each img path & txt
            src_tk = src['img']
            tgt_tk = src['text']
            # src_tk: img
            src_tk, ran_sampled_txt, random_itm_label = random_pair_sampling_origin(src_tk, tgt_tk)
            self.ex_list.append((src_tk, ran_sampled_txt, random_itm_label))

        print('Load {0} documents'.format(len(self.ex_list)))

    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):
        instance = self.ex_list[idx]
        proc = choices(self.bi_uni_pipeline, weights=[self.s2s_prob, self.bi_prob])[0]  # list to element
        instance = proc(instance)  # for img2txt tasks the answer is replace by dummy.
        return instance

    def __iter__(self):  # iterator to load data
        for __ in range(math.ceil(len(self.ex_list) / float(self.batch_size))):
            batch = []
            for __ in range(self.batch_size):
                idx = randint(0, len(self.ex_list) - 1)
                batch.append(self.__getitem__(idx))
            # To Tensor
            yield batch_list_to_batch_tensors(batch)


# For encoder seq2seq model
class Preprocess4Seq2seq(Pipeline):
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, tokenizer, transforms, mode, seq_len, num_image_embeds, new_segment_ids, bert_model):
        super().__init__()
        self.mode = mode
        # self.max_seq_len = max_seq_len  # 512
        self.seq_len = seq_len  # 253, fix !
        self.max_seq_len = seq_len + num_image_embeds  # 253 + (100/256)
        self._tril_matrix = torch.tril(torch.ones((self.max_seq_len + 3, self.max_seq_len + 3), dtype=torch.long))
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.bert_model = bert_model
        self.new_segment_ids = new_segment_ids
        self.num_image_embeds = num_image_embeds

        assert mode in ("s2s", "bi", "ori")

        if bert_model == "albert-base-v2":
            self.albert_tokenizer = AlbertTokenizer.from_pretrained(bert_model)
            self.vocab_stoi = self.albert_tokenizer.get_vocab()  # <unk>, <pad>
            self.vocab_len = len(self.vocab_stoi)  # 30000

        elif bert_model == "emilyalsentzer/Bio_ClinicalBERT":
            self.BertTokenizer = AutoTokenizer.from_pretrained(bert_model)
            self.vocab_stoi = self.BertTokenizer.vocab
            self.vocab_len = len(self.vocab_stoi)  # 28996

        elif bert_model == "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12":
            self.BertTokenizer = AutoTokenizer.from_pretrained(bert_model)
            self.vocab_stoi = self.BertTokenizer.vocab
            self.vocab_len = len(self.vocab_stoi)  # 30522

        else:  # BERT-base, small, tiny
            self.BertTokenizer = BertTokenizer.from_pretrained(bert_model)
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

    def __call__(self, instance):
        # img_path, tokenized_sentence, label, is_aligned = instance[:4]
        img_path, tokenized_sentence, is_aligned = instance[:3]
        image = Image.open(img_path)
        image = self.transforms(image)

        truncate_img_txt(tokenized_sentence, self.seq_len)

        if self.bert_model == "albert-base-v2":
            encoded_sentence = [self.vocab_stoi[w] if w in self.vocab_stoi else self.vocab_stoi["<unk>"]
                                for w in tokenized_sentence]
        else:
            encoded_sentence = [self.vocab_stoi[w] if w in self.vocab_stoi else self.vocab_stoi["[UNK]"]
                                for w in tokenized_sentence]  # [178, 8756, 1126, 12075]

        input_ids, txt_labels = self.random_word(encoded_sentence)

        input_ids = [self.vocab_stoi["[SEP]"]] + input_ids + [self.vocab_stoi["[SEP]"]]
        txt_labels_t = [-100] + txt_labels + [-100]  # [SEP], txt, [SEP]  # 0
        txt_labels_i = [-100] * (self.num_image_embeds + 1)  # 0 [CLS]

        if self.bert_model == "albert-base-v2":
            padding = [self.vocab_stoi["<pad>"] for _ in range(self.seq_len - len(input_ids) + 2)]  # 2 [SEP]
        else:
            padding = [self.vocab_stoi["[PAD]"] for _ in range(self.seq_len - len(input_ids) + 2)]  # 2 [SEP]
            label_padding = [-100 for _ in range(self.seq_len - len(input_ids) + 2)]  # 2 [SEP]

        # ###self-attention mask###
        extended_attn_masks = torch.zeros(self.max_seq_len + 3, self.max_seq_len + 3, dtype=torch.long)
        second_st, second_end = self.num_image_embeds + 2, self.num_image_embeds + len(input_ids) + 1  # CLS, SEP,  #CLS
        if self.mode == "s2s":
            # print("MODE", self.mode)
            extended_attn_masks[:, :self.num_image_embeds + 2].fill_(1)
            extended_attn_masks[second_st:second_end, second_st:second_end].copy_(
                self._tril_matrix[:second_end - second_st, :second_end - second_st])
            # print("extended_attn_masks", extended_attn_masks)
            # print("size of extended_attn_masks", extended_attn_masks.size())
            attn_masks = extended_attn_masks

        elif self.mode == "bi":
            # print("img + txt + special token :",self.max_seq_len+3) # -> 512가 max가 아님. 253+100+3 이 맥스임
            extended_attn_masks = torch.tensor([1] * (self.num_image_embeds + len(input_ids) + 1) + [0] * len(padding),
                                               dtype=torch.long) \
                .unsqueeze(0).expand(self.max_seq_len + 3, self.max_seq_len + 3).clone()  # torch.size([356,356])
            # print("extended_attn_masks", extended_attn_masks)
            # print("size of extended_attn_masks", extended_attn_masks.size())
            attn_masks = extended_attn_masks

        elif self.mode == "ori":
            attn_masks_t = [1] * len(input_ids)
            attn_masks_i = [1] * (self.num_image_embeds + 1)  # [CLS]
            attn_masks_t.extend(padding)
            attn_masks = attn_masks_i + attn_masks_t  # attn_masks [1, 1, 1, 1, 1, 1, 1, 1, 0, 0] -> Img_feat, Token, Pad
            attn_masks = torch.tensor(attn_masks)

        input_ids.extend(padding)
        txt_labels_t.extend(label_padding)
        txt_labels = txt_labels_i + txt_labels_t

        if self.new_segment_ids:
            if self.mode == 's2s':
                # segment_ids = [4] * (len(tokens_a)+2) + [5] * (len(tokens_b)+1)
                segment = [5 for _ in range(self.seq_len + 2)]  # 2 [SEP]
            elif self.mode == 'bi':
                segment = [1 for _ in range(self.seq_len + 2)]  # 2 [SEP]
        else:
            segment = [1 for _ in range(self.seq_len + 2)]  # 2 [SEP]

        cls_tok = [self.vocab_stoi["[CLS]"]]
        cls_tok = torch.tensor(cls_tok)
        input_ids = torch.tensor(input_ids)
        txt_labels = torch.tensor(txt_labels)

        segment = torch.tensor(segment)

        is_aligned = torch.tensor(is_aligned)

        return (cls_tok, input_ids, txt_labels, attn_masks, image, segment, is_aligned)
