from misc.configs import *

import torch
from torch.utils.data import DataLoader

from transformers import BertTokenizer

IMG_DIM = 768
TXT_DIM = 768

class InputFeatures(object):

    def __init__(self, input_ids, segment_ids_txt, input_mask, visual, segment_ids_img):
        self.input_ids = input_ids
        self.segment_ids_txt = segment_ids_txt
        self.input_mask = input_mask
        self.visual = visual
        self.segment_ids_img = segment_ids_img

def get_tokenizer(model):
    if model == 'bert-base-uncased':
        return BertTokenizer.from_pretrained(model)
    elif model == 'bluebert':
        return BertTokenizer.from_pretrained(model)
    else:
        raise ValueError("Expected 'bert-base-uncased' or 'bluebert', but received {}".format(model))

# TODO: PixelBert, N-random sampling
def img_Embedding(input):
    visual = input
    return visual

# TODO: segment_ids_img
def prepare_bert_input(txt, visual):

    tokenizer = get_tokenizer(args.model)

    features = []
    encoded_txt = tokenizer(txt, padding='max_length', max_length=args.max_seq_length, return_tensors='pt') #512

    input_ids = encoded_txt['input_ids']
    segment_ids_txt = encoded_txt['token_type_ids']
    input_mask = encoded_txt['attention_mask']

    segment_ids_img = torch.tensor([1] * len(input_ids))

    assert len(input_ids) == args.max_seq_length
    assert len(segment_ids_txt) == args.max_seq_length
    assert len(input_mask) == args.max_seq_length
    assert visual.shape[0] == args.max_seq_length
    assert len(segment_ids_img) == args.max_seq_length

    features.append(
        InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids_txt=segment_ids_txt,
            visual=visual,
            segment_ids_img=segment_ids_img
        )
    )
    return features

def get_dataset(txt, img):
    pass

class CXRDataset(torch.utils.data.Dataset):
    def __init__(self, txt, img):
        pass

train_data = CXRDataset()
eval_data = CXRDataset()
test_data = CXRDataset()