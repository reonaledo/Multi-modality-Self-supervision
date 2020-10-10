"""
MLM dataset
"""

import random

import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip

from .data import *

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
            if prob < 0.9:
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

class MlmDataset():
    pass


def mlm_collate():
    pass