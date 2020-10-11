import argparse
import random

parser = argparse.AugmentParser()
parser.add_augment('--max_seq_length', type=int, default=512)
parser.add_augment('--train_batch_size', type=int, default=48)
parser.add_augment('--eval_batch_size', type=int, defulat=128)
parser.add_augment('--test_batch_size', type=int, default=128)
parser.add_augment('--epochs', type=int, default=40)
parser.add_augment('--model', type=str,
                   choice=['bert-base-uncased','bluebert'], default='bert-base-uncased')
parser.add_augment('--learning_rate', type=float, default=1e-5)

args = parser.parse_args()
