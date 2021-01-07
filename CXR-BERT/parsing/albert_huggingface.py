import torch
import torch.nn as nn

from transformers import AlbertConfig, AlbertModel, AlbertTokenizer, BertTokenizer, BertConfig
from transformers.tokenization_albert import AlbertTokenizer
# from transformers.modeling_albert import AlbertModel


config = AlbertConfig('albert-base-v2')  # load albert-base-v2 configuration
model = AlbertModel.from_pretrained('albert-base-v2', return_dict=True)
configuration = model.config
albert_embeds = model.embeddings
albert_encoder = model.encoder

# t = albert_embeds.word_embeddings(torch.LongTensor([0]))

albert_tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
seq = 'Pneumothorax'  # 'I ate an apple'
tokenized_seq = albert_tokenizer(seq, padding='max_length')
# print(tokenizer.get_vocab())
print(len(tokenized_seq['input_ids']))
print(albert_tokenizer.tokenize(seq))
for i in albert_tokenizer.tokenize(seq):
    print(i)
    print(albert_tokenizer.convert_tokens_to_ids(i))
# outputs = model(**tokenized_seq)
# print(outputs.last_hidden_state)

"""
self.BertTokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
self.vocab_stoi = self.BertTokenizer.vocab
self.vocab_itos = self.BertTokenizer.ids_to_tokens
self.vocab_len = len(self.vocab_itos)

encoded_sentence = [self.vocab_stoi[w] if w in self.vocab_stoi else self.vocab_stoi["[UNK]"]  # [178, 8756, 1126, 12075]
                            for w in tokenized_sentence]
"""

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
seq = 'I ate an apple'
tokenized_seq = bert_tokenizer.tokenize(seq)
print(tokenized_seq)
vocab_stoi = bert_tokenizer.vocab
vocab_itos = bert_tokenizer.ids_to_tokens  # actually not used ...
vocab_len = len(vocab_itos)

bert_vocab = bert_tokenizer.get_vocab()

albert_tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
al_tokenized_seq = albert_tokenizer.tokenize(seq)
albert_vocab = albert_tokenizer.get_vocab()  # !!! vocab_stoi = bert_tokenizer.vocab  ,,,, <unk>, <pad>
albert_vocab_len = len(albert_vocab)  # vocab_len = len(vocab_itos)


# print(vocab_stoi)
# print(bert_tokenizer.get_vocab())

Albert_config = AlbertConfig('albert-base-v1')
Albert2_config = AlbertConfig('albert-base-v2')

Bert_config = BertConfig('bert-base-uncased')

print(Albert_config)
print(Albert2_config)
print(Bert_config)