"""
test to loading the trained cxr-model
"""

import os
import argparse
from datetime import date, time, datetime

import torch
import torch.nn as nn
import torch.optim as optim

from models. retrieval import CXRBertForImageRetrieval
from transformers import PreTrainedModel, BertConfig, AutoConfig   # BertModel, PreTrainedModel, AutoModel
from transformers.modeling_bert import BertModel
from transformers.modeling_utils import PreTrainedModel
from models.cxrbert import CXRBERT
# prepare model

'''
"epoch": epoch,
# "state_dict": self.model.module.state_dict(),  # DataParallel
"state_dict": self.model.state_dict(),
"optimizer": self.optimizer.state_dict(),
'''
def main(args):

    # file = os.path.join('/home/ubuntu/HG/Multi-modality-Self-supervision/CXR-BERT_HG/output/1025', "cxrbert_ep4.pt")
    # print(file)
    # checkpoint = torch.load(file)
    # print(checkpoint.keys())
    # epoch = checkpoint["epoch"]
    # model = checkpoint["state_dict"]
    # optimizer = checkpoint["optimizer"]
    # print(epoch)
    # print(optimizer.keys())
    # print(model.keys())

    # bluebert = '/home/ubuntu/HG/Multi-modality-Self-supervision/CXR-BERT_HG/bluebert'
    # # config = BertConfig.from_pretrained(bluebert)
    # #print(config)
    # model_state_dict = torch.load(os.path.join(bluebert, 'pytorch_model.bin'))
    # # print(model_state_dict)

    # for param in model_state_dict:
    #     print(param)
    # path = '/home/ubuntu/HG/Multi-modality-Self-supervision/CXR-BERT_HG/output/save_pretrained_test'
    # cxr = PreTrainedModel.from_pretrained(pretrained_model_name_or_path=path)
    # print(cxr)




    # torch.Size([bsz, input_txt_len])
    # input_txt = torch.Tensor(3,20)
    # segment = torch.Tensor(3, 20)

    # Working correctly
    # config = BertConfig.from_pretrained('/home/ubuntu/HG/cxr-bert/BlueBERT')
    # model_state_dict = torch.load('/home/ubuntu/HG/cxr-bert/BlueBERT/pytorch_model.bin')
    # bert = BertModel.from_pretrained('/home/ubuntu/HG/cxr-bert/BlueBERT', state_dict=model_state_dict, config=config)
    #
    # bert_embed = bert.embeddings.word_embeddings(torch.LongTensor([0]))
    # print(bert_embed)

    # bert_embed = bert.embeddings.word_embeddings(torch.LongTensor([0])) !!!!!!

    # config = BertConfig.from_pretrained('/home/ubuntu/HG/cxr-bert/output/2020-11-02 23:33:28.988900')
    # model_state_dict = torch.load('/home/ubuntu/HG/cxr-bert/output/2020-11-02 23:33:28.988900/pytorch_model.bin')
    # bert = BertModel.from_pretrained('/home/ubuntu/HG/cxr-bert/output/2020-11-02 23:33:28.988900', state_dict=model_state_dict, config=config)
    # bert_embed = bert.embeddings.word_embeddings(torch.LongTensor([0]))
    # print(bert_embed)

    config = AutoConfig.from_pretrained('/home/ubuntu/HG/cxr-bert/output/2020-11-03 02:07:42.236882')
    print(config)
    # model_state_dict = torch.load('/home/ubuntu/HG/cxr-bert/output/2020-11-03 02:07:42.236882/pytorch_model.bin')
    # CXRBERT.from_pretrained('/home/ubuntu/HG/cxr-bert/output/2020-11-03 02:07:42.236882', config=config, args=args)


    # config = BertConfig.from_pretrained(args.init_model)

    # model_state_dict = torch.load('/home/ubuntu/HG/cxr-bert/output/2020-11-03 00:00:05.110129/pytorch_model.bin')
    # bert = BertModel.from_pretrained('/home/ubuntu/HG/cxr-bert/output/2020-11-03 00:00:05.110129', state_dict=model_state_dict, config=config)
    # bert_embed = bert.embeddings.word_embeddings(torch.LongTensor([0]))
    # print(bert_embed)




    """
     .from_pretrained(config=BertConfig.from_pretrained(<pretrain 할 때 썼던 모델 이름>), state_dict=<torch.load된 state_dict>)
    """

    """
    # bert = BertModel.from_pretrained(args.bert_model)
        if args.init_model == 'bert-base-uncased':
            bert = BertModel.from_pretrained(args.bert_model)
        elif args.init_model == 'BlueBERT':
            config = BertConfig.from_pretrained('bluebert')
            model_state_dict = torch.load('bluebert/pytorch_model.bin')
            bert = BertModel.from_pretrained('bluebert', state_dict=model_state_dict, config=config)

        self.txt_embeddings = bert.embeddings
        self.img_embeddings = ImageBertEmbeddings(args, self.txt_embeddings)

        self.img_encoder = ImageEncoder(args)
        self.encoder = bert.encoder

        self.pooler = bert.pooler

    """


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 256x256
    parser.add_argument("--train_dataset", type=str, default='/home/ubuntu/HG/cxr-bert/dset/cxr_valid.json',
                        help="train dataset for training")
    parser.add_argument("--test_dataset", type=str, default='/home/ubuntu/HG/cxr-bert/dset/cxr_valid.json',
                        help='test dataset for evaluate train set')

    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n inter: setting n")
    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: True or False")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")

    parser.add_argument("--epochs", type=int, default=1, help='number of epochs')
    parser.add_argument("--batch_size", type=int, default=16, help="number of batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="dataloader worker size")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout_prob", type=float, default=0.1)

    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--lr_patience", type=int, default=10)  # lr_scheduler.ReduceLROnPlateau
    parser.add_argument("--lr_factor", type=float, default=0.2)
    # TODO: img-SGD, txt-AdamW
    parser.add_argument("--beta1", type=float, default=0.9, help="adams first beta value")
    parser.add_argument("--beta2", type=float, default=0.999, help="adams first beta value")
    parser.add_argument("--eps", type=float, default=1e-6, help="adams epsilon")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight_decay of adam")  # 0.01 , AdamW

    # TODO: loading BlueBERT
    parser.add_argument("--bert_model", type=str, default='bert-base-uncased',
                        choices=["bert-base-uncased", "BlueBERT", "albert-base-v2"])  # for tokenizing ...
    parser.add_argument("--init_model", type=str, default='bert-base-uncased',
                        choices=["bert-base-uncased", "BlueBERT", "albert-base-v2"])
    parser.add_argument("--vocab_size", type=int, default=30522, choices=[30522, 30000])
    parser.add_argument("--max_seq_len", type=int, default=512, help="maximum sequence len")
    parser.add_argument("--hidden_size", type=int, default=768, choices=[768, 128])
    parser.add_argument("--img_hidden_sz", type=int, default=2048)
    parser.add_argument("--img_embed_pool_type", type=str, default="max", choices=["max", "avg"])
    parser.add_argument("--num_image_embeds", type=int, default=49)  # TODO: 224x224, output fiber 9...
    # -------------------------------------------------------------------------------------------
    # TODO: ...!
    parser.add_argument("--gradient_accumulation_steps", type=int, default=24)  # loss, optimizer.step() slowly
    parser.add_argument("--warmup", type=float, default=0.1)  # optimizer = BertAdam(warmup=args.warmup)
    parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()

    main(args)