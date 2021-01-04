"""
origin main.py, clean up

"""
"""
Default setting for training CXR-BERT
    1. init with BERT-base
        - BERT-base-uncased
        - BlueBERT
    2. init with BERT-small
    3. init with BERT-tiny
    4. scratch from BERT-small
    5. init with AlBERT
"""
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


import wandb
import argparse
from datetime import datetime

from data.dataset import CXRDataset
from data.helper import get_transforms
from torch.utils.data import DataLoader

from utils.utils import *
from models.train import CXRBERT_Trainer  # CXR-BERT
# from models.train_bertformlm_bluebert import CXRBERT_Trainer

from transformers import BertTokenizer, AlbertTokenizer, AutoTokenizer

def train(args):
    wandb.init(config=args, project='CXR-BERT')

    set_seed(args.seed)

    # TODO: bert-base,small,tiny tokenizer
    if args.bert_model == "albert-base-v2":
        tokenizer = AlbertTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize
    elif args.bert_model == "emilyalsentzer/Bio_ClinicalBERT":  # same with Bert-base-cased model
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model).tokenize
    elif args.bert_model == "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12":
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model).tokenize
    elif args.bert_model == "bert-small-scratch":
        tokenizer = BertTokenizer.from_pretrained("google/bert_uncased_L-4_H-512_A-8", do_lower_case=True).tokenize
    # elif args.bert_model == "load_pretrained_model":
    #     tokenizer = BertTokenizer.from_pretrained(args.init_model, do_lower_case=True).tokenize
    else:
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize

    transforms = get_transforms()

    print("Load Train dataset", args.train_dataset)
    train_dataset = CXRDataset(args.train_dataset, tokenizer, transforms, args)

    print("Load Test dataset", args.test_dataset)
    test_dataset = CXRDataset(args.test_dataset, tokenizer, transforms, args) \
        if args.test_dataset is not None else None

    print("Create DataLoader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False) \
        if test_dataset is not None else None

    print("Building CXRBERT model")
    # TODO: Remove after check, CXRBERT or CXRBertEncoder... ?
    # cxr_bert = CXRBertEncoder(args)

    print("Creating BERT Trainer")
    trainer = CXRBERT_Trainer(args, train_dataloader=train_data_loader, test_dataloader=test_data_loader)

    print("Training Start!")
    for epoch in range(args.epochs):
        trainer.train(epoch)
        trainer.save(epoch, args.output_path)

        # if test_data_loader is not None:
        #     trainer.test(epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dataset", type=str,
                        default='/home/ubuntu/HG/cxr-bert/dset/img_512_3ch_label/cxr_train_label.json',
                        help="train dataset for training")
    parser.add_argument("--test_dataset", type=str,
                        default='/home/ubuntu/HG/cxr-bert/dset/img_512_3ch_label/cxr_valid_label.json',
                        help='test dataset for evaluating train set')

    # parser.add_argument("--train_dataset", type=str,
    #                     default='/home/mimic-cxr/dataset/image_preprocessing/Train.jsonl',
    #                     help="train dataset for training")
    # parser.add_argument("--test_dataset", type=str,
    #                     default='/home/mimic-cxr/dataset/image_preprocessing/Valid.jsonl',
    #                     help='test dataset for evaluating train set')

    output_path = 'output/' + str(datetime.now())
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        os.chmod(output_path, 0o777)

    parser.add_argument("--output_path", type=str, default=output_path, help="ex)path/to/save/model")
    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n inter: setting n")
    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: True or False")
    # parser.add_argument("--cuda_devices", type=int, default=[0], help="CUDA device ids")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")

    parser.add_argument("--mlm_task", type=str, default=True,
                        help="The model will train only mlm task!! | True | False")
    parser.add_argument("--itm_task", type=str, default=True,
                        help="The model will train only itm task!! | True | False")
    # TODO: attention !!!
    parser.add_argument('--attn_1d', type=bool, default=False, help='choose 1d attn(True) or full attn(False)')
    # parser.add_argument("--freeze", type=str, default=False, help="if we set true, it will use fixed feature")

    # TODO: for the VLP attn, (s2s_prob + bi_prob) == 1 !!!
    parser.add_argument('--s2s_prob', default=0.5, type=float,
                        help="S2S attention prob.")
    parser.add_argument('--bi_prob', default=0.5, type=float,
                        help="Full_attention prob.")

    parser.add_argument("--epochs", type=int, default=300, help='number of epochs')
    parser.add_argument("--batch_size", type=int, default=56, help="number of batch size")
    parser.add_argument("--num_workers", type=int, default=20, help="dataloader worker size")

    # TODO: init model
    parser.add_argument("--hidden_size", type=int, default=512, choices=[768, 512, 128])
    parser.add_argument("--embedding_size", type=int, default=512, choices=[768, 512, 128])

    parser.add_argument("--bert_model", type=str, default="google/bert_uncased_L-4_H-512_A-8",
                        choices=["albert-base-v2",
                                 "bert-base-uncased",
                                 "google/bert_uncased_L-4_H-512_A-8",  # BERT-Small
                                 "google/bert_uncased_L-2_H-128_A-2",  # BERT-Tiny
                                 "emilyalsentzer/Bio_ClinicalBERT",  # Clinical-BERT
                                 "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12",  # BlueBERT
                                 "bert-small-scratch",  # BERT-Small-scratch
                                 "load_pretrained_model"])  # pre-trained CXR-BERT
    # parser.add_argument("--init_model", type=str, default="google/bert_uncased_L-4_H-512_A-8")  # pre-trained CXR-BERT's init model

    parser.add_argument("--vocab_size", type=int, default=30522, choices=[30522, 30000])

    parser.add_argument("--img_postion", default=True, help='img_postion use!')
    parser.add_argument("--seq_len", type=int, default=253, help="maximum sequence len")  # 253
    parser.add_argument("--max_seq_len", type=int, default=512, help="total sequence len")


    parser.add_argument("--img_hidden_sz", type=int, default=2048)
    parser.add_argument("--num_image_embeds", type=int, default=200, choices=[0, 3, 200, 256])
    parser.add_argument("--img_encoder", type=str, default='random-pixel',
                        choices=['random-pixel', 'full-fiber', 'ViT'])
    parser.add_argument("--img_channel", type=int, default=3, choices=[1, 3])
    parser.add_argument("--img_size", type=int, default=512, choices=[256, 512])
    parser.add_argument("--img_embed_pool_type", type=str, default="max", choices=["max", "avg"])

    #-------------------------------------------------------------------------------------------
    # TODO: ...!
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)  # loss, optimizer.step() slowly
    parser.add_argument("--warmup", type=float, default=0.1)  # optimizer = BertAdam(warmup=args.warmup)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--dropout_prob", type=float, default=0.1)

    parser.add_argument("--beta1", type=float, default=0.9, help="adams first beta value")
    parser.add_argument("--beta2", type=float, default=0.999, help="adams first beta value")
    parser.add_argument("--eps", type=float, default=1e-6, help="adams epsilon")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight_decay of AdamW")  # 0.01 , AdamW

    args = parser.parse_args()

    train(args)