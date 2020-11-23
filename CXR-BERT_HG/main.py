"""
for running BertForMaskedLM with bert-base-uncased, bio-clinical-bert, bert-tiny

"""

import wandb
import argparse
from datetime import date, time, datetime
from torch.utils.data import DataLoader

from data.helper import get_transforms
from models.cxrbert import CXRBERT, CXRBertEncoder

# from data.dataset import CXRDataset
# from models.train_cxrbert import CXRBERT_Trainer  # CXR_BERT

# from data.dataset_bertformlm import CXRDataset
from data.dataset import CXRDataset
from models.train import CXRBERT_Trainer  # BertForMaskedLM


from transformers import BertTokenizer, AlbertTokenizer, AutoTokenizer
from utils.utils import *

def train(args):
    wandb.init(config=args, project='CXR-BERT')

    set_seed(args.seed)
    #torch.save(args, os.path.join(args.output_path, "args.pt"))

    if args.bert_model == "albert-base-v2":
        tokenizer = AlbertTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize
    elif args.bert_model == "bert-base-uncased":
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize
    elif args.bert_model == "Bio_clinical_bert":
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").tokenize
    elif args.bert_model == "bert_tiny":
        tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2").tokenize

    transforms = get_transforms()

    print("Load Train dataset", args.train_dataset)
    train_dataset = CXRDataset(args.train_dataset, tokenizer, transforms, args)

    print("Load Test dataset", args.test_dataset)
    test_dataset = CXRDataset(args.test_dataset, tokenizer, transforms, args) \
        if args.test_dataset is not None else None

    print("Create DataLoader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True) \
        if test_dataset is not None else None

    print("Building CXRBERT model")
    # TODO: CXRBERT or CXRBertEncoder... ?
    cxr_bert = CXRBertEncoder(args)

    print("Creating BERT Trainer")
    trainer = CXRBERT_Trainer(args, cxr_bert, train_dataloader=train_data_loader, test_dataloader=test_data_loader)

    print("Training Start!")
    for epoch in range(args.epochs):
        trainer.train(epoch)
        trainer.save(epoch, args.output_path)

        if test_data_loader is not None:
            trainer.test(epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dataset", type=str, default='/home/ubuntu/HG/cxr-bert/dset/img_512/cxr_train.json',
                        help="train dataset for training")
    parser.add_argument("--test_dataset", type=str, default='/home/ubuntu/HG/cxr-bert/dset/img_512/cxr_test.json',
                        help='test dataset for evaluate train set')

    output_path = 'output/' + str(datetime.now())
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    parser.add_argument("--output_path", type=str, default=output_path, help="ex)path/to/save/model")

    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n inter: setting n")
    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: True or False")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")

    parser.add_argument("--epochs", type=int, default=1000, help='number of epochs')
    parser.add_argument("--batch_size", type=int, default=16, help="number of batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="dataloader worker size")

    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--dropout_prob", type=float, default=0.1)

    # parser.add_argument("--warmup_steps", type=int, default=3000)
    parser.add_argument("--lr_patience", type=int, default=10)  # lr_scheduler.ReduceLROnPlateau
    parser.add_argument("--lr_factor", type=float, default=0.2)
    # TODO: img-SGD, txt-AdamW
    parser.add_argument("--beta1", type=float, default=0.9, help="adams first beta value")
    parser.add_argument("--beta2", type=float, default=0.999, help="adams first beta value")
    parser.add_argument("--eps", type=float, default=1e-6, help="adams epsilon")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight_decay of adam")  # 0.01 , AdamW

    # TODO: loading BlueBERT
    parser.add_argument("--hidden_size", type=int, default=512, choices=[768, 512, 128])
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased",
                        choices=["bert-base-uncased", "BlueBERT",
                                 "albert-base-v2", "Bio_clinical_bert",
                                 "bert_tiny"])  # for tokenizing ...
    parser.add_argument("--init_model", type=str, default="google/bert_uncased_L-4_H-512_A-8",
                        choices=["bert-base-uncased", "BlueBERT", "albert-base-v2",
                                 "google/bert_uncased_L-4_H-512_A-8", "google/bert_uncased_L-2_H-128_A-2"])

    parser.add_argument("--max_seq_len", type=int, default=512, help="maximum sequence len")
    parser.add_argument("--vocab_size", type=int, default=30522, choices=[30522, 30000])
    parser.add_argument("--embedding_size", type=int, default=512, choices=[768, 512, 128])


    parser.add_argument("--img_hidden_sz", type=int, default=2048)
    parser.add_argument("--img_embed_pool_type", type=str, default="max", choices=["max", "avg"])
    parser.add_argument("--num_image_embeds", type=int, default=100)  # TODO: 224x224, output fiber 49...
    #-------------------------------------------------------------------------------------------
    # TODO: ...!
    parser.add_argument("--gradient_accumulation_steps", type=int, default=24)  # loss, optimizer.step() slowly
    parser.add_argument("--warmup", type=float, default=0.1)  # optimizer = BertAdam(warmup=args.warmup)
    parser.add_argument("--seed", type=int, default=123)

    #parser.add_argument("--patience", type=int, default=10)  # n_no_improve > args.patienc: break
    # parser.add_argument("--weight_classes", type=int, default=1)  # for multi-label classification weight

    args = parser.parse_args()

    train(args)
