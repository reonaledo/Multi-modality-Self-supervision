import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]= "5,6,7"
scenari = 7

import wandb
import argparse
from datetime import date, time, datetime
from torch.utils.data import DataLoader
from data.helper import get_transforms
from data.dataset import CXRDataset, Preprocess4Seq2seq
from data.dataset_origin import CXRDataset_origin#, Preprocess4Seq2seq
from models.cxrbert import CXRBERT, CXRBertEncoder
from models.new_train_cxrbert import CXRBERT_Trainer
from utils.loader_utils import batch_list_to_batch_tensors

# from models.train_origin import CXRBERT_Trainer_origin  # CXR-BERT
from models.train_vlp import CXRBERT_Trainer_origin  # CXR-BERT

from transformers import BertTokenizer, AlbertTokenizer, AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoConfig
from utils.utils import *


def train(args):
    # wandb.init(config=args, project="report_generation", entity="mimic-cxr")
    

    print(" # PID :", os.getpid())
    set_seed(args.seed)
    #torch.save(args, os.path.join(args.output_path, "args.pt"))

    if args.bert_model == "bert-base-uncased":
        tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True).tokenize
    elif args.bert_model == 'bert_small':
        tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-4_H-512_A-8").tokenize
    elif args.bert_model == "ClinicalBERT":
        tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").tokenize
    elif args.bert_model == "bert_tiny":
        tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-2_H-128_A-2").tokenize
    
    transforms = get_transforms(args)

    # tokenizer, transforms, mode=None, seq_len, num_image_embeds, new_segment_ids, bert_model

    ################################################################################
    if args.dataset_option == 'new_dataet':
        bi_uni_pipeline = [Preprocess4Seq2seq(tokenizer, transforms, mode="s2s", seq_len=args.seq_len, num_image_embeds=args.num_image_embeds, new_segment_ids=args.new_segment_ids, bert_model=args.bert_model, mask_prob=args.mask_prob,attn_1d=args.attn_1d, masked_attnt_dropout=args.masked_attnt_dropout)]
        bi_uni_pipeline.append(Preprocess4Seq2seq(tokenizer, transforms, mode="bi", seq_len=args.seq_len, num_image_embeds=args.num_image_embeds, new_segment_ids=args.new_segment_ids, bert_model=args.bert_model, mask_prob=args.mask_prob, attn_1d=args.attn_1d, masked_attnt_dropout=args.masked_attnt_dropout))

        train_dataset = CXRDataset(args.train_dataset, tokenizer, bi_uni_pipeline=bi_uni_pipeline, batch_size=args.batch_size, s2s_prob=args.s2s_prob, bi_prob=args.bi_prob)

        print("Load Test dataset", args.test_dataset)
        test_dataset = CXRDataset(args.test_dataset, tokenizer, bi_uni_pipeline=bi_uni_pipeline, batch_size=args.batch_size, s2s_prob=args.s2s_prob, bi_prob=args.bi_prob) \
            if args.test_dataset is not None else None

        print("Create DataLoader")
        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=batch_list_to_batch_tensors,shuffle=True)
        test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=batch_list_to_batch_tensors,shuffle=False) \
            if test_dataset is not None else None
    
    ################################################################################
    if args.dataset_option == 'origin_dataset':
        print("Load Train dataset", args.train_dataset)
        train_dataset = CXRDataset_origin(args.train_dataset, tokenizer, transforms, args)
        print("Load Test dataset", args.test_dataset)
        test_dataset = CXRDataset_origin(args.test_dataset, tokenizer, transforms, args) \
            if args.test_dataset is not None else None
        print("Create DataLoader")
        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False) \
            if test_dataset is not None else None
    ################################################################################
    
    if args.trainer_option == 'new_train':
        print("Building CXRBERT model")
        cxr_bert = CXRBertEncoder(args)
        print("Creating BERT Trainer")
        trainer = CXRBERT_Trainer(args, cxr_bert, train_dataloader=train_data_loader, test_dataloader=test_data_loader)
    else:
        trainer = CXRBERT_Trainer_origin(args, train_dataloader=train_data_loader, test_dataloader=test_data_loader)

    print("Training Start!")
    for epoch in range(args.epochs):
        trainer.train(epoch)
        trainer.save(epoch, args.output_path)

        # if test_data_loader is not None:
        #     trainer.test(epoch)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset", type=str, default='/home/mimic-cxr/dataset/image_preprocessing/Train.jsonl',
                        help="train dataset for training")
    parser.add_argument("--test_dataset", type=str, default='/home/mimic-cxr/dataset/image_preprocessing/Valid.jsonl',
                        help='test dataset for evaluate train set')
    
    
    #scenari 1> 224x224, img 36, txt 128, 2d full attention., you should 추가 helper에서 resize option.
    if scenari == 1:
        output_path = '/home/mimic-cxr/model/downstream_model/report_generation/img36_txt128_bi_only'
        parser.add_argument('--s2s_prob', default=0, type=float,
                            help="Percentage of examples that are bi-uni-directional LM (seq2seq). This must be turned off!!!!!!! because this is not for seq2seq model!!!")
        parser.add_argument('--bi_prob', default=1, type=float,
                            help="Percentage of examples that are bidirectional LM.")
        parser.add_argument("--masked_attnt_dropout", type=int, default=False, choices=[True, False])
        parser.add_argument("--num_image_embeds", type=int, default=36, choices=[25, 36, 49])  # 50%, 30%, 0% drop
        parser.add_argument("--seq_len", type=int, default=128, help="maximum sequence len")
        parser.add_argument("--from_scratch", type=str, default=False, help="The model will be trained from scratch!!: True or False")
        parser.add_argument("--mixed_attnt", type=str, default=False, help="The model will be trained from scratch!!: True or False")

    # #scenari 2> 224x224, img 36, txt 128, redqr attention., you should 추가 helper에서 resize option.
    elif scenari == 2:
        output_path = '/home/mimic-cxr/model/downstream_model/report_generation/img36_txt128_PAR'
        parser.add_argument('--s2s_prob', default=0, type=float,
                        help="Percentage of examples that are bi-uni-directional LM (seq2seq). This must be turned off!!!!!!! because this is not for seq2seq model!!!")
        parser.add_argument('--bi_prob', default=1, type=float,
                        help="Percentage of examples that are bidirectional LM.")
        parser.add_argument("--masked_attnt_dropout", type=int, default=True, choices=[True, False])
        parser.add_argument("--num_image_embeds", type=int, default=36, choices=[25, 36, 49])  # 50%, 30%, 0% drop
        parser.add_argument("--seq_len", type=int, default=128, help="maximum sequence len")
        parser.add_argument("--from_scratch", type=str, default=False, help="The model will be trained from scratch!!: True or False")
        parser.add_argument("--mixed_attnt", type=str, default=False, help="The model will be trained from scratch!!: True or False")

    # #scenari 3> 224x224, img 36, txt 128, 2d s2s attention., you should 추가 helper에서 resize option.
    elif scenari == 3:
        output_path = '/home/mimic-cxr/model/downstream_model/report_generation/img36_txt128_s2s_only'
        parser.add_argument('--s2s_prob', default=1, type=float,
                        help="Percentage of examples that are bi-uni-directional LM (seq2seq). This must be turned off!!!!!!! because this is not for seq2seq model!!!")
        parser.add_argument('--bi_prob', default=0, type=float,
                        help="Percentage of examples that are bidirectional LM.")
        parser.add_argument("--masked_attnt_dropout", type=int, default=False, choices=[True, False])
        parser.add_argument("--num_image_embeds", type=int, default=36, choices=[25, 36, 49])  # 50%, 30%, 0% drop
        parser.add_argument("--seq_len", type=int, default=128, help="maximum sequence len")
        parser.add_argument("--from_scratch", type=str, default=False, help="The model will be trained from scratch!!: True or False")
        parser.add_argument("--mixed_attnt", type=str, default=False, help="The model will be trained from scratch!!: True or False")

    # #scenari 4> 224x224, img 36, txt 128, mixed attention., you should 추가 helper에서 resize option.
    elif scenari == 4:
        output_path = '/home/mimic-cxr/model/downstream_model/report_generation/img36_txt128_bi_s2s'
        parser.add_argument('--s2s_prob', default=0.75, type=float,
                        help="Percentage of examples that are bi-uni-directional LM (seq2seq). This must be turned off!!!!!!! because this is not for seq2seq model!!!")
        parser.add_argument('--bi_prob', default=0.25, type=float,
                        help="Percentage of examples that are bidirectional LM.")
        parser.add_argument("--masked_attnt_dropout", type=int, default=True, choices=[True, False])
        parser.add_argument("--num_image_embeds", type=int, default=36, choices=[25, 36, 49])  # 50%, 30%, 0% drop
        parser.add_argument("--seq_len", type=int, default=128, help="maximum sequence len")
        parser.add_argument("--from_scratch", type=str, default=False, help="The model will be trained from scratch!!: True or False")
        parser.add_argument("--mixed_attnt", type=str, default=True, help="The model will be trained from scratch!!: True or False")

    #scenari 5> 512x512, img 180, txt 128, 2d full attention., you should 추가 helper에서 resize option.
    if scenari == 5:
        output_path = '/home/mimic-cxr/model/downstream_model/report_generation/img180_txt253_bi_only'
        parser.add_argument('--s2s_prob', default=0, type=float,
                        help="Percentage of examples that are bi-uni-directional LM (seq2seq). This must be turned off!!!!!!! because this is not for seq2seq model!!!")
        parser.add_argument('--bi_prob', default=1, type=float,
                        help="Percentage of examples that are bidirectional LM.")
        parser.add_argument("--masked_attnt_dropout", type=int, default=False, choices=[True, False])
        parser.add_argument("--num_image_embeds", type=int, default=180, choices=[128, 180, 256])  # 50%, 30%, 0% drop
        parser.add_argument("--seq_len", type=int, default=253, help="maximum sequence len")
        parser.add_argument("--from_scratch", type=str, default=False, help="The model will be trained from scratch!!: True or False")
        parser.add_argument("--mixed_attnt", type=str, default=False, help="The model will be trained from scratch!!: True or False")

    # #scenari 6> 512x512, img 36, txt 128, redqr attention., you should 추가 helper에서 resize option.
    elif scenari == 6:
        output_path = '/home/mimic-cxr/model/downstream_model/report_generation/img180_txt253_PAR'
        parser.add_argument('--s2s_prob', default=0, type=float,
                        help="Percentage of examples that are bi-uni-directional LM (seq2seq). This must be turned off!!!!!!! because this is not for seq2seq model!!!")
        parser.add_argument('--bi_prob', default=1, type=float,
                        help="Percentage of examples that are bidirectional LM.")
        parser.add_argument("--masked_attnt_dropout", type=int, default=True, choices=[True, False])
        parser.add_argument("--num_image_embeds", type=int, default=180, choices=[128, 180, 256])  # 50%, 30%, 0% drop
        parser.add_argument("--seq_len", type=int, default=253, help="maximum sequence len")
        parser.add_argument("--from_scratch", type=str, default=False, help="The model will be trained from scratch!!: True or False")
        parser.add_argument("--mixed_attnt", type=str, default=False, help="The model will be trained from scratch!!: True or False")

    # #scenari 7> 512x512, img 36, txt 128, 2d s2s attention., you should 추가 helper에서 resize option.
    elif scenari == 7:
        output_path = '/home/mimic-cxr/model/downstream_model/report_generation/img180_txt253_s2s_only'
        parser.add_argument('--s2s_prob', default=1, type=float,
                        help="Percentage of examples that are bi-uni-directional LM (seq2seq). This must be turned off!!!!!!! because this is not for seq2seq model!!!")
        parser.add_argument('--bi_prob', default=0, type=float,
                        help="Percentage of examples that are bidirectional LM.")
        parser.add_argument("--masked_attnt_dropout", type=int, default=False, choices=[True, False])
        parser.add_argument("--num_image_embeds", type=int, default=180, choices=[128, 180, 256])  # 50%, 30%, 0% drop
        parser.add_argument("--seq_len", type=int, default=253, help="maximum sequence len")
        parser.add_argument("--from_scratch", type=str, default=False, help="The model will be trained from scratch!!: True or False")
        parser.add_argument("--mixed_attnt", type=str, default=False, help="The model will be trained from scratch!!: True or False")

        
    # #scenari 8> 512x512, img 36, txt 128, random attention., you should 추가 helper에서 resize option.
    elif scenari == 8:
        output_path = '/home/mimic-cxr/model/downstream_model/report_generation/img180_txt253_bi_s2s'
        parser.add_argument('--s2s_prob', default=0.75, type=float,
                        help="Percentage of examples that are bi-uni-directional LM (seq2seq). This must be turned off!!!!!!! because this is not for seq2seq model!!!")
        parser.add_argument('--bi_prob', default=0.25, type=float,
                        help="Percentage of examples that are bidirectional LM.")
        parser.add_argument("--masked_attnt_dropout", type=int, default=True, choices=[True, False])
        parser.add_argument("--num_image_embeds", type=int, default=180, choices=[128, 180, 256])  # 50%, 30%, 0% drop
        parser.add_argument("--seq_len", type=int, default=253, help="maximum sequence len")
        parser.add_argument("--from_scratch", type=str, default=False, help="The model will be trained from scratch!!: True or False")
        parser.add_argument("--mixed_attnt", type=str, default=True, help="The model will be trained from scratch!!: True or False")

    
    print("output_path : ", output_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        os.chmod(output_path, 0o777)
    #-------------------------------------------------------------------------------------------
    parser.add_argument("--trainer_option", type=str, default='past', choices=['new_train','past'])

    parser.add_argument("--new_segment_ids", default = False, action='store_true',
                        help="Use new segment ids for bi-uni-directional LM.")

    parser.add_argument("--dataset_option", type=str, default='origin_dataset', choices=['new_dataet', 'origin_dataset'])

    parser.add_argument("--mlm_task", type=str, default=True, help="The model will train only mlm task!! | True | False")
    parser.add_argument("--itm_task", type=str, default=False, help="The model will train only itm task!! | True | False")
    
    parser.add_argument("--freeze", type=str, default=False, help="if we set true, it will use fixed feature")
    parser.add_argument("--mask_prob", type=int, default=0.5, help="maximum sequence len")
    parser.add_argument("--attn_1d", action='store_true', default=False, help='1d attnetion use!')
    parser.add_argument("--epochs", type=int, default=50, help='number of epochs')
    parser.add_argument("--batch_size", type=int, default=12, help="number of batch size")
    
    #-------------------------------------------------------------------------------------------
    parser.add_argument("--img_postion", default=True, help='img_postion use!')
    parser.add_argument("--img_encoding", type=str, default='random_sample', choices=['random_sample', 'Img_patch_embedding', 'fully_use_cnn'])  

    parser.add_argument("--bert_model", type=str, default="bert-base-uncased",
                        choices=["bert-base-uncased", "BlueBERT",
                                 "albert-base-v2", "Bio_clinical_bert",
                                 "bert_small","bert_tiny"])  # for tokenizing ...

    parser.add_argument("--init_model", type=str, default = 'google/bert_uncased_L-4_H-512_A-8', #default="google/bert_uncased_L-4_H-512_A-8",
                        choices=["bert-base-uncased", "BlueBERT", "albert-base-v2", "emilyalsentzer/Bio_ClinicalBERT"
                                 "google/bert_uncased_L-4_H-512_A-8", "google/bert_uncased_L-2_H-128_A-2"])

    parser.add_argument("--vocab_size", type=int, default=30522, choices=[30522, 30000, 28996])
    parser.add_argument("--embedding_size", type=int, default=512, choices=[768, 512, 128])
    parser.add_argument("--hidden_size", type=int, default=512, choices=[768, 512, 128])

    #-------------------------------------------------------------------------------------------    
    # parser.add_argument("--new_segment_ids", default = True, action='store_true',
    #                     help="Use new segment ids for bi-uni-directional LM.")

    parser.add_argument("--output_path", type=str, default=output_path, help="ex)path/to/save/model")
    parser.add_argument("--log_freq", type=int, default=1000, help="printing loss every n inter: setting n")
    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: True or False")
    parser.add_argument("--cuda_devices", type=int, nargs='+', help="CUDA device ids")  

    parser.add_argument("--num_workers", type=int, default=20, help="dataloader worker size")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--dropout_prob", type=float, default=0.1)

    # parser.add_argument("--lr_patience", type=int, default=10)  # lr_scheduler.ReduceLROnPlateau
    # parser.add_argument("--lr_factor", type=float, default=0.2)
    # TODO: img-SGD, txt-AdamW
    # parser.add_argument("--beta1", type=float, default=0.9, help="adams first beta value")
    # parser.add_argument("--beta2", type=float, default=0.999, help="adams first beta value")
    # parser.add_argument("--eps", type=float, default=1e-6, help="adams epsilon")
    # parser.add_argument("--weight_decay", type=float, default=0.01, help="weight_decay of adam")  # 0.01 , AdamW

    # TODO: loading BlueBERT
    parser.add_argument("--img_hidden_sz", type=int, default=2048)
    parser.add_argument("--img_embed_pool_type", type=str, default="max", choices=["max", "avg"])
    #-------------------------------------------------------------------------------------------
    # TODO: ...!
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)  # loss, optimizer.step() slowly
    parser.add_argument("--warmup", type=float, default=0.1)  # optimizer = BertAdam(warmup=args.warmup)
    parser.add_argument("--seed", type=int, default=123)

    #parser.add_argument("--patience", type=int, default=10)  # n_no_improve > args.patienc: break
    # parser.add_argument("--weight_classes", type=int, default=1)  # for multi-label classification weight

    args = parser.parse_args()

    train(args)