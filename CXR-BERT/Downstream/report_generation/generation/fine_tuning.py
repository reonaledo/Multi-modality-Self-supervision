"""BERT finetuning runner."""
                

#For pre-training 

############################################################
# down stream task 용도로 이거 사용 하면 된다.
############################################################
# export PYTHONPATH=$CODE_ROOT/pythia:$CODE_ROOT/pythia/pythia/legacy:$CODE_ROOT:$PYTHONPATH

# python vlp/run_img2txt_dist.py --do_train
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]= "6,7,4,5"
scenari = 3

import sys
import logging
import glob
import math
import json
import argparse
from tqdm import tqdm, trange
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import random
import copy
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer, WhitespaceTokenizer
from pytorch_pretrained_bert.modeling_like_cxrbert import BertForPreTrainingLossMask, BertForSeq2SeqDecoder
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from vlp.loader_utils import batch_list_to_batch_tensors
import vlp.seq2seq_loader as seq2seq_loader
from vlp.scst_utils import *
from misc.data_parallel import DataParallelImbalance
from vlp.image_embedding import Img_patch_embedding, fully_sampling, random_sampling 
import wandb
from pytorch_pretrained_bert import BertModel

#--Lets's load cxr pre-training model--
# fn_model_list = glob.glob(os.path.join(output_dir, "cxrbert_ep*.pt"))


def _get_max_epoch_model(output_dir):
    fn_model_list = glob.glob(os.path.join(output_dir, "model.*.bin"))
    fn_optim_list = glob.glob(os.path.join(output_dir, "optim.*.bin"))
    if (not fn_model_list) or (not fn_optim_list):
        return None
    both_set = set([int(Path(fn).stem.split('.')[-1]) for fn in fn_model_list]
                   ) & set([int(Path(fn).stem.split('.')[-1]) for fn in fn_optim_list])
    if both_set:
        return max(both_set)
    else:
        return None


def main():
    parser = argparse.ArgumentParser()
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # setting gpu number
    parser.add_argument("--cuda", default="1", type=str,
                        help="Specify your gpu number.")
    # General
    # config = AutoConfig.from_pretrained('prajjwal1/bert-tiny')
    parser.add_argument("--data_set", default="train", type=str,
                        help="train | valid")
    parser.add_argument('--img_hidden_sz', type=int, default=2048,
                        help="Whether to use amp for fp16")
    parser.add_argument('--hidden_size', type=int, default=512,
                        help="Whether to use amp for fp16")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-cased, bert-large-cased.")
    # parser.add_argument("--bert_model", default="emilyalsentzer/Bio_ClinicalBERT", type=str,
    #                     help="Bert pre-trained model selected in the list: bert-base-cased, bert-large-cased.")

    # # scenari 0> test_version.
    # if scenari == 1:
    parser.add_argument("--config_path", default='/home/mimic-cxr/model/12.26_img36_txt128_itm_sml_redsqr/config.json', type=str,
                        help="Bert config file path.")
    # config = AutoConfig.from_pretrained('prajjwal1/bert-tiny')
    if scenari == 1:
        parser.add_argument("--model_recover_path", default='/home/mimic-cxr/model/12.31_img36_txt128_full_attn/pytorch_model.bin', type=str,
                            help="The file of fine-tuned pretraining model.") # model load
        parser.add_argument("--output_dir",
                            default='/home/mimic-cxr/model/downstream_model/report_generation/scenario_1',
                            type=str,
                            help="The output directory where the model predictions and checkpoints will be written.")

    # # scenari 1> h&h
    elif scenari == 2:
        parser.add_argument("--model_recover_path", default='/home/mimic-cxr/model/12.31_img36_txt128_red_sqr/pytorch_model.bin', type=str,
                            help="The file of fine-tuned pretraining model.") # model load
        parser.add_argument("--output_dir",
                            default='/home/mimic-cxr/model/downstream_model/report_generation/scenario_2',
                            type=str,
                            help="The output directory where the model predictions and checkpoints will be written.")

    ##scenari 2> bi
    elif scenari == 3:
        parser.add_argument("--model_recover_path", default='/home/mimic-cxr/model/12.31_img36_txt128_s2s/pytorch_model.bin', type=str,
                            help="The file of fine-tuned pretraining model.") # model load
        parser.add_argument("--output_dir",
                            default='/home/mimic-cxr/model/downstream_model/report_generation/scenario_3',
                            type=str,
                            help="The output directory where the model predictions and checkpoints will be written.")

    ##scenari 3> s2s
    elif scenari == 4:
        parser.add_argument("--model_recover_path", default='/home/mimic-cxr/model/12.31_img36_txt128_random_50prob/pytorch_model.bin', type=str,
                            help="The file of fine-tuned pretraining model.") # model load
        parser.add_argument("--output_dir",
                            default='/home/mimic-cxr/model/downstream_model/report_generation/scenario_4',
                            type=str,
                            help="The output directory where the model predictions and checkpoints will be written.")
    
    # scenari 4> from-scratch
    # parser.add_argument("--config_path", default='/home/ubuntu/Multi-modality-Self-supervision/CXR-BERT_HG/output/s2s0.5_bi0.5_ran_e50_b32/config.json', type=str,
    #                     help="Bert config file path.")
    # parser.add_argument("--model_recover_path", default=None, type=str,
    #                     help="The file of fine-tuned pretraining model.") # model load
    # parser.add_argument("--output_dir",
    #                     default='/home/ubuntu/VLP/new_dir/scratch',
    #                     type=str,
    #                     help="The output directory where the model predictions and checkpoints will be written.")
    # parser.add_argument('--from_scratch', action='store_true', default = True,
    #                     help="Initialize parameters with random values (i.e., training from scratch).")

    parser.add_argument("--log_file",
                        default="training.log",
                        type=str,
                        help="The output directory where the log will be written.")
    
    ##########################################################################################################################################################################
    parser.add_argument('--img_postion', default = True,
                        help="It will give img_position.")
    parser.add_argument("--img_encoding", type=str, default='random_sample', choices=['random_sample', 'Img_patch_embedding', 'fully_use_cnn'])
    
    parser.add_argument('--from_scratch', action='store_true', default = False,
                        help="Initialize parameters with random values (i.e., training from scratch).")

    parser.add_argument("--do_train",
                        action='store_true', default = True,
                        help="Whether to run training. This should ALWAYS be set to True.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")


    ############################################################################################################
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--cnn", default="random_sample", type=str,
                    help=" image_patch | random_sample | fully_sample")
    parser.add_argument('--len_vis_input', type=int, default=49,
                        help="The length of visual token input") #visual token의 fixed length를 100이라 하면, <Unknown> token 100개가 되고, 100개의 word 생성 가능.
    parser.add_argument("--train_batch_size",
                        default=80,
                        type=int,
                        help="Total batch size for training.")
    ############################################################################################################

    # parser.add_argument("--learning_rate", default=3e-5, type=float,
    #                     help="The initial learning rate for Adam.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--label_smoothing", default=0, type=float,
                        help="The initial learning rate for Adam.")
    # parser.add_argument("--weight_decay",
    #                     default=0.01,
    #                     type=float,
    #                     help="The weight decay rate for Adam.")
    parser.add_argument("--weight_decay",
                        default=0.0,
                        type=float,
                        help="The weight decay rate for Adam.")
    parser.add_argument("--finetune_decay",
                        action='store_true',
                        help="Weight decay to the original weights.")
    parser.add_argument("--num_train_epochs",
                        default=30,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    # parser.add_argument("--warmup_proportion",
    #                     default=0,
    #                     type=float,
    #                     help="Proportion of training to perform linear learning rate warmup for. "
    #                          "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--global_rank",
                        type=int,
                        default=-1,
                        help="global_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=123,
                        help="random seed for initialization")
    
    
    parser.add_argument('--fp16', action='store_true', default = False,
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp32_embedding', action='store_true',default = False,
                        help="Whether to use 32-bit float precision instead of 32-bit for embeddings")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--amp', action='store_true', default = False,
                        help="Whether to use amp for fp16")
                        

    parser.add_argument('--new_segment_ids', default = False, action='store_true',
                        help="Use new segment ids for bi-uni-directional LM.")
    parser.add_argument('--tokenized_input', action='store_true',
                        help="Whether the input is tokenized.")

    parser.add_argument('--max_len_b', type=int, default=128,
                        help="Truncate_config: maximum length of segment B.")
    parser.add_argument('--trunc_seg', default='b',
                        help="Truncate_config: first truncate segment A/B (option: a, b).")
    parser.add_argument('--always_truncate_tail', action='store_true',
                        help="Truncate_config: Whether we should always truncate tail.")
    # parser.add_argument("--mask_prob", default=0.15, type=float,
    #                     help="Number of prediction is sometimes less than max_pred when sequence is short.") 
    
    parser.add_argument("--mask_prob", default=0.7, type=float,
                        help="Number of prediction is sometimes less than max_pred when sequence is short.")
    parser.add_argument('--max_pred', type=int, default=128,
                        help="Max tokens of prediction.")
    parser.add_argument("--num_workers", default=30, type=int,
                        help="Number of workers for the data loader.")
    parser.add_argument('--max_position_embeddings', type=int, default=None,
                        help="max position embeddings")

    # Others for VLP
    # parser.add_argument("--src_file", default=['/home/ubuntu/simclr_/Multi-modality-Self-supervision/*.jsonl'],
    #                     type=str, nargs='+',
    #                     help="The input data file name.")

    parser.add_argument("--src_file", default='/home/mimic-cxr/dataset/image_preprocessing/Train.jsonl',
                        type=str, help="The input data file name.")
    parser.add_argument('--enable_visdom', action='store_true')
    parser.add_argument('--visdom_port', type=int, default=8888)
    parser.add_argument('--image_root', type=str, default='/home/mimic-cxr/dataset/image_preprocessing/re_512_3ch/Train')
    parser.add_argument('--dataset', default='cxr', type=str,
                        help='coco | flickr30k | cc | cxr')
    parser.add_argument('--split', type=str, nargs='+', default=['train', 'valid'])

    parser.add_argument('--world_size', default = 1, type = int,
                        help = 'number of distributed processes')
    parser.add_argument('--dist_url', default='file://[PT_OUTPUT_DIR]/nonexistent_file', type = str,
                        help = 'url used to set up distributed training')
    parser.add_argument('--file_valid_jpgs', default='/home/mimic-cxr/dataset/image_preprocessing/Valid.jsonl', type=str)
    parser.add_argument('--sche_mode', default='warmup_linear', type=str,
                        help="warmup_linear | warmup_constant | warmup_cosine")
    parser.add_argument('--drop_prob', default=0.1, type=float)
    parser.add_argument('--use_num_imgs', default=-1, type=int)
    parser.add_argument('--vis_mask_prob', default=0, type=float)
    parser.add_argument('--max_drop_worst_ratio', default=0, type=float)
    parser.add_argument('--drop_after', default=6, type=int)

    parser.add_argument('--s2s_prob', default=1, type=float,
                        help="Percentage of examples that are bi-uni-directional LM (seq2seq). This must be turned off!!!!!!! because this is not for seq2seq model!!!")
    parser.add_argument('--bi_prob', default=0, type=float,
                        help="Percentage of examples that are bidirectional LM.")
                        
    parser.add_argument('--enable_butd', default = False,
                        help='set to take in region features')

    # parser.add_argument('--region_bbox_file', default='coco_detection_vg_thresh0.2_feat_gvd_checkpoint_trainvaltest.h5', type=str)
    # parser.add_argument('--region_det_file_prefix', default='feat_cls_1000/coco_detection_vg_100dets_gvd_checkpoint_trainval', type=str)
    parser.add_argument('--region_bbox_file', default='', type=str)
    parser.add_argument('--region_det_file_prefix', default='', type=str)
    
    parser.add_argument('--tasks', default='img2txt',
                        help='img2txt | vqa2 | txtonly')
    parser.add_argument('--relax_projection',
                        action='store_true',
                        help="Use different projection layers for tasks.")
    parser.add_argument('--scst', default = False,#action='store_true',
                        help='Self-critical sequence training')

    args = parser.parse_args()

    print('global_rank: {}, local rank: {}'.format(args.global_rank, args.local_rank))

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda  # setting gpu number

    wandb.init(config=args, project="report_generation", entity="mimic-cxr")
    wandb.config["more"] = "custom"

    args.max_seq_length = args.max_len_b + args.len_vis_input + 3 # +3 for 2x[SEP] and [CLS]
    args.mask_image_regions = (args.vis_mask_prob > 0) # whether to mask out image regions
    args.dist_url = args.dist_url.replace('[PT_OUTPUT_DIR]', args.output_dir)
    print(" # PID :", os.getpid())
    # arguments inspection
    assert(args.tasks in ('img2txt', 'vqa2', 'txtonly'))
    # assert args.enable_butd == True, 'only support region attn! featmap attn deprecated'
    # assert (not args.scst) or args.dataset == 'coco', 'scst support on coco only!'
    # if args.scst:
    #     assert args.dataset == 'coco', 'scst support on coco only!'
    #     assert args.max_pred == 0 and args.mask_prob == 0, 'no mask for scst!'
    #     rl_crit = RewardCriterion()

    # output config
    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(args.__dict__, open(os.path.join(
        args.output_dir, 'opt.json'), 'w'), sort_keys=True, indent=2)

    logging.basicConfig(
        filename=os.path.join(args.output_dir, args.log_file),
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)
    logger = logging.getLogger(__name__)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        print("device",device)
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        print("device",device)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl', init_method = args.dist_url,
            world_size=args.world_size, rank=args.global_rank)
            
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(
        args.train_batch_size / args.gradient_accumulation_steps)

    # fix random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # plotting loss, optional
    if args.enable_visdom:
        import visdom
        vis = visdom.Visdom(port=args.visdom_port, env=args.output_dir)
        vis_window={'iter': None, 'score':None}

    # tokenizer = BertTokenizer.from_pretrained(
    #     args.bert_model, do_lower_case=True,
    #     cache_dir=args.output_dir+'/.pretrained_model_{}'.format(args.global_rank))
    

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=True)

    # tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    if args.max_position_embeddings:
        tokenizer.max_len = args.max_position_embeddings

    data_tokenizer = WhitespaceTokenizer() if args.tokenized_input else tokenizer

    if args.do_train:
        print("args.mask_prob",args.mask_prob)
        print("args.train_batch_size",args.train_batch_size)
        bi_uni_pipeline = [seq2seq_loader.Preprocess4Seq2seq(args.max_pred, args.mask_prob,
            list(tokenizer.vocab.keys()), tokenizer.convert_tokens_to_ids, args.max_seq_length,
            new_segment_ids=args.new_segment_ids, truncate_config={
            'max_len_b': args.max_len_b, 'trunc_seg': args.trunc_seg, 'always_truncate_tail':
            args.always_truncate_tail}, mask_image_regions=args.mask_image_regions,
            mode="s2s", len_vis_input=args.len_vis_input,
            vis_mask_prob=args.vis_mask_prob, enable_butd=args.enable_butd,
            region_bbox_file=args.region_bbox_file, region_det_file_prefix=args.region_det_file_prefix,
            local_rank=args.local_rank, load_vqa_ann=(args.tasks=='vqa2'))]

        bi_uni_pipeline.append(seq2seq_loader.Preprocess4Seq2seq(args.max_pred, args.mask_prob,
            list(tokenizer.vocab.keys()), tokenizer.convert_tokens_to_ids, args.max_seq_length,
            new_segment_ids=args.new_segment_ids, truncate_config={
            'max_len_b': args.max_len_b, 'trunc_seg': args.trunc_seg, 'always_truncate_tail':
            args.always_truncate_tail}, mask_image_regions=args.mask_image_regions,
            mode="bi", len_vis_input=args.len_vis_input,
            vis_mask_prob=args.vis_mask_prob, enable_butd=args.enable_butd,
            region_bbox_file=args.region_bbox_file, region_det_file_prefix=args.region_det_file_prefix,
            local_rank=args.local_rank, load_vqa_ann=(args.tasks=='vqa2')))

        train_dataset = seq2seq_loader.Img2txtDataset( args.data_set,
            args.src_file, args.image_root,args.split, args.train_batch_size,
            data_tokenizer, args.max_seq_length, file_valid_jpgs=args.file_valid_jpgs,
            bi_uni_pipeline=bi_uni_pipeline, use_num_imgs=args.use_num_imgs,
            s2s_prob=args.s2s_prob, # this must be set to 1.
             bi_prob=args.bi_prob, tasks=args.tasks)
        
        if args.world_size == 1:
            train_sampler = RandomSampler(train_dataset, replacement=False)
        else:
            train_sampler = DistributedSampler(train_dataset)

        train_dataloader = torch.utils.data.DataLoader(train_dataset,
            batch_size=args.train_batch_size, sampler=train_sampler, num_workers=args.num_workers,
            collate_fn=batch_list_to_batch_tensors, pin_memory=True)

    # note: args.train_batch_size has been changed to (/= args.gradient_accumulation_steps)
    
    t_total = int(len(train_dataloader) * args.num_train_epochs * 1. /
                  args.gradient_accumulation_steps)

    amp_handle = None
    if args.fp16 and args.amp:
        from apex import amp
        amp_handle = amp.init(enable_caching=True)
        logger.info("enable fp16 with amp")

    # Prepare model
    recover_step = _get_max_epoch_model(args.output_dir)
    cls_num_labels = 2
    type_vocab_size = 6 if args.new_segment_ids else 2
    relax_projection = 4 if args.relax_projection else 0
    task_idx_proj = 3 if args.tasks == 'img2txt' or 'txtonly' else 0

    mask_word_id, eos_word_ids, pad_word_ids = tokenizer.convert_tokens_to_ids(["[MASK]", "[SEP]", "[PAD]"]) # index in BERT vocab: 103, 102, 0

    _state_dict = {}
    model = BertForPreTrainingLossMask.from_pretrained( 
        args.bert_model, state_dict=_state_dict, args=args, num_labels=cls_num_labels,
        type_vocab_size=type_vocab_size, relax_projection=relax_projection,
        config_path=args.config_path, task_idx=task_idx_proj,
        max_position_embeddings=args.max_position_embeddings, label_smoothing=args.label_smoothing,
        fp32_embedding=args.fp32_embedding, cache_dir=args.output_dir+'/.pretrained_model_{}'.format(args.global_rank),
        drop_prob=args.drop_prob, enable_butd=args.enable_butd,
        len_vis_input=args.len_vis_input, tasks=args.tasks)

    print("original vlp model's statedict : ")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    # input("STOP!")
    
    # BERT model will be loaded! from scratch
    if (recover_step is None) and (args.model_recover_path is None):
        _state_dict = {} if args.from_scratch else None

        _state_dict = {}
        model = BertForPreTrainingLossMask.from_pretrained(
            args.bert_model, state_dict=_state_dict, args=args, num_labels=cls_num_labels,
            type_vocab_size=type_vocab_size, relax_projection=relax_projection,
            config_path=args.config_path, task_idx=task_idx_proj,
            max_position_embeddings=args.max_position_embeddings, label_smoothing=args.label_smoothing,
            fp32_embedding=args.fp32_embedding, cache_dir=args.output_dir+'/.pretrained_model_{}'.format(args.global_rank),
            drop_prob=args.drop_prob, enable_butd=args.enable_butd,
            len_vis_input=args.len_vis_input, tasks=args.tasks)

        print("scratch model's statedict : ")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        # input("STOP!")
        global_step = 0
        print("The loaded model is BertForPreTrainingLossMask train from scratch")
        
    else:
        print("OUR Training goal :",args.tasks, args.s2s_prob)
        # if recover_step:
        #     logger.info("***** Recover model: %d *****", recover_step)
        #     model_recover = torch.load(os.path.join(
        #         args.output_dir, "model.{0}.bin".format(recover_step)))
        #     # recover_step == number of epochs
        #     global_step = math.floor(
        #         recover_step * t_total * 1. / args.num_train_epochs)

        # elif args.model_recover_path:
        print("Recoverd model :", args.model_recover_path)

        for model_recover_path in glob.glob(args.model_recover_path.strip()):
            logger.info("***** Recover model: %s *****",
                        args.model_recover_path)

            # if args.model_recover_path == 'bert_base':
            #     model_recover = BertModel.from_pretrained('bert-base-uncased')
            # else:             
            model_recover = torch.load(model_recover_path)

            for key in list(model_recover.keys()):
                model_recover[key.replace('enc.', '')] = model_recover.pop(key)

            print("Loaded Model's state_dict:")
            for param_tensor in model_recover:
                print(param_tensor, "\t", model_recover[param_tensor].size())

            global_step = 0

        model = BertForPreTrainingLossMask.from_pretrained(
            args.bert_model, 
            state_dict=model_recover,
            args=args,num_labels=cls_num_labels,
            type_vocab_size=type_vocab_size, 
            relax_projection=relax_projection,
            config_path=args.config_path, 
            task_idx=task_idx_proj,
            max_position_embeddings=args.max_position_embeddings, 
            label_smoothing=args.label_smoothing,
            fp32_embedding=args.fp32_embedding, 
            cache_dir=args.output_dir+'/.pretrained_model_{}'.format(args.global_rank),
             drop_prob=args.drop_prob, 
            enable_butd=args.enable_butd,
            len_vis_input=args.len_vis_input, tasks=args.tasks)

        model.load_state_dict(model_recover, strict=False)

        
        print("The pretrained model loaded and fine-tuning.")
        del model_recover
        torch.cuda.empty_cache()

    if args.fp16:
        model.half()
        cnn.half()
        if args.fp32_embedding:
            model.bert.embeddings.word_embeddings.float()
            model.bert.embeddings.position_embeddings.float()
            model.bert.embeddings.token_type_embeddings.float()
    model.to(device)
    if args.local_rank != -1:
        try:
            # from apex.parallel import DistributedDataParallel as DDP
            from torch.nn.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model, device_ids = [args.local_rank], output_device = args.local_rank, find_unused_parameters=True)

    elif n_gpu > 1:
        model = DataParallelImbalance(model)

    # Prepare optimizer
    wandb.watch(model)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    if args.fp16:
        try:
            # from apex.optimizers import FP16_Optimizer
            from pytorch_pretrained_bert.optimization_fp16 import FP16_Optimizer_State
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer_State(
                optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer_State(
                optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             schedule=args.sche_mode,
                             t_total=t_total)

    if recover_step:
        logger.info("***** Recover optimizer: %d *****", recover_step)
        optim_recover = torch.load(os.path.join(
            args.output_dir, "optim.{0}.bin".format(recover_step)))
        if hasattr(optim_recover, 'state_dict'):
            optim_recover = optim_recover.state_dict()
        optimizer.load_state_dict(optim_recover)
        if args.loss_scale == 0:
            logger.info("***** Recover optimizer: dynamic_loss_scale *****")
            optimizer.dynamic_loss_scale = True

    logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()

    if args.do_train:
        logger.info("***** Running training *****")
        model.train()
        print("Total Parameters:", sum([p.nelement() for p in model.parameters()]))

        if recover_step:
            start_epoch = recover_step+1
        else:
            start_epoch = 1

        for i_epoch in trange(start_epoch, args.num_train_epochs+1, desc="Epoch"):
            if args.local_rank >= 0:
                train_sampler.set_epoch(i_epoch-1)
            iter_bar = tqdm(train_dataloader, desc='Iter (loss=X.XXX, avg_mlm_acc=X.XXX, mlm_acc=X.XXX)')
            nbatches = len(train_dataloader) # batch: 12
            train_loss = []
            pretext_loss = []
            vqa2_loss = []
            scst_reward = []

            avg_loss = 0.0
            total_itm_prob = 0
            total_mlm_prob = 0
            batch_count = 0
            for step, batch in enumerate(iter_bar):
                batch = [t.to(device) for t in batch]
                input_ids, segment_ids, input_mask, lm_label_ids, masked_pos, masked_weights, is_next, task_idx, img, vis_masked_pos, vis_pe, ans_labels = batch
                if args.fp16:
                    img = img.half()
                    vis_pe = vis_pe.half()

                loss_tuple = model(img, vis_pe, input_ids, segment_ids,
                    input_mask, lm_label_ids, ans_labels, is_next, masked_pos=masked_pos,
                    masked_weights=masked_weights, task_idx=task_idx,
                    vis_masked_pos=vis_masked_pos, mask_image_regions=args.mask_image_regions,
                    drop_worst_ratio=args.max_drop_worst_ratio if i_epoch > args.drop_after else 0)

                mean_reward = loss_tuple[2].new(1).fill_(0)
                masked_lm_loss, pretext_loss_deprecated, ans_loss, mlm_acc = loss_tuple

                if n_gpu > 1:    # mean() to average on multi-gpu. For dist, this is done through gradient addition.
                    mlm_acc = mlm_acc.mean()
                    masked_lm_loss = masked_lm_loss.mean()
                    pretext_loss_deprecated = pretext_loss_deprecated.mean()
                    ans_loss = ans_loss.mean()

                batch_count += 1
                total_mlm_prob += mlm_acc
                loss = masked_lm_loss 
           
                iter_bar.set_description('Iter (loss=%5.3f, avg_mlm_acc=%.2f, mlm_acc=%.2f)' %(loss.item(), total_mlm_prob/batch_count, mlm_acc.item()))

                train_loss.append(loss.item())
                pretext_loss.append(pretext_loss_deprecated.item())
                vqa2_loss.append(ans_loss.item())
                scst_reward.append(mean_reward.item())

                if step%100 == 0:
                    logger.info("Epoch {}, Iter {}, Loss {:.2f}, Pretext {:.2f}, VQA2 {:.2f}, Mean R {:.3f}\n".format(i_epoch, step, np.mean(train_loss), np.mean(pretext_loss), np.mean(vqa2_loss), np.mean(scst_reward)))

                if args.enable_visdom:
                    if vis_window['iter'] is None:
                        vis_window['iter'] = vis.line(
                            X=np.tile(np.arange((i_epoch-1)*nbatches+step,
                                      (i_epoch-1)*nbatches+step+1), (1,1)).T,
                            Y=np.column_stack((np.asarray([np.mean(train_loss)]),)),
                            opts=dict(title='Training Loss',
                                      xlabel='Training Iteration',
                                      ylabel='Loss',
                                      legend=['total'])
                        )
                    else:
                        vis.line(
                            X=np.tile(np.arange((i_epoch-1)*nbatches+step,
                                      (i_epoch-1)*nbatches+step+1), (1,1)).T,
                            Y=np.column_stack((np.asarray([np.mean(train_loss)]),)),
                            opts=dict(title='Training Loss',
                                      xlabel='Training Iteration',
                                      ylabel='Loss',
                                      legend=['total']),
                            win=vis_window['iter'],
                            update='append'
                        )

                # ensure that accumlated gradients are normalized
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                    if amp_handle:
                        amp_handle._clear_cache()
                else:
                    loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    lr_this_step = args.learning_rate * \
                        warmup_linear(global_step/t_total,
                                      args.warmup_proportion)
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1
            
            
            wandb.log({"s2s_mlm_loss": np.mean(train_loss),
                        "s2s_mlm_acc": total_mlm_prob/batch_count,
                        # "s2s_mlm_itr_acc": mlm_acc.item()
                        })

            # Save a trained model
            logger.info(
                "** ** * Saving fine-tuned model and optimizer ** ** * ")
            model_to_save = model.module if hasattr(
                model, 'module') else model  # Only save the model it-self
            output_config_file = os.path.join(args.output_dir, 'config.json')
            
            with open(output_config_file, 'w') as f:
                f.write(model_to_save.config.to_json_string())
            
            output_model_file = os.path.join(
                args.output_dir, "model.{0}.bin".format(i_epoch))
            output_optim_file = os.path.join(
                args.output_dir, "optim.{0}.bin".format(i_epoch))
            if args.global_rank in (-1, 0): # save model if the first device or no dist
                torch.save(copy.deepcopy(model_to_save).cpu().state_dict(), output_model_file)
                # torch.save(optimizer.state_dict(), output_optim_file) # disable for now, need to sanitize state and ship everthing back to cpu

            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

            if args.world_size > 1:
                torch.distributed.barrier()


if __name__ == "__main__":
    main()
