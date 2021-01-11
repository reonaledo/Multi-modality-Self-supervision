"""BERT finetuning runner."""

# export PYTHONPATH=$CODE_ROOT/pythia:$CODE_ROOT/pythia/pythia/legacy:$CODE_ROOT:$PYTHONPATH
# python vlp/decode_img2txt.py 


# This py file is for generating the captioning.

############### coco decoding test ###############
# conda activate vlp_test
# cd VLP_test/VLP
# export PYTHONPATH=$CODE_ROOT/pythia:$CODE_ROOT/pythia/pythia/legacy:$CODE_ROOT:$PYTHONPATH
# python vlp/decode_img2txt.py     --model_recover_path /home/ubuntu/VLP_test/VLP/data/flickr30k_g8_lr3e-5_batch512_ft_from_s0.75_b0.25/model.21.bin     --new_segment_ids --batch_size 10 --beam_size 3 --enable_butd     --image_root /home/ubuntu/VLP_test/VLP/data/flickr30k/region_feat_gvd_wo_bgd/     --dataset flickr30k --region_bbox_file /home/ubuntu/VLP_test/VLP/data/flickr30k/region_feat_gvd_wo_bgd/flickr30k_detection_vg_thresh0.2_feat_gvd_checkpoint_trainvaltest.h5     --src_file /home/ubuntu/VLP_test/VLP/data/flickr30k/annotations/dataset_flickr30k.json     --file_valid_jpgs /home/ubuntu/VLP_test/VLP/data/flickr30k/annotations/flickr30k_valid_jpgs.json

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]= "4,5"

import logging
import glob
import json
import argparse
import math
from tqdm import tqdm, trange
import numpy as np
import torch
import random
import pickle
import torch
import torch.nn as nn
import torchvision

from transformers import BertTokenizer

# from pytorch_pretrained_bert.tokenization import BertTokenizer, WhitespaceTokenizer
# from pytorch_pretrained_bert.decode_modeling import BertForSeq2SeqDecoder

# from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.modeling_like_cxrbert import BertForSeq2SeqDecoder

import vlp.seq2seq_loader as seq2seq_loader
from vlp.lang_utils import language_eval
from misc.data_parallel import DataParallelImbalance
from vlp.image_embedding import Img_patch_embedding, fully_sampling, random_sampling
import wandb

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

SPECIAL_TOKEN = ["[UNK]", "[PAD]", "[CLS]", "[MASK]"]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print(" # PID :", os.getpid())

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

def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list


def ascii_print(text):
    text = text.encode("ascii", "ignore")
    print(text)


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--cnn", default="random_sample", type=str,
    #                 help=" image_patch | random_sample | fully_sample")
    parser.add_argument("--img_postion", type=str, default=True, choices=[True | False])
    parser.add_argument("--img_encoding", type=str, default='random_sample', choices=['random_sample', 'Img_patch_embedding', 'fully_use_cnn'])
    parser.add_argument('--img_hidden_sz', type=int, default=2048,
                        help="Whether to use amp for fp16")
    parser.add_argument('--hidden_size', type=int, default=768,
                        help="Whether to use amp for fp16")
    # # General/home/ubuntu/Multi-modality-Self-supervision/CXR-BERT_HG/output/12.07ori_bi_ran_e50_b32(newseg)/config.json

    # this was used.
    # parser.add_argument("--config_path", default='/home/ubuntu/VLP/new_dir/hnh(pre)_fds2s_repo/config.json', type=str,
    #                     help="Bert config file path.")
    # parser.add_argument("--model_recover_path", default='/home/ubuntu/VLP/new_dir/hnh(pre)_fds2s_repo/model.8.bin', type=str,
    #                     help="The file of fine-tuned pretraining model.") # model load
    #######################
    # # scenari 0> test_version.
    parser.add_argument("--config_path", default='/home/ubuntu/VLP/scenario_1/config.json', type=str,
                        help="Bert config file path.")
    parser.add_argument("--model_recover_path", default='/home/ubuntu/VLP/scenario_1/model.30.bin', type=str,
                        help="The file of fine-tuned pretraining model.") # model load
    # parser.add_argument("--output_dir",
    #                     default='/home/ubuntu/VLP/new_dir/itm_only_from_test_version',
    #                     type=str,
    #                     help="The output directory where the model predictions and checkpoints will be written.")

    # parser.add_argument("--config_path", default='/home/ubuntu/Multi-modality-Self-supervision/CXR-BERT_HG/output/12.07ori_bi_ran_e50_b32(newseg)/config.json', type=str,
    #                     help="Bert config file path.")
    # parser.add_argument("--config_path", default=None, type=str,
    #                     help="Bert config file path.")


    parser.add_argument("--bert_model", default="bert-base-uncased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-cased, bert-large-cased")
    
    # parser.add_argument("--model_recover_path", default='/home/ubuntu/VLP/finetune_cxr_orgin_newseg/model.5.bin', type=str,
    #                     help="The file of fine-tuned pretraining model.") # model load
    # parser.add_argument("--model_recover_path", default='/home/ubuntu/VLP/vlp_finetune_orgin/model.27.bin', type=str,
    #                     help="The file of fine-tuned pretraining model.") # model load 

    # parser.add_argument("--model_recover_path", default='/home/ubuntu/Multi-modality-Self-supervision/CXR-BERT_HG/output/s2s0.75_bi0.25_ran_e50_b32/pytorch_model.bin', type=str,
    #                     help="The file of fine-tuned pretraining model.") # model load

    parser.add_argument('--max_position_embeddings', type=int, default=512,
                        help="max position embeddings")
                    
    # For decoding
    parser.add_argument('--fp16', action='store_true', default= False,
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--amp', action='store_true',
                        help="Whether to use amp for fp16")
    parser.add_argument('--seed', type=int, default=123,
                        help="random seed for initialization")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--new_segment_ids', action='store_true', default = True,
                        help="Use new segment ids for bi-uni-directional LM.")
    parser.add_argument('--batch_size', type=int, default=8,
                        help="Batch size for decoding.")
    parser.add_argument('--beam_size', type=int, default=1,
                        help="Beam size for searching")
    parser.add_argument('--length_penalty', type=float, default=0,
                        help="Length penalty for beam search")

    parser.add_argument('--forbid_duplicate_ngrams', action='store_true')
    parser.add_argument('--forbid_ignore_word', type=str, default=None,
                        help="Forbid the word during forbid_duplicate_ngrams")
    parser.add_argument("--min_len", default=None, type=int)
    parser.add_argument('--ngram_size', type=int, default=3)
    parser.add_argument('--max_txt_length', type=int, default=128,
                        help="maximum length of target sequence")

    # Others for VLP
    parser.add_argument("--src_file", default='/home/ubuntu/image_preprocessing/Valid.jsonl', type=str,		
                        help="The input data file name.")		
    parser.add_argument('--dataset', default='cxr', type=str,
                        help='coco | flickr30k | cc | cxr')
    parser.add_argument('--len_vis_input', type=int, default=49)
    parser.add_argument('--image_root', type=str, default='/home/ubuntu/image_preprocessing/re_512_3ch/valid')		
    parser.add_argument('--split', type=str, default='valid')
    parser.add_argument('--drop_prob', default=0.1, type=float)
    parser.add_argument('--file_valid_jpgs', default='', type=str)

    args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda  # setting gpu number

     # fix random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)

    args.max_seq_length = args.max_txt_length + args.len_vis_input + 3 # +3 for 2x[SEP] and [CLS]
    # tokenizer.max_len = args.max_seq_length

    bi_uni_pipeline = []
    # def __init__(self, tokenizer, max_len, max_txt_length, new_segment_ids=False, mode="s2s", len_vis_input=None):
    bi_uni_pipeline.append(seq2seq_loader.Preprocess4Seq2seqDecoder(tokenizer.convert_tokens_to_ids, args.max_seq_length,
        max_txt_length=args.max_txt_length, new_segment_ids=args.new_segment_ids,
        mode='s2s', len_vis_input=args.len_vis_input))

    # print("bi_uni_pipeline",bi_uni_pipeline)
    amp_handle = None
    if args.fp16 and args.amp:
        from apex import amp
        amp_handle = amp.init(enable_caching=True)
        logger.info("enable fp16 with amp")

    # Prepare model
    cls_num_labels = 2
    type_vocab_size = 6 if args.new_segment_ids else 2

    mask_word_id, eos_word_ids = tokenizer.convert_tokens_to_ids(["[MASK]", "[SEP]"])
    
    forbid_ignore_set = None
    if args.forbid_ignore_word:
        w_list = []
        for w in args.forbid_ignore_word.split('|'):
            if w.startswith('[') and w.endswith(']'):
                w_list.append(w.upper())
            else:
                w_list.append(w)
        forbid_ignore_set = set(tokenizer.convert_tokens_to_ids(w_list))

    # _state_dict = {}
    model = BertForSeq2SeqDecoder.from_pretrained(args.bert_model,
            max_position_embeddings=args.max_position_embeddings, config_path=args.config_path,
            state_dict={}, args=args, num_labels=cls_num_labels,
            type_vocab_size=type_vocab_size, task_idx=3, mask_word_id=mask_word_id,
            search_beam_size=args.beam_size, length_penalty=args.length_penalty,
            eos_id=eos_word_ids, forbid_duplicate_ngrams=args.forbid_duplicate_ngrams,
            forbid_ignore_set=forbid_ignore_set, ngram_size=args.ngram_size, min_len=args.min_len,
            len_vis_input=args.len_vis_input)

    print("original vlp decoder's statedict : ")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    input("STOP!")

    for model_recover_path in glob.glob(args.model_recover_path.strip()):
        logger.info("***** Recover model: %s *****", model_recover_path)
        model_recover = torch.load(model_recover_path)

        for key in list(model_recover.keys()):
                model_recover[key.replace('txt_embeddings', 'bert.txt_embeddings'). replace('img_embeddings', 'bert.img_embeddings'). replace('img_encoder.model', 'bert.img_encoder.model'). replace('encoder.layer', 'bert.encoder.layer'). replace('pooler', 'bert.pooler')] = model_recover.pop(key)

        print("Loaded Model's state_dict:")
        for param_tensor in model_recover:
            print(param_tensor, "\t", model_recover[param_tensor].size())

        model = BertForSeq2SeqDecoder.from_pretrained(args.bert_model,
            max_position_embeddings=args.max_position_embeddings, config_path=args.config_path,
            state_dict=model_recover, args=args, num_labels=cls_num_labels,
            type_vocab_size=type_vocab_size, task_idx=3, mask_word_id=mask_word_id,
            search_beam_size=args.beam_size, length_penalty=args.length_penalty,
            eos_id=eos_word_ids, forbid_duplicate_ngrams=args.forbid_duplicate_ngrams,
            forbid_ignore_set=forbid_ignore_set, ngram_size=args.ngram_size, min_len=args.min_len,
            len_vis_input=args.len_vis_input)
        
        model.load_state_dict(model_recover, strict=False)

        del model_recover

        if args.fp16:
            model.half()
        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
        torch.cuda.empty_cache()
        model.eval()
        eval_lst = []

        img_dat = [json.loads(l) for l in open(args.src_file)]

        img_idx = 0
        for src in img_dat:
            src_tk = os.path.join(src['img'])
            imgid = str(src['id'])
            eval_lst.append((img_idx, imgid, src_tk)) # img_idx: index 0~n, imgid: studyID, src_tk: img_path
            img_idx += 1

        input_lines = eval_lst
        next_i = 0
        output_lines = [""] * len(input_lines)
        total_batch = math.ceil(len(input_lines) / args.batch_size)

        print('start the caption evaluation...')
        with tqdm(total=total_batch) as pbar:
            while next_i < len(input_lines):
                _chunk = input_lines[next_i:next_i + args.batch_size] # 배치만큼 이미지 path 불러오기

                buf_id = [x[0] for x in _chunk]
                buf = [x[2] for x in _chunk]

                next_i += args.batch_size

                instances = []
                for instance in [(x, args.len_vis_input) for x in buf]:
                    for proc in bi_uni_pipeline:
                        instances.append(proc(instance))
                with torch.no_grad():
                    batch = batch_list_to_batch_tensors(
                        instances)
                    batch = [t.to(device) for t in batch]
                    input_ids, token_type_ids, position_ids, input_mask, task_idx, img, vis_pe = batch

                    if args.fp16:
                        img = img.half()
                        vis_pe = vis_pe.half()

                    # with amp.autocast():
                    traces = model(img, vis_pe, input_ids,  token_type_ids,
                            position_ids, input_mask, task_idx=task_idx)
                    
                    if args.beam_size > 1:
                        traces = {k: v.tolist() for k, v in traces.items()}
                        output_ids = traces['pred_seq']
                    else:
                        output_ids = traces[0].tolist()


                    for i in range(len(buf)):
                        w_ids = output_ids[i]
                        output_buf = tokenizer.convert_ids_to_tokens(w_ids)
                        output_tokens = []
                        for t in output_buf:
                            if t in ("[SEP]", "[PAD]"):
                                break
                            output_tokens.append(t)
                        output_sequence = ' '.join(detokenize(output_tokens))
                        output_lines[buf_id[i]] = output_sequence
                pbar.update(1)

        predictions = [{'image_id': tup[1], 'caption': output_lines[img_idx]} for img_idx, tup in enumerate(input_lines)]
        
        print("predictions",predictions)
        input("Stop!!")
        lang_stats = language_eval(args.dataset, predictions, args.model_recover_path.split('/')[-2]+'-'+args.split+'-'+args.model_recover_path.split('/')[-1].split('.')[-2], args.split)


if __name__ == "__main__":
    main()
