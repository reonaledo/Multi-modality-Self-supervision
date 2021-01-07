"""
Construct CXR-BERT or BertForMaskedLM, Training and Saving
"""
import os
import tqdm
import wandb
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.cuda.amp as amp
# from models.cxrbert import CXRBERT
from models.cxrbert_origin import CXRBERT, CXRConfig
from models.optim_schedule import ScheduledOptim
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import BertConfig, AlbertConfig, AutoConfig
from transformers.modeling_bert import BertForMaskedLM

class CXRBERT_Trainer_origin():
    def __init__(self, args, train_dataloader, test_dataloader=None):
        self.args = args
        cuda_condition = torch.cuda.is_available() and args.with_cuda
        self.device = torch.device("cuda" if cuda_condition else "cpu")
        print('Current cuda device ', torch.cuda.current_device())  # check
        if args.bert_model == "albert-base-v2":
            config = AlbertConfig.from_pretrained(args.bert_model)
        elif args.bert_model == "emilyalsentzer/Bio_ClinicalBERT":
            config = AutoConfig.from_pretrained(args.bert_model)
        elif args.bert_model == "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12":
            config = AutoConfig.from_pretrained(args.bert_model)
        elif args.bert_model == "bert-small-scratch":
            config = BertConfig.from_pretrained("google/bert_uncased_L-4_H-512_A-8")
        # elif args.bert_model == "load_pretrained_model":
        #     config = AutoConfig.from_pretrained('/home/ubuntu/HG/cxr-bert/output/2020-12-04 00:36:04.520462')
        else:
            config = BertConfig.from_pretrained(args.bert_model)  # bert-base, small, tiny
        self.model = CXRBERT(config, args).to(self.device)

        for param_tensor in self.model.state_dict():
            print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())
        wandb.watch(self.model)
        if args.with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=args.cuda_devices)
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.optimizer = AdamW(self.model.parameters(), lr=args.lr)
        self.mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.itm_criterion = nn.CrossEntropyLoss()
        self.log_freq = args.log_freq
        self.step_cnt = 0
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
    
    def train(self, epoch):
        self.model.train()
        # self.optimizer.zero_grad()
        train_losses = []
        train_data_iter = tqdm.tqdm(enumerate(self.train_data),
                                    desc=f'EP_:{epoch}',
                                    total=len(self.train_data),
                                    bar_format='{l_bar}{r_bar}')
        total_correct = 0
        total_element = 0
        total_mlm_correct = 0
        total_mlm_element = 0
        total_valid_correct = 0
        total_valid_element = 0
        total_mlm_valid_correct = 0
        total_mlm_valid_element = 0
        for i, data in train_data_iter:
            cls_tok, input_ids, txt_labels, attn_masks, img, segment, is_aligned, sep_tok = data
            cls_tok = cls_tok.to(self.device)
            input_ids = input_ids.to(self.device)
            txt_labels = txt_labels.to(self.device)
            attn_masks = attn_masks.to(self.device)
            img = img.to(self.device)
            segment = segment.to(self.device)
            is_aligned = is_aligned.to(self.device)
            sep_tok = sep_tok.to(self.device)
            mlm_output, itm_output = self.model(cls_tok, input_ids, attn_masks, segment, img, sep_tok)
            if self.args.mlm_task and self.args.itm_task ==False:
                mlm_loss = self.mlm_criterion(mlm_output.transpose(1, 2), txt_labels)
                loss = mlm_loss
            if self.args.itm_task and self.args.mlm_task==False:
                itm_loss = self.itm_criterion(itm_output, is_aligned)
                loss = itm_loss
            if self.args.mlm_task and self.args.itm_task:
                mlm_loss = self.mlm_criterion(mlm_output.transpose(1, 2), txt_labels)
                itm_loss = self.itm_criterion(itm_output, is_aligned)
                loss = itm_loss + mlm_loss
            # if self.gradient_accumulation_steps > 1:
            #     loss = loss / self.gradient_accumulation_steps
            train_losses.append(loss.item())
            self.optimizer.zero_grad()  # above
            loss.backward()
            # global_step += 1
            # if global_step % self.gradient_accumulation_steps == 0:
            self.optimizer.step()
            # self.scheduler.step()
            if self.args.itm_task:
                correct = itm_output.argmax(dim=-1).eq(is_aligned).sum().item()
                total_correct += correct
                total_element += is_aligned.nelement()
            if self.args.mlm_task:
                eq = (mlm_output.argmax(dim=-1).eq(txt_labels)).cpu().numpy()
                txt_labels_np = txt_labels.cpu().numpy()
                for bs, label in enumerate(txt_labels_np):
                    index = np.where(label == -100)[0]
                    f_label = np.delete(label, index)
                    f_eq = np.delete(eq[bs], index)
                    total_mlm_correct += f_eq.sum()
                    total_mlm_element += len(f_label)
        print("avg loss per epoch", np.mean(train_losses))
        print("avg itm acc per epoch", round(total_correct / total_element * 100, 3))
        if self.args.mlm_task and self.args.itm_task:
            wandb.log({
                "avg_loss": np.mean(train_losses),
                "itm_acc": total_correct / total_element * 100,
                "mlm_acc": total_mlm_correct / total_mlm_element * 100
            }, step=epoch)
        if self.args.itm_task and self.args.mlm_task == False:
            wandb.log({
                "avg_loss": np.mean(train_losses),
                "itm_epoch_acc": total_correct / total_element * 100
            }, step=epoch)
        if self.args.mlm_task and self.args.itm_task == False:
            wandb.log({
                "avg_loss": np.mean(train_losses),
                "mlm_epoch_acc": total_mlm_correct / total_mlm_element * 100
            }, step=epoch)
        test_data_iter = tqdm.tqdm(enumerate(self.test_data),
                                   desc=f'EP_:{epoch}',
                                   total=len(self.test_data),
                                   bar_format='{l_bar}{r_bar}')
        self.model.eval()
        with torch.no_grad():
            eval_losses = []
            for i, data in test_data_iter:
                cls_tok, input_ids, txt_labels, attn_masks, img, segment, is_aligned, sep_tok = data
                cls_tok, input_ids, txt_labels, attn_masks, img = cls_tok.to(self.device), input_ids.to(
                    self.device), txt_labels.to(self.device), attn_masks.to(self.device), img.to(self.device)
                segment, is_aligned, sep_tok = segment.to(self.device), is_aligned.to(
                    self.device), sep_tok.to(self.device)
                mlm_output, itm_output = self.model(cls_tok, input_ids, attn_masks, segment, img, sep_tok)
                if self.args.mlm_task and self.args.itm_task == False:
                    valid_mlm_loss = self.mlm_criterion(mlm_output.transpose(1, 2), txt_labels)
                    valid_loss = valid_mlm_loss
                    print('only valid mlm loss')
                if self.args.itm_task and self.args.mlm_task == False:
                    valid_itm_loss = self.itm_criterion(itm_output, is_aligned)
                    valid_loss = valid_itm_loss
                    print('only valid itm loss')
                if self.args.mlm_task and self.args.itm_task:
                    # TODO: weight each loss, mlm > itm
                    valid_mlm_loss = self.mlm_criterion(mlm_output.transpose(1, 2), txt_labels)
                    valid_itm_loss = self.itm_criterion(itm_output, is_aligned)
                    valid_loss = valid_itm_loss + valid_mlm_loss
                    print('only valid mlm, itm loss')
                eval_losses.append(valid_loss.item())
                if self.args.itm_task:
                    valid_correct = itm_output.argmax(dim=-1).eq(is_aligned).sum().item()
                    total_valid_correct += valid_correct
                    total_valid_element += is_aligned.nelement()
                if self.args.mlm_task:
                    eq = (mlm_output.argmax(dim=-1).eq(txt_labels)).cpu().numpy()
                    txt_labels_np = txt_labels.cpu().numpy()
                    for bs, label in enumerate(txt_labels_np):
                        index = np.where(label == -100)[0]
                        f_label = np.delete(label, index)
                        f_eq = np.delete(eq[bs], index)
                        total_mlm_valid_correct += f_eq.sum()
                        total_mlm_valid_element += len(f_label)
            print("avg loss in testset", np.mean(eval_losses))
            print("avg itm acc in testset", round(total_valid_correct / total_valid_element * 100, 3))
            if self.args.mlm_task and self.args.itm_task:
                wandb.log({
                    "eval_avg_loss": np.mean(eval_losses),
                    "eval_itm_acc": total_valid_correct / total_valid_element * 100,
                    "eval_mlm_acc": total_mlm_valid_correct / total_mlm_valid_element * 100
                }, step=epoch)
            if self.args.itm_task and self.args.mlm_task == False:
                wandb.log({
                    "eval_avg_loss": np.mean(eval_losses),
                    "eval_itm_epoch_acc": total_valid_correct / total_valid_element * 100
                }, step=epoch)
            if self.args.mlm_task and self.args.itm_task == False:
                wandb.log({
                    "eval_avg_loss": np.mean(eval_losses),
                    "eval_mlm_epoch_acc": total_mlm_valid_correct / total_mlm_valid_element * 100
                }, step=epoch)
    def save(self, epoch, file_path):
        if torch.cuda.device_count() > 1:
            self.model.module.save_pretrained(file_path)
            print(f'Multi_EP: {epoch} Model saved on {file_path}')
        else:
            self.model.save_pretrained(file_path)
            print(f'Single_EP: {epoch} Model saved on {file_path}')
        os.chmod(file_path + '/pytorch_model.bin', 0o777)