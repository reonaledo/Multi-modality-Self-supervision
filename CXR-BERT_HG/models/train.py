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

from models.cxrbert import CXRBERT
from models.optim_schedule import ScheduledOptim

from transformers.optimization import AdamW
from transformers import BertConfig, AlbertConfig, AutoConfig
from transformers.modeling_bert import BertForMaskedLM

class CXRBERT_Trainer():
    def __init__(self, args, bert, train_dataloader, test_dataloader=None):
        self.args = args

        cuda_condition = torch.cuda.is_available() and args.with_cuda

        if self.args.cuda_devices == [1]:
            self.device = torch.device("cuda:1" if cuda_condition else "cpu")
        else:
            self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # TODO: remove after check
        self.bert = bert  # CXRBertEncoder

        if args.bert_model == "albert-base-v2":
            config = AlbertConfig.from_pretrained(args.bert_model)
        elif args.bert_model == "emilyalsentzer/Bio_ClinicalBERT":
            config = AutoConfig.from_pretrained(args.bert_model)
        elif args.bert_model == "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12":
            config = AutoConfig.from_pretrained(args.bert_model)
        else:
            config = BertConfig.from_pretrained(args.bert_model)  # bert-base, small, tiny

        self.model = CXRBERT(config, args).to(self.device)
        wandb.watch(self.model)

        if args.cuda_devices == [0, 1]:
            if args.with_cuda and torch.cuda.device_count() > 1:
                print("Using %d GPUS for BERT" % torch.cuda.device_count())
                self.model = nn.DataParallel(self.model, device_ids=args.cuda_devices)
        else:
            print(f"Using {args.cuda_devices} GPU for BERT")

        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # TODO: IMG-SGD
        self.optimizer = AdamW(self.model.parameters(), lr=args.lr)
        # self.optim_schedule = ScheduledOptim(self.optimizer, args.hidden_size, n_warmup_steps=args.warmup_steps)

        self.mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.itm_criterion = nn.CrossEntropyLoss()

        self.log_freq = args.log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        str_code = 'Train' if train else "Test"
        self.model.train() if train else self.model.eval()

        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc=f'EP_{str_code}:{epoch}',
                              total=len(data_loader),
                              bar_format='{l_bar}{r_bar}')
        avg_loss = 0.0
        total_correct = 0
        total_element = 0
        total_mlm_correct = 0
        total_mlm_element = 0

        scaler = amp.GradScaler()
        for i, data in data_iter:
            cls_tok, input_ids, txt_labels, attn_masks, img, segment, is_aligned = data

            cls_tok = cls_tok.to(self.device)
            input_ids = input_ids.to(self.device)
            txt_labels = txt_labels.to(self.device)
            attn_masks = attn_masks.to(self.device)
            img = img.to(self.device)
            segment = segment.to(self.device)
            is_aligned = is_aligned.to(self.device)

            with amp.autocast():
                # mlm_output: bsz, max_len, vocab_sz, itm_output: bsz, 2
                mlm_output, itm_output = self.model(cls_tok, input_ids, attn_masks, segment, img)

                mlm_loss = self.mlm_criterion(mlm_output.transpose(1, 2), txt_labels)  # txt_labels: bsz, max_len
                itm_loss = self.itm_criterion(itm_output, is_aligned)  # is_aligned: torch.Size([8])

                loss = itm_loss + mlm_loss

            if train:
                # self.optim_schedule.zero_grad()
                self.optimizer.zero_grad()
                # loss.backward()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                # self.optim_schedule.step_and_update_lr()
                # self.optim_schedule.update_lr()
                scaler.update()

            avg_loss += loss.item()

            # MLM prediction accuracy
            eq = (mlm_output.argmax(dim=-1).eq(txt_labels)).cpu().numpy()
            txt_labels_np = txt_labels.cpu().numpy()
            for bs, label in enumerate(txt_labels_np):
                index = np.where(label == -100)[0]
                f_label = np.delete(label, index)
                f_eq = np.delete(eq[bs], index)
                total_mlm_correct += f_eq.sum()
                total_mlm_element += len(f_label)

            # ITM prediction accuracy
            correct = itm_output.argmax(dim=-1).eq(is_aligned).sum().item()
            total_correct += correct
            total_element += is_aligned.nelement()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": round(avg_loss / (i + 1), 3),
                "mlm_avg_acc": round(total_mlm_correct / total_mlm_element * 100, 3),
                "itm_avg_acc": round(total_correct / total_element * 100, 3),
                "loss": round(loss.item(), 3),
                "mlm_loss": round(mlm_loss.item(), 3),
                "itm_loss": round(itm_loss.item(), 3),
            }

            if train:
                wandb.log({
                    "avg_loss": avg_loss / (i + 1),
                    "mlm_acc": total_mlm_correct / total_mlm_element * 100,
                    "itm_acc": total_correct / total_element * 100,
                    "mlm_loss": mlm_loss.item(),
                    "itm_loss": itm_loss.item(),
                    "loss": loss.item(),
                })

            else:
                wandb.log({
                    "eval_avg_loss": avg_loss / (i + 1),
                    "eval_mlm_acc": total_mlm_correct / total_mlm_element * 100,
                    "eval_itm_acc": total_correct / total_element * 100,
                    "eval_mlm_loss": mlm_loss.item(),
                    "eval_itm_loss": itm_loss.item(),
                    "eval_loss": loss.item(),
                })

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))

        # print(f'EP{epoch}_{str_code}, '
        #       f'avg_loss = {round(avg_loss / len(data_iter), 3)}, '
        #       f'total_mlm_acc = {round(total_mlm_correct / total_mlm_element * 100.0, 3)}')

    def save(self, epoch, file_path):
        if self.args.cuda_devices == [0, 1]:  # multi GPU
            # torch.nn.modules.module.ModuleAttributeError: 'DataParallel' object has no attribute 'save_pretrained'
            self.model.module.save_pretrained(file_path)
        else:
            self.model.save_pretrained(file_path)

        output_path = os.path.join(file_path, f'cxrbert_ep{epoch}.pt')
        # state = {
        #     "epoch": epoch,
        #     # "state_dict": self.model.module.state_dict(),  # DataParallel
        #     "state_dict": self.model.state_dict(),
        #     "optimizer": self.optimizer.state_dict(),
        #     # TODO: scheduler .... save or not
        #     # "scheduler": self.optim_schedule.state_dict(),  # state=True, if use optim.scheduler
        #     # "scheduler": self.scheduler.state_dict(),
        # }
        # torch.save(state, output_path)

        print(f'EP: {epoch} Model saved on {file_path}')
