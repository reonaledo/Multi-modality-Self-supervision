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
from models.cxrbert import CXRBERT, CXRConfig
from models.optim_schedule import ScheduledOptim

from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import BertConfig, AlbertConfig, AutoConfig
from transformers.modeling_bert import BertForMaskedLM

class CXRBERT_Trainer():
    def __init__(self, args, train_dataloader, test_dataloader=None):
        self.args = args

        cuda_condition = torch.cuda.is_available() and args.with_cuda

        self.device = torch.device("cuda" if cuda_condition else "cpu")
        print('Current cuda device ', torch.cuda.current_device())  # check


        # TODO: remove after check
        # self.bert = bert  # CXRBertEncoder

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

        # config = AutoConfig.from_pretrained('/home/ubuntu/HG/cxr-bert/output/2020-12-03 23:39:25.785091')

        self.model = CXRBERT(config, args).to(self.device)

        # TODO: freeze, ATTEND !
        # for param in self.model.enc.img_encoder.parameters():
        #     param.requires_grad = not args.freeze  # not freeze_img
        #
        # for param in self.model.enc.encoder.parameters():
        #     param.requires_grad = not args.freeze  # freeze_txt_all  #not freeze_txt


        wandb.watch(self.model)

        if args.with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=args.cuda_devices)

        self.train_data = train_dataloader
        self.test_data = test_dataloader

        self.optimizer = AdamW(self.model.parameters(), lr=args.lr)

        # self.optimizer = AdamW(self.model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2),
        #                        eps=args.eps, weight_decay=args.weight_decay)

        # num_training_steps = len(self.train_data) * args.epochs
        #
        # self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=args.warmup_steps,
        #                                                  num_training_steps=num_training_steps)

        self.mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.itm_criterion = nn.CrossEntropyLoss()

        self.log_freq = args.log_freq
        self.step_cnt = 0

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):

        self.model.train()
        # self.optimizer.zero_grad()
        train_losses = []
        train_itm_loss = []
        train_mlm_loss = []

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

            # print('***attn_masks***:', attn_masks.size())
            # print(attn_masks)
            # print('***input_ids***:', input_ids.size())
            # print('***is_aligned***:', is_aligned.size())

            mlm_output, itm_output = self.model(cls_tok, input_ids, attn_masks, segment, img, sep_tok)
            if self.args.mlm_task and self.args.itm_task == False:
                mlm_loss = self.mlm_criterion(mlm_output.transpose(1, 2), txt_labels)
                loss = mlm_loss
                print('only mlm_loss')

            if self.args.itm_task and self.args.mlm_task == False:
                itm_loss = self.itm_criterion(itm_output, is_aligned)
                loss = itm_loss
                print('only itm_loss')

            if self.args.mlm_task and self.args.itm_task:
                mlm_loss = self.mlm_criterion(mlm_output.transpose(1, 2), txt_labels)
                itm_loss = self.itm_criterion(itm_output, is_aligned)
                train_mlm_loss.append(mlm_loss.item())
                train_itm_loss.append(itm_loss.item())

                loss = itm_loss + mlm_loss
                print('mlm,itm loss')

            train_losses.append(loss.item())
            self.optimizer.zero_grad()  # above
            loss.backward()
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
                "avg_mlm_loss": np.mean(train_mlm_loss),
                "avg_itm_loss": np.mean(train_itm_loss),
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
            eval_mlm_loss = []
            eval_itm_loss = []
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
                    eval_mlm_loss.append(valid_mlm_loss.item())
                    eval_itm_loss.append(valid_itm_loss.item())

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
                    "eval_mlm_loss": np.mean(eval_mlm_loss),
                    "eval_itm_loss": np.mean(eval_itm_loss),
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


# """
# Construct CXR-BERT or BertForMaskedLM, Training and Saving
# """
# import os
# import tqdm
# import wandb
# import datetime
# import numpy as np
#
# import torch
# import torch.nn as nn
# import torch.cuda.amp as amp
#
# # from models.cxrbert import CXRBERT
# from models.cxrbert_origin import CXRBERT, CXRConfig
# from models.optim_schedule import ScheduledOptim
#
# from transformers.optimization import AdamW
# from transformers import BertConfig, AlbertConfig, AutoConfig
# from transformers.modeling_bert import BertForMaskedLM
#
# class CXRBERT_Trainer():
#     def __init__(self, args, train_dataloader, test_dataloader=None):
#         self.args = args
#
#         cuda_condition = torch.cuda.is_available() and args.with_cuda
#
#         # if self.args.cuda_devices == [1]:
#         #     self.device = torch.device("cuda:1" if cuda_condition else "cpu")
#         # else:
#         #     self.device = torch.device("cuda:0" if cuda_condition else "cpu")
#
#         self.device = torch.device("cuda" if cuda_condition else "cpu")
#         print('Current cuda device ', torch.cuda.current_device())  # check
#
#
#         # TODO: remove after check
#         # self.bert = bert  # CXRBertEncoder
#
#         if args.bert_model == "albert-base-v2":
#             config = AlbertConfig.from_pretrained(args.bert_model)
#         elif args.bert_model == "emilyalsentzer/Bio_ClinicalBERT":
#             config = AutoConfig.from_pretrained(args.bert_model)
#         elif args.bert_model == "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12":
#             config = AutoConfig.from_pretrained(args.bert_model)
#         elif args.bert_model == "bert-small-scratch":
#             config = BertConfig.from_pretrained("google/bert_uncased_L-4_H-512_A-8")
#         # elif args.bert_model == "load_pretrained_model":
#         #     config = AutoConfig.from_pretrained('/home/ubuntu/HG/cxr-bert/output/2020-12-04 00:36:04.520462')
#         else:
#             config = BertConfig.from_pretrained(args.bert_model)  # bert-base, small, tiny
#
#         # config = AutoConfig.from_pretrained('/home/ubuntu/HG/cxr-bert/output/2020-12-03 23:39:25.785091')
#
#         self.model = CXRBERT(config, args).to(self.device)
#         wandb.watch(self.model)
#
#         # if args.cuda_devices == [0, 1]:
#         #     if args.with_cuda and torch.cuda.device_count() > 1:
#         #         print("Using %d GPUS for BERT" % torch.cuda.device_count())
#         #         self.model = nn.DataParallel(self.model, device_ids=args.cuda_devices)
#         # else:
#         #     print(f"Using {args.cuda_devices} GPU for BERT")
#
#         # Multi GPU
#         if args.with_cuda and torch.cuda.device_count() > 1:
#             print("Using %d GPUS for BERT" % torch.cuda.device_count())
#             self.model = nn.DataParallel(self.model, device_ids=args.cuda_devices)
#
#         # print(f"Using {args.cuda_devices} GPU for BERT")
#
#         self.train_data = train_dataloader
#         self.test_data = test_dataloader
#
#         # TODO: IMG-SGD
#         self.optimizer = AdamW(self.model.parameters(), lr=args.lr)
#         # self.optimizer = nn.SGD(self.model.enc.img_encoder.parameters())
#         # self.optim_schedule = ScheduledOptim(self.optimizer, args.hidden_size, n_warmup_steps=args.warmup_steps)
#
#         self.mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
#         self.itm_criterion = nn.CrossEntropyLoss()
#
#         self.log_freq = args.log_freq
#
#         print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
#
#     def train(self, epoch):
#         self.iteration(epoch, self.train_data)
#
#     def test(self, epoch):
#         self.iteration(epoch, self.test_data, train=False)
#
#     def iteration(self, epoch, data_loader, train=True):
#         str_code = 'Train' if train else "Test"
#         self.model.train() if train else self.model.eval()
#
#         data_iter = tqdm.tqdm(enumerate(data_loader),
#                               desc=f'EP_{str_code}:{epoch}',
#                               total=len(data_loader),
#                               bar_format='{l_bar}{r_bar}')
#         avg_loss = 0.0
#         total_correct = 0
#         total_element = 0
#         total_mlm_correct = 0
#         total_mlm_element = 0
#         len_data_iter = len(data_iter)
#
#         scaler = amp.GradScaler()
#         for i, data in data_iter:
#             cls_tok, input_ids, txt_labels, attn_masks, img, segment, is_aligned, sep_tok = data
#
#             cls_tok = cls_tok.to(self.device)
#             input_ids = input_ids.to(self.device)
#             txt_labels = txt_labels.to(self.device)
#             attn_masks = attn_masks.to(self.device)
#             img = img.to(self.device)
#             segment = segment.to(self.device)
#             is_aligned = is_aligned.to(self.device)
#             sep_tok = sep_tok.to(self.device)
#
#             with amp.autocast():
#                 # mlm_output: bsz, max_len, vocab_sz, itm_output: bsz, 2
#                 mlm_output, itm_output = self.model(cls_tok, input_ids, attn_masks, segment, img, sep_tok)
#
#                 mlm_loss = self.mlm_criterion(mlm_output.transpose(1, 2), txt_labels)  # txt_labels: bsz, max_len
#                 itm_loss = self.itm_criterion(itm_output, is_aligned)  # is_aligned: torch.Size([8])
#
#                 if self.args.mlm_task and self.args.itm_task:
#                     loss = mlm_loss + itm_loss
#                 if self.args.mlm_task and not self.args.itm_task:
#                     loss = mlm_loss
#                 if not self.args.mlm_task and self.args.itm_task:
#                     loss = itm_loss
#
#             if train:
#                 # self.optim_schedule.zero_grad()
#                 self.optimizer.zero_grad()
#                 # loss.backward()
#                 scaler.scale(loss).backward()
#                 scaler.step(self.optimizer)
#                 # self.optim_schedule.step_and_update_lr()
#                 # self.optim_schedule.update_lr()
#                 scaler.update()
#
#             avg_loss += loss.item()
#
#             # MLM prediction accuracy
#             eq = (mlm_output.argmax(dim=-1).eq(txt_labels)).cpu().numpy()
#             txt_labels_np = txt_labels.cpu().numpy()
#             for bs, label in enumerate(txt_labels_np):
#                 index = np.where(label == -100)[0]
#                 f_label = np.delete(label, index)
#                 f_eq = np.delete(eq[bs], index)
#                 total_mlm_correct += f_eq.sum()
#                 total_mlm_element += len(f_label)
#
#             # ITM prediction accuracy
#             correct = itm_output.argmax(dim=-1).eq(is_aligned).sum().item()
#             total_correct += correct
#             total_element += is_aligned.nelement()
#
#             post_fix = {
#                 "epoch": epoch,
#                 "iter": i,
#                 "avg_loss": round(avg_loss / (i + 1), 3),
#                 "mlm_avg_acc": round(total_mlm_correct / total_mlm_element * 100, 3),
#                 "itm_avg_acc": round(total_correct / total_element * 100, 3),
#                 "loss": round(loss.item(), 3),
#                 "mlm_loss": round(mlm_loss.item(), 3),
#                 "itm_loss": round(itm_loss.item(), 3),
#             }
#         if self.args.itm_task and self.args.mlm_task:  # both ITM and MLM
#             if train:
#                 wandb.log({
#                     # "avg_loss": avg_loss / (i + 1),
#                     "avg_loss": avg_loss / len_data_iter,
#                     "mlm_acc": total_mlm_correct / total_mlm_element * 100,
#                     "itm_acc": total_correct / total_element * 100,
#                 }, step=epoch)
#
#             else:
#                 wandb.log({
#                     # "eval_avg_loss": avg_loss / (i + 1),
#                     "eval_avg_loss": avg_loss / len_data_iter,
#                     "eval_mlm_acc": total_mlm_correct / total_mlm_element * 100,
#                     "eval_itm_acc": total_correct / total_element * 100,
#                 }, step=epoch)
#
#             if self.args.mlm_task and not self.args.itm_task:  # only MLM task
#                 if train:
#                     wandb.log({
#                         # "avg_loss": avg_loss / (i + 1),
#                         "avg_loss": avg_loss / len_data_iter,
#                         "mlm_acc": total_mlm_correct / total_mlm_element * 100,
#                     }, step=epoch)
#
#                 else:
#                     wandb.log({
#                         # "eval_avg_loss": avg_loss / (i + 1),
#                         "eval_avg_loss": avg_loss / len_data_iter,
#                         "eval_mlm_acc": total_mlm_correct / total_mlm_element * 100,
#                     }, step=epoch)
#
#             if not self.args.mlm_task and self.args.itm_task:  # only ITM task
#                 if train:
#                     wandb.log({
#                         # "avg_loss": avg_loss / (i + 1),
#                         "avg_loss": avg_loss / len_data_iter,
#                         "itm_acc": total_correct / total_element * 100,
#                     }, step=epoch)
#
#                 else:
#                     wandb.log({
#                         # "eval_avg_loss": avg_loss / (i + 1),
#                         "eval_avg_loss": avg_loss / len_data_iter,
#                         "eval_itm_acc": total_correct / total_element * 100,
#                     }, step=epoch)
#
#
#             if i % self.log_freq == 0:
#                 data_iter.write(str(post_fix))
#
#         # print(f'EP{epoch}_{str_code}, '
#         #       f'avg_loss = {round(avg_loss / len(data_iter), 3)}, '
#         #       f'total_mlm_acc = {round(total_mlm_correct / total_mlm_element * 100.0, 3)}')
#
#     def save(self, epoch, file_path):
#         # if self.args.cuda_devices == [0, 1]:  # multi GPU
#
#         if torch.cuda.device_count() > 1:
#             # torch.nn.modules.module.ModuleAttributeError: 'DataParallel' object has no attribute 'save_pretrained'
#             self.model.module.save_pretrained(file_path)
#             print(f'Multi_EP: {epoch} Model saved on {file_path}')
#         else:
#             self.model.save_pretrained(file_path)
#             print(f'Single_EP: {epoch} Model saved on {file_path}')
#         os.chmod(file_path + '/pytorch_model.bin', 0o777)
#
#
#         # self.model.save_pretrained(file_path)
#         # print(f'Single_EP: {epoch} Model saved on {file_path}')
#
#         # output_path = os.path.join(file_path, f'cxrbert_ep{epoch}.pt')
#         # state = {
#         #     "epoch": epoch,
#         #     # "state_dict": self.model.module.state_dict(),  # DataParallel
#         #     "state_dict": self.model.state_dict(),
#         #     "optimizer": self.optimizer.state_dict(),
#         #     # TODO: scheduler .... save or not
#         #     # "scheduler": self.optim_schedule.state_dict(),  # state=True, if use optim.scheduler
#         #     # "scheduler": self.scheduler.state_dict(),
#         # }
#         # torch.save(state, output_path)
