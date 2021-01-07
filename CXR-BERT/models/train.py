import os
import tqdm
import wandb
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torch.optim import AdamW
import torch.cuda.amp as amp
# TODO: change when pytorch 1.6 installed instead of torch 1.6
# from optims.AdamW import AdamW

from models.cxrbert import CXRBERT
from models.optim_schedule import ScheduledOptim

from transformers import BertModel, BertConfig, AlbertConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import AdamW
from torch.utils.tensorboard import SummaryWriter

now = datetime.datetime.now()
# summary = SummaryWriter('./runs/1029')

class CXRBERT_Trainer():
    """
    Pretrained CXRBert model with two pretraining tasks
    1. Masked Language Modeling
    2. Image Text Matching
    """
    def __init__(self, args, bert, train_dataloader, test_dataloader=None):
        """
        :param bert: BERT model which you want to train
        :param train_dataloader: train dataset loader
        :param test_dataloader: test dataset loader [can be None]
        """
        self.args = args
        self.lr = args.lr
        cuda_condition = torch.cuda.is_available() and args.with_cuda
        self.device = torch.device("cuda:1" if cuda_condition else "cpu")

        # This BERT model will be saved every epoch
        self.bert = bert  # CXRBertEncoder

        # Initialize the BERT Language Model with BERT model
        # config = BertConfig.from_pretrained('/home/ubuntu/HG/cxr-bert/output/2020-11-03 00:00:05.110129')
        if args.init_model == "albert-base-v2":
            config = AlbertConfig.from_pretrained(args.init_model)
        else:
            config = BertConfig.from_pretrained(args.init_model)
        self.model = CXRBERT(config, args).to(self.device)
        wandb.watch(self.model)

        if args.with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=args.cuda_devices)

        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # TODO: IMG-SGD
        # self.optimizer = Adam(self.model.parameters(), lr=args.lr,
        #                       betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)

        # self.optimizer = AdamW(self.model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2),
        #                        eps=args.eps, weight_decay=args.weight_decay)
        self.optimizer = AdamW(self.model.parameters(), lr=args.lr)

        # self.optim_schedule = ScheduledOptim(self.optimizer, args.hidden_size, n_warmup_steps=args.warmup_steps)

        # self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=args.lr_patience,
        #                                                 factor=args.lr_factor, verbose=True)

        self.mlm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.itm_criterion = nn.CrossEntropyLoss()

        self.log_freq = args.log_freq
        self.step_cnt = 0

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or test
        if on train status, backward operation is activated
        and also auto save the model every epoch

        epoch: current epoch index
        data_loader: torch.utils.data.DataLoader for iteration
        train: boolean value of is train or test
        return: None
        """

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
            cls_tok, input_ids, txt_labels, attn_masks, img, segment, is_aligned, input_ids_itm = data

            cls_tok = cls_tok.to(self.device)
            input_ids = input_ids.to(self.device)
            txt_labels = txt_labels.to(self.device)
            attn_masks = attn_masks.to(self.device)
            img = img.to(self.device)
            segment = segment.to(self.device)
            is_aligned = is_aligned.to(self.device)
            input_ids_itm = input_ids_itm.to(self.device)

            with amp.autocast():
                mlm_output, itm_output = self.model(cls_tok, input_ids, attn_masks, segment, img)
                ###_, itm_output = self.model(cls_tok, input_ids_itm, attn_masks, segment, img)  # [bsz, hidden_sz] -> [bsz, 2] ..?

                # print('mlm_output.size:', mlm_output.size())  # [bsz, seq_len, vocab_sz]
                # print('txt_labels', txt_labels.size()) # torch.Size([16, 512])
                mlm_loss = self.mlm_criterion(mlm_output.transpose(1, 2), txt_labels)
                itm_loss = self.itm_criterion(itm_output, is_aligned)

                # TODO: weight each loss, mlm > itm
                loss = itm_loss + mlm_loss

            if train:
                # self.model.train()
                # self.optim_schedule.zero_grad()
                self.optimizer.zero_grad()                
                # loss.backward()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                # self.optim_schedule.step_and_update_lr()
                #self.optim_schedule.update_lr()  # self.optim_schedule.step_and_update_lr()
                scaler.update()
            else:
                self.model.eval()

            avg_loss += loss.item()

            # itm prediction accuracy
            correct = itm_output.argmax(dim=-1).eq(is_aligned).sum().item()
            total_correct += correct
            total_element += is_aligned.nelement()

            eq = (mlm_output.argmax(dim=-1).eq(txt_labels)).cpu().numpy()
            txt_labels_np = txt_labels.cpu().numpy()
            for bs, label in enumerate(txt_labels_np):
                index = np.where(label == -100)[0]
                f_label = np.delete(label, index)
                f_eq = np.delete(eq[bs], index)
                total_mlm_correct += f_eq.sum()
                total_mlm_element += len(f_label)

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": round(avg_loss / (i + 1), 3),
                "mlm_avg_acc": round(total_mlm_correct / total_mlm_element * 100, 3),
                "itm_avg_acc": round(total_correct / total_element * 100, 3),
                "loss": round(loss.item(), 3),
                "mlm_loss": round(mlm_loss.item(), 3),
                "itm_loss": round(itm_loss.item(), 3),
                # "lr": round(self.optimizer.state_dict()['param_groups'][0]['lr'], 3),
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
                    # "mlm_loss": round(loss.item(), 3),
                    "eval_mlm_loss": mlm_loss.item(),
                    "eval_itm_loss": itm_loss.item(),
                    "eval_loss": loss.item()
                })
        print(f'EP{epoch}_{str_code}, '
              f'avg_loss = {round(avg_loss / len(data_iter), 3)}, '
              f'total_mlm_acc = {round(total_mlm_correct / total_mlm_element * 100.0, 3)}, '
              f'total_itm_acc = {round(total_correct / total_element * 100.0, 3)}')

    def save(self, epoch, file_path):
        """
        Save the current BERT model on file_path
        """
        filename = f'cxrbert_ep{epoch}.pt'
        output_path = os.path.join(file_path, filename)
        # state = {
        #     "epoch": epoch,
        #     #"state_dict": self.model.module.state_dict(),  # DataParallel
        #     "state_dict": self.model.state_dict(),
        #     "optimizer": self.optimizer.state_dict(),
        #
        #     # TODO: scheduler state_dict()........
        #     # "scheduler": self.optim_schedule.state_dict(),  # state=True, if use optim.scheduler
        #     # "scheduler": self.scheduler.state_dict(),
        # }
        # torch.save(state, output_path)
        # PreTrainedModel.save_pretrained('./output/save_pretrained_test')
        # self.model.save_pretrained('output/save_pretrained_amp')

        # model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        # model_to_save.save
        self.model.module.save_pretrained(file_path)
        # self.model.save_pretrained()
        # self.model.save_pretrained(file_path)  # torch.nn.modules.module.ModuleAttributeError: 'DataParallel' object has no attribute 'save_pretrained'
        # print(f'EP: {epoch} Model saved on {output_path}')
        print(f'EP: {epoch} Model saved on {file_path}')
        return file_path
