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

from transformers import BertModel, BertConfig, AlbertConfig, AutoConfig, AutoModel
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

        cuda_condition = torch.cuda.is_available()
        self.device = torch.device("cuda" if cuda_condition else "cpu")
        print ('Current cuda device ', torch.cuda.current_device()) # check

        # This BERT model will be saved every epoch
        
        #*
        # self.bert = bert # CXRBertEncoder 

        # Initialize the BERT Language Model with BERT model
        # config = BertConfig.from_pretrained('/home/ubuntu/HG/cxr-bert/output/2020-11-03 00:00:05.110129')
        if args.init_model == "albert-base-v2":
            config = AlbertConfig.from_pretrained(args.init_model)
        else:
            # config = BertConfig.from_pretrained(args.init_model)
            config = AutoConfig.from_pretrained(args.init_model)
            # model = AutoModel.from_config(config)

        self.model = CXRBERT(config, args).to(self.device)

        for param_tensor in self.model.state_dict():
            print(param_tensor, "\t", self.model.state_dict()[param_tensor].size())

        #*
        # for param in self.model.enc.img_encoder.parameters():
        #     param.requires_grad = not args.freeze  #not freeze_img

        # for param in self.model.enc.encoder.parameters():
        #     param.requires_grad = not args.freeze #freeze_txt_all  #not freeze_txt

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

        # self.gradient_accumulation_steps = args.gradient_accumulation_steps

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
        count = 0 
        global_step = 0
        # scaler = amp.GradScaler()

        # len_data_iter = len(data_iter)
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


            # with amp.autocast():
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
            # self.optimizer.zero_grad()

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
                post_fix = {
                    "epoch": epoch,
                    "avg_loss": np.mean(train_losses),
                    "mlm_avg_acc": round(total_mlm_correct / total_mlm_element * 100, 3),
                    "itm_avg_acc": round(total_correct / total_element * 100, 3)
                }
                wandb.log({
                    "avg_loss": np.mean(train_losses),
                    "itm_acc": total_correct / total_element * 100,
                    "mlm_acc": total_mlm_correct / total_mlm_element * 100
                }, step=epoch)
                if i % self.log_freq == 0:
                    train_data_iter.write(str(post_fix))

        if self.args.itm_task and self.args.mlm_task ==False:
                post_fix = {
                "epoch": epoch,
                "avg_loss": np.mean(train_losses),
                "itm_avg_acc": round(total_correct / total_element * 100, 3)
                }
                wandb.log({
                    "avg_loss": np.mean(train_losses),
                    "itm_epoch_acc": total_correct / total_element * 100
                }, step=epoch)
                if i % self.log_freq == 0:
                    train_data_iter.write(str(post_fix))

        if self.args.mlm_task and self.args.itm_task ==False:
                post_fix = {
                    "epoch": epoch,
                    "avg_loss": np.mean(train_losses),
                    "mlm_epoch_acc": round(total_mlm_correct / total_mlm_element * 100, 3)
                }
                wandb.log({
                    "avg_loss": np.mean(train_losses),
                    "mlm_epoch_acc": total_mlm_correct / total_mlm_element * 100
                }, step=epoch)
                if i % self.log_freq == 0:
                    train_data_iter.write(str(post_fix))

        test_data_iter = tqdm.tqdm(enumerate(self.test_data),
                              desc=f'EP_:{epoch}',
                              total=len(self.test_data),
                              bar_format='{l_bar}{r_bar}')
        self.model.eval()
        with torch.no_grad():
            eval_losses = []
            for i, data in test_data_iter:
                cls_tok, input_ids, txt_labels, attn_masks, img, segment, is_aligned, sep_tok = data
                cls_tok,input_ids,txt_labels,attn_masks,img = cls_tok.to(self.device), input_ids.to(self.device), txt_labels.to(self.device), attn_masks.to(self.device), img.to(self.device)
                segment,is_aligned, sep_tok = segment.to(self.device), is_aligned.to(self.device), sep_tok.to(self.device)

                mlm_output, itm_output = self.model(cls_tok, input_ids, attn_masks, segment, img, sep_tok)
                if self.args.mlm_task and self.args.itm_task ==False:
                    valid_mlm_loss = self.mlm_criterion(mlm_output.transpose(1, 2), txt_labels)
                    valid_loss = valid_mlm_loss
                
                if self.args.itm_task and self.args.mlm_task==False:
                    valid_itm_loss = self.itm_criterion(itm_output, is_aligned)
                    valid_loss = valid_itm_loss
               
                if self.args.mlm_task and self.args.itm_task:
                    # TODO: weight each loss, mlm > itm
                    valid_mlm_loss = self.mlm_criterion(mlm_output.transpose(1, 2), txt_labels)
                    valid_itm_loss = self.itm_criterion(itm_output, is_aligned)
                    valid_loss = valid_itm_loss + valid_mlm_loss
                    
                # if self.gradient_accumulation_steps > 1:
                #     valid_loss = valid_loss / self.gradient_accumulation_steps
                
                # self.scheduler.step(valid_loss)
                eval_losses.append(valid_loss.item())

                if self.args.itm_task:
                    # itm prediction accuracy
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
            
            print("avg loss in testset",np.mean(eval_losses))
            print("avg itm acc in testset",round(total_valid_correct / total_valid_element * 100, 3))

            if self.args.mlm_task and self.args.itm_task:
                    wandb.log({
                        "eval_avg_loss": np.mean(eval_losses),
                        "eval_itm_acc": total_valid_correct / total_valid_element * 100,
                        "eval_mlm_acc": total_mlm_valid_correct / total_mlm_valid_element * 100
                    }, step=epoch)


            if self.args.itm_task and self.args.mlm_task ==False:
                    wandb.log({
                        "eval_avg_loss": np.mean(eval_losses),
                        "eval_itm_epoch_acc": total_valid_correct / total_valid_element * 100
                    }, step=epoch)

            if self.args.mlm_task and self.args.itm_task ==False:        
                    wandb.log({
                        "eval_avg_loss": np.mean(eval_losses),
                        "eval_mlm_epoch_acc": total_mlm_valid_correct / total_mlm_valid_element * 100
                    }, step=epoch)

    def save(self, epoch, file_path):
        if torch.cuda.device_count() > 1:
            self.model.module.save_pretrained(file_path)
        else:
            self.model.save_pretrained(file_path)
        print(f'EP: {epoch} Model saved on {file_path}')
        os.chmod(file_path+'/pytorch_model.bin', 0o777)