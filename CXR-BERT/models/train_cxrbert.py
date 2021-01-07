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
        self.bert = bert # CXRBertEncoder

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

        for param in self.model.enc.img_encoder.parameters():
            param.requires_grad = not args.freeze  #not freeze_img

        for param in self.model.enc.encoder.parameters():
            param.requires_grad = not args.freeze #freeze_txt_all  #not freeze_txt

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

        self.gradient_accumulation_steps = args.gradient_accumulation_steps

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
        count = 0 
        scaler = amp.GradScaler()

        # print("len(data_iter)",len(data_iter))
        # input("STSOP!!!")
        len_data_iter = len(data_iter)
        
        if train:
            self.model.train()
            for i, data in data_iter:
                cls_tok, input_ids, txt_labels, attn_masks, img, segment, is_aligned, input_ids_itm, sep_tok = data

                cls_tok = cls_tok.to(self.device)
                input_ids = input_ids.to(self.device)
                txt_labels = txt_labels.to(self.device)
                attn_masks = attn_masks.to(self.device)
                img = img.to(self.device)
                segment = segment.to(self.device)
                is_aligned = is_aligned.to(self.device)
                input_ids_itm = input_ids_itm.to(self.device)
                sep_tok = sep_tok.to(self.device)
                self.optimizer.zero_grad()
                
                with amp.autocast():
                    mlm_output, itm_output = self.model(cls_tok, input_ids, attn_masks, segment, img, sep_tok)
                    ###_, itm_output = self.model(cls_tok, input_ids_itm, attn_masks, segment, img)  # [bsz, hidden_sz] -> [bsz, 2] ..?

                    # print('mlm_output.size:', mlm_output.size())  # [bsz, seq_len, vocab_sz]
                    # print('txt_labels', txt_labels.size()) # torch.Size([16, 512])
                    if self.args.mlm_task and self.args.itm_task ==False:
                        mlm_loss = self.mlm_criterion(mlm_output.transpose(1, 2), txt_labels)
                        loss = mlm_loss
                    
                    if self.args.itm_task and self.args.mlm_task==False:
                        itm_loss = self.itm_criterion(itm_output, is_aligned)
                        loss = itm_loss
                    
                    if self.args.mlm_task and self.args.itm_task:
                        # TODO: weight each loss, mlm > itm
                        mlm_loss = self.mlm_criterion(mlm_output.transpose(1, 2), txt_labels)
                        itm_loss = self.itm_criterion(itm_output, is_aligned)
                        loss = itm_loss + mlm_loss
                    
                    if self.gradient_accumulation_steps > 1:
                        loss = loss / self.gradient_accumulation_steps
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

            avg_loss += loss.item()

            if self.args.itm_task:
                # itm prediction accuracy
                correct = itm_output.argmax(dim=-1).eq(is_aligned).sum().item()
                total_correct += correct
                total_element += is_aligned.nelement()
                count += 1
                batch_itm_out = [element.item() for element in itm_output.argmax(dim=-1).flatten()]
                align_lt = [element.item() for element in is_aligned.flatten()]
                matched_itm = [index for index, (e1, e2) in enumerate(zip(batch_itm_out, align_lt)) if e1 == e2]

            if self.args.mlm_task:
                eq = (mlm_output.argmax(dim=-1).eq(txt_labels)).cpu().numpy()
                txt_labels_np = txt_labels.cpu().numpy()
                for bs, label in enumerate(txt_labels_np):
                    index = np.where(label == -100)[0]
                    f_label = np.delete(label, index)
                    f_eq = np.delete(eq[bs], index)
                    total_mlm_correct += f_eq.sum()
                    total_mlm_element += len(f_label)
                
                out_list = [element.item() for element in mlm_output.argmax(dim=-1).flatten()]
                label_list = [element.item() for element in txt_labels.flatten()]
                masked_index2 = [itr for itr, val in enumerate(label_list) if val != -100] #input 중 masking된 친구들의 index값 

                aa, bb = [], []
                for itr in masked_index2:
                    aa.append(out_list[itr])
                    bb.append(label_list[itr])

                matched_lm = [index for index, (e1, e2) in enumerate(zip(aa, bb)) if e1 == e2]
        else:
            self.model.eval()
            with torch.no_grad():
                for i, data in data_iter:
                    cls_tok, input_ids, txt_labels, attn_masks, img, segment, is_aligned, input_ids_itm, sep_tok = data
                    cls_tok = cls_tok.to(self.device)
                    input_ids = input_ids.to(self.device)
                    txt_labels = txt_labels.to(self.device)
                    attn_masks = attn_masks.to(self.device)
                    img = img.to(self.device)
                    segment = segment.to(self.device)
                    is_aligned = is_aligned.to(self.device)
                    input_ids_itm = input_ids_itm.to(self.device)
                    sep_tok = sep_tok.to(self.device)
                    with amp.autocast():
                        mlm_output, itm_output = self.model(cls_tok, input_ids, attn_masks, segment, img, sep_tok)
                        ###_, itm_output = self.model(cls_tok, input_ids_itm, attn_masks, segment, img)  # [bsz, hidden_sz] -> [bsz, 2] ..?

                        # print('mlm_output.size:', mlm_output.size())  # [bsz, seq_len, vocab_sz]
                        # print('txt_labels', txt_labels.size()) # torch.Size([16, 512])
                        if self.args.mlm_task and self.args.itm_task ==False:
                            mlm_loss = self.mlm_criterion(mlm_output.transpose(1, 2), txt_labels)
                            loss = mlm_loss
                        
                        if self.args.itm_task and self.args.mlm_task==False:
                            itm_loss = self.itm_criterion(itm_output, is_aligned)
                            loss = itm_loss
                        
                        if self.args.mlm_task and self.args.itm_task:
                            # TODO: weight each loss, mlm > itm
                            mlm_loss = self.mlm_criterion(mlm_output.transpose(1, 2), txt_labels)
                            itm_loss = self.itm_criterion(itm_output, is_aligned)
                            loss = itm_loss + mlm_loss
                        
                        if self.gradient_accumulation_steps > 1:
                            loss = loss / self.gradient_accumulation_steps
                    
                avg_loss += loss.item()

                if self.args.itm_task:
                    # itm prediction accuracy
                    correct = itm_output.argmax(dim=-1).eq(is_aligned).sum().item()
                    total_correct += correct
                    total_element += is_aligned.nelement()
                    count += 1
                    batch_itm_out = [element.item() for element in itm_output.argmax(dim=-1).flatten()]
                    align_lt = [element.item() for element in is_aligned.flatten()]
                    matched_itm = [index for index, (e1, e2) in enumerate(zip(batch_itm_out, align_lt)) if e1 == e2]

                if self.args.mlm_task:
                    eq = (mlm_output.argmax(dim=-1).eq(txt_labels)).cpu().numpy()
                    txt_labels_np = txt_labels.cpu().numpy()
                    for bs, label in enumerate(txt_labels_np):
                        index = np.where(label == -100)[0]
                        f_label = np.delete(label, index)
                        f_eq = np.delete(eq[bs], index)
                        total_mlm_correct += f_eq.sum()
                        total_mlm_element += len(f_label)
                    
                    out_list = [element.item() for element in mlm_output.argmax(dim=-1).flatten()]
                    label_list = [element.item() for element in txt_labels.flatten()]
                    masked_index2 = [itr for itr, val in enumerate(label_list) if val != -100] #input 중 masking된 친구들의 index값 

                    aa, bb = [], []
                    for itr in masked_index2:
                        aa.append(out_list[itr])
                        bb.append(label_list[itr])

                    matched_lm = [index for index, (e1, e2) in enumerate(zip(aa, bb)) if e1 == e2]

        print("avg loss per epoch",round(avg_loss / len_data_iter, 3))
        print("avg acc per epoch",round(total_correct / total_element * 100, 3))

        if self.args.mlm_task and self.args.itm_task:
            if train:
                post_fix = {
                    "epoch": epoch,
                    "avg_loss": round(avg_loss / len_data_iter, 3),
                    # "itm_itr_acc": round(len(matched_itm)/len(align_lt)* 100, 3),
                    # "mlm_itr_acc": round(len(matched_lm)/len(masked_index2)* 100, 3),
                    "mlm_avg_acc": round(total_mlm_correct / total_mlm_element * 100, 3),
                    "itm_avg_acc": round(total_correct / total_element * 100, 3),
                    # "loss": round(loss.item(), 3),
                    # "mlm_loss": round(mlm_loss.item(), 3),
                    # "itm_loss": round(itm_loss.item(), 3),
                    # "lr": round(self.optimizer.state_dict()['param_groups'][0]['lr'], 3),
                }
                wandb.log({
                    "avg_loss": avg_loss / len_data_iter,
                    # "itm_itr_acc": (len(matched_itm)/len(align_lt)* 100),
                    # "mlm_itr_acc": (len(matched_lm)/len(masked_index2)* 100),
                    "itm_acc": total_correct / total_element * 100,
                    "mlm_acc": total_mlm_correct / total_mlm_element * 100,
                    # "mlm_loss": mlm_loss.item(),
                    # "itm_loss": itm_loss.item(),
                    # "loss": loss.item(),
                }, step=epoch)
                if i % self.log_freq == 0:
                    data_iter.write(str(post_fix))
        
            else:
                wandb.log({
                    "eval_avg_loss": avg_loss /len_data_iter,
                    # "eval_mlm_itr_acc": (len(matched_lm)/len(masked_index2)* 100),
                    # "eval_mlm_loss": mlm_loss.item(),
                    # "eval_itm_itr_acc": (len(matched_itm)/len(align_lt)* 100),
                    "eval_itm_acc": total_correct / total_element * 100,
                    "eval_mlm_acc": total_mlm_correct / total_mlm_element * 100,
                    # "eval_itm_loss": itm_loss.item(),
                    # "eval_loss": loss.item()
                }, step=epoch)


        if self.args.itm_task and self.args.mlm_task ==False:
            if train:
                post_fix = {
                "epoch": epoch,
                "avg_loss": round(avg_loss / len_data_iter, 3),
                # "itm_itr_acc": round(len(matched_itm)/len(align_lt)* 100, 3),
                "itm_avg_acc": round(total_correct / total_element * 100, 3),
                # "loss": round(loss.item(), 3),
                # "itm_loss": round(itm_loss.item(), 3),
                }
                wandb.log({
                    "avg_loss": avg_loss / len_data_iter,
                    # "itm_itr_acc": (len(matched_itm)/len(align_lt)* 100),
                    "itm_epoch_acc": total_correct / total_element * 100,
                    # "itm_loss": itm_loss.item(),
                    # "loss": loss.item(),
                }, step=epoch)
                if i % self.log_freq == 0:
                    data_iter.write(str(post_fix))
        
            else:
                wandb.log({
                    "eval_avg_loss": avg_loss / len_data_iter,
                    # "eval_itm_itr_acc": (len(matched_itm)/len(align_lt)* 100),
                    "eval_itm_epoch_acc": total_correct / total_element * 100,
                    # "eval_itm_loss": itm_loss.item(),
                    # "eval_loss": loss.item()
                }, step=epoch)

        if self.args.mlm_task and self.args.itm_task ==False:
            if train:
                post_fix = {
                    "epoch": epoch,
                    "avg_loss": round(avg_loss / len_data_iter, 3),
                    # "mlm_itr_acc": round(len(matched_lm)/len(masked_index2)* 100, 3),
                    "mlm_epoch_acc": round(total_mlm_correct / total_mlm_element * 100, 3),
                    # "loss": round(loss.item(), 3),
                    # "mlm_loss": round(mlm_loss.item(), 3),
                }
                wandb.log({
                    "avg_loss": avg_loss / len_data_iter,
                    # "itm_itr_acc": (len(matched_itm)/len(align_lt)* 100),
                    # "mlm_itr_epoch_acc": (len(matched_lm)/len(masked_index2)* 100),
                    "mlm_epoch_acc": total_mlm_correct / total_mlm_element * 100,
                    # "mlm_loss": mlm_loss.item(),
                    # "loss": loss.item(),
                }, step=epoch)
                if i % self.log_freq == 0:
                    data_iter.write(str(post_fix))
        
            else:
                wandb.log({
                    "eval_avg_loss": avg_loss / len_data_iter,
                    # "eval_mlm_epoch_acc": (len(matched_lm)/len(masked_index2)* 100),
                    "eval_mlm_epoch_acc": total_mlm_correct / total_mlm_element * 100,
                    # "eval_mlm_loss": mlm_loss.item(),
                    # "eval_loss": loss.item()
                }, step=epoch)
                    
                                    
        # print(f'EP{epoch}_{str_code}, '
        #     f'avg_loss = {round(avg_loss / len(data_iter), 3)}, '
        #     f'itm_itr_acc = {round(len(matched_itm)/len(align_lt)* 100 ,3)},'
        #     f'mlm_itr_acc = {round(len(matched_lm)/len(masked_index2)* 100 ,3)},'
        #     f'total_mlm_acc = {round(total_mlm_correct / total_mlm_element * 100.0, 3)}, '
        #     f'total_itm_acc = {round(total_correct / total_element * 100.0, 3)}')

        
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
        
        if  torch.cuda.device_count() > 1:
            self.model.module.save_pretrained(file_path)
        else:

        # self.model.save_pretrained()
            self.model.save_pretrained(file_path)  # torch.nn.modules.module.ModuleAttributeError: 'DataParallel' object has no attribute 'save_pretrained'
        # print(f'EP: {epoch} Model saved on {output_path}')
        os.chmod(file_path+'/pytorch_model.bin', 0o777)
        print(f'EP: {epoch} Model saved on {file_path}')
        return file_path
