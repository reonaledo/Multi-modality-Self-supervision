"""
CXR-BERT training code,,,
"""
import os
import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam, AdamW, SGD
from torch.utils.data import DataLoader

from models.cxrbert import CXRBERT, CXRBertEncoder
from models.optim_schedule import ScheduledOptim



class CXRBERT_Trainer():
    """
    Pretrained CXRBert model with two pretraining tasks
    1. Masked Language Modeling
    2. Image Text Matching
    """
    def __init__(self, args, bert: CXRBertEncoder, train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10):
        """
        :param bert: BERT model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset loader
        :param test_dataloader: test dataset loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay parameter
        :param with_cuda: training with cuda
        :param cuda_devices
        :param log_freq: logginf frequency of the batch iteration
        """

        # Setup cuda device for BERT training
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        # This BERT model will be saved every epoch
        self.bert = bert  # CXRBertEncoder
        # Initialize the BERT Language Model wiht BERT model
        self.model = CXRBERT(args).to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Set the train, test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Set the Adam optimizer with hyper-param
        # TODO: TXT-AdamW, IMG-SGD
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, args.hidden_sz, n_warmup_steps=warmup_steps)

        # Use Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.NLLLoss(ignore_index=0)

        self.log_freq = log_freq

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

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc=f'EP_{str_code}:{epoch}',
                              total=len(data_loader),
                              bar_format='{l_bar}{r_bar}')

        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        for i, data in data_iter:

            # 0. batch_data will be sent into the device(GPU or CPU)
            input_ids, txt_labels, attn_masks, img, segment, is_aligned, input_ids_ITM = data

            # 1. forward the MLM and ITM
            mlm_output, _ = self.model(input_ids, attn_masks, segment, img)  # forward(self, input_txt, attn_mask, segment, input_img)
            _, itm_output = self.model(input_ids_ITM, attn_masks, segment, img)  # forward(self, input_txt, attn_mask, segment, input_img):

            # 2-1. NLL loss of predicting masked token word
            # TODO: mlm_output.transpose(1, 2)
            print('mlm_output:', mlm_output)
            print('txt_labels:', txt_labels)
            mlm_loss = self.criterion(mlm_output, txt_labels)

            # 2-2. NLL loss of is_aligned classification result
            print('itm_output:', itm_output)
            print('is_aligned:', is_aligned)
            itm_loss = self.criterion(itm_output, is_aligned)

            # 2-3. Summing mlm_loss and itm_loss
            loss = mlm_loss + itm_loss

            # 3. backward and optimization only in train
            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            # itm prediction accuracy
            correct = itm_output.argmax(dim=-1).eq(is_aligned).sum().item()
            avg_loss += loss.item()
            total_correct += correct
            total_element += is_aligned.nelement()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "avg_acc": total_correct / total_element * 100,
                "loss": loss.item()
            }

            if i % self.log_freq == 0: data_iter.write(str(post_fix))

        print(f'EP{epoch}_{str_code}, '
              f'avg_loss = {avg_loss / len(data_iter)}, '
              f'total_acc = {total_correct / total_element * 100.0}')

    def save(self, epoch, file_path='output/CXRBert_trained.model'):
        """
        Save the current BERT model on file_path
        """
        # TODO: check, filename & save state_dict() or whole, .cpu()
        filename = f'epoch_{epoch}.pth'
        output_path = os.path.join(file_path, filename)
        torch.save(self.bert.state_dict(), output_path)
        print(f'EP: {epoch} Model saved on {output_path}')
        return output_path
