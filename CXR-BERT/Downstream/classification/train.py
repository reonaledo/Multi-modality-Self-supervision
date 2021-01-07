#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]= "6,7"
scenari = 3


import argparse
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_pretrained_bert import BertAdam

from data.helpers import get_data_loaders
from models import get_model
from utils.logger import create_logger
from utils.utils import *
import wandb


def get_args(parser):
    parser.add_argument("--batch_sz", type=int, default=32)

    parser.add_argument("--data_path", type=str, default='/home/mimic-cxr/dataset/image_preprocessing/',
                        help="train dataset for training")
    # parser.add_argument("--data_path", type=str, default="/path/to/data_dir/")

    parser.add_argument("--bert_model", type=str, default="bert-base-uncased", choices=["bert-base-uncased", "bert-large-uncased"])
    parser.add_argument("--init_model", type=str, default="google/bert_uncased_L-4_H-512_A-8",
                        choices=["bert-base-uncased", "BlueBERT", "albert-base-v2", "emilyalsentzer/Bio_ClinicalBERT"
                                 "google/bert_uncased_L-4_H-512_A-8", "google/bert_uncased_L-2_H-128_A-2"])

    parser.add_argument("--drop_img_percent", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    # parser.add_argument("--embed_sz", type=int, default=300)

    parser.add_argument("--embed_sz", type=int, default=512, choices=[768, 512, 128])

    parser.add_argument("--freeze_img", type=int, default=0)
    parser.add_argument("--freeze_txt", type=int, default=0)

    parser.add_argument("--freeze_img_all", type=str, default=True)
    parser.add_argument("--freeze_txt_all", type=str, default=True)

    parser.add_argument("--glove_path", type=str, default="/path/to/glove_embeds/glove.840B.300d.txt")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=24)
    parser.add_argument("--hidden", nargs="*", type=int, default=[])
    # parser.add_argument("--hidden_sz", type=int, default=768)
    parser.add_argument("--hidden_sz", type=int, default=512, choices=[768, 512, 128])

    parser.add_argument("--img_embed_pool_type", type=str, default="avg", choices=["max", "avg"])
    parser.add_argument("--img_hidden_sz", type=int, default=2048)
    parser.add_argument("--include_bn", type=int, default=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--lr_patience", type=int, default=2)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--model", type=str, default="mmbt", choices=["bow", "img", "bert", "concatbow", "concatbert", "mmbt"])
    parser.add_argument("--n_workers", type=int, default=24)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--savedir", type=str, default="/home/mimic-cxr/model/downstream_model/classification")


    ############################################################################################################
    # # scenari 1> Bidirectional Attention(img36, txt128)_bert_small
    if scenari == 1:
        parser.add_argument("--loaddir", type=str, default="/home/mimic-cxr/model/12.31_img36_txt128_full_attn")
        parser.add_argument("--name", type=str, default="scenario_1_class")
        parser.add_argument("--num_image_embeds", type=int, default=49)

    # # scenari 2> Partially Auto-regressive (img36, txt128)_bert_small
    elif scenari == 2:
        parser.add_argument("--loaddir", type=str, default="/home/mimic-cxr/model/12.31_img36_txt128_red_sqr")
        parser.add_argument("--name", type=str, default="scenario_2_class")
        parser.add_argument("--num_image_embeds", type=int, default=49)

    # # scenari 3> seq2seq Attention (img36, txt128)_bert_small
    elif scenari == 3:
        parser.add_argument("--loaddir", type=str, default="/home/mimic-cxr/model/12.31_img36_txt128_s2s")
        parser.add_argument("--name", type=str, default="scenario_3_class")
        parser.add_argument("--num_image_embeds", type=int, default=49)

    # # scenari 4> total 50% (img36, txt128)_bert_small
    elif scenari == 4:
        parser.add_argument("--loaddir", type=str, default="/home/mimic-cxr/model/12.31_img36_txt128_random_50prob")
        parser.add_argument("--name", type=str, default="scenario_4_class")
        parser.add_argument("--num_image_embeds", type=int, default=49)

    ############################################################################################################

    elif scenari == 5:
        parser.add_argument("--loaddir", type=str, default="/home/mimic-cxr/model/12.31_img36_txt128_full_attn")
        parser.add_argument("--name", type=str, default="scenario_1_class_text_only")
        parser.add_argument("--num_image_embeds", type=int, default=0)

    # # scenari 2> Partially Auto-regressive (img36, txt128)_bert_small
    elif scenari == 6:
        parser.add_argument("--loaddir", type=str, default="/home/mimic-cxr/model/12.31_img36_txt128_red_sqr")
        parser.add_argument("--name", type=str, default="scenario_2_class_text_only")
        parser.add_argument("--num_image_embeds", type=int, default=0)

    # # scenari 3> seq2seq Attention (img36, txt128)_bert_small
    elif scenari == 7:
        parser.add_argument("--loaddir", type=str, default="/home/mimic-cxr/model/12.31_img36_txt128_s2s")
        parser.add_argument("--name", type=str, default="scenario_3_class_text_only")
        parser.add_argument("--num_image_embeds", type=int, default=0)

    # # scenari 4> total 50% (img36, txt128)_bert_small
    elif scenari == 8:
        parser.add_argument("--loaddir", type=str, default="/home/mimic-cxr/model/12.31_img36_txt128_random_50prob")
        parser.add_argument("--name", type=str, default="scenario_4_class_text_only")
        parser.add_argument("--num_image_embeds", type=int, default=0)

    # parser.add_argument("--loaddir", type=str, default="/home/mimic-cxr")

    parser.add_argument("--seed", type=int, default=123)
    # parser.add_argument("--task", type=str, default="mmimdb", choices=["mmimdb", "vsnli", "food101", cxr])
    parser.add_argument("--task_type", type=str, default="multilabel", choices=["multilabel", "classification"])
    parser.add_argument("--warmup", type=float, default=0.1)
    parser.add_argument("--weight_classes", type=int, default=1)


def get_criterion(args, device):
    if args.task_type == "multilabel":
        if args.weight_classes:
            freqs = [args.label_freqs[l] for l in args.labels]
            label_weights = (torch.FloatTensor(freqs) / args.train_data_len) ** -1
            criterion = nn.BCEWithLogitsLoss(pos_weight=label_weights.to(device))
        else:
            criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    return criterion


def auc_graph():
    from scipy import interp
    from sklearn.metrics import auc
    from matplotlib import pyplot as plt
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes=15):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes=15)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes=15):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= 15 #n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    colors = ['aqua', 'darkorange', 'cornflowerblue','teal','indigo','magenta','lightblue','gold','tan','slategrey','gray','coral','peru','olivedrab','pink']
    
    for i, color in zip(range(n_classes=15), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    # plt.show()
    plt.savefig(args.savedir+'/auc_class.png')
    
def get_optimizer(model, args):
    if args.model in ["bert", "concatbert", "mmbt"]:
        total_steps = (
            args.train_data_len
            / args.batch_sz
            / args.gradient_accumulation_steps
            * args.max_epochs
        )
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0,},
        ]
        optimizer = BertAdam(
            optimizer_grouped_parameters,
            lr=args.lr,
            warmup=args.warmup,
            t_total=total_steps,
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    return optimizer


def get_scheduler(optimizer, args):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "max", patience=args.lr_patience, verbose=True, factor=args.lr_factor
    )

def computeAUROC(gt, pred, num_class):

    outAUROC = []

    for i in range(num_class):
        outAUROC.append(roc_auc_score(gt[:, i], pred[:, i]))

    return outAUROC


def model_eval(i_epoch, data, model, args, criterion, device, store_preds=False):
    with torch.no_grad():
        losses, preds, tgts = [], [], []
        outAUROC = []

        for batch in data:
            loss, out, tgt = model_forward(i_epoch, model, args, criterion, batch, device)
            losses.append(loss.item())

            if args.task_type == "multilabel":
                pred = torch.sigmoid(out).cpu().detach().numpy() > 0.5
            else:
                pred = torch.nn.functional.softmax(out, dim=1).argmax(dim=1).cpu().detach().numpy()

            preds.append(pred)
            tgt = tgt.cpu().detach().numpy()
            tgts.append(tgt)

    metrics = {"loss": np.mean(losses)}
    classACC = dict()

    if args.task_type == "multilabel":
        tgts = np.vstack(tgts)
        preds = np.vstack(preds)
        # print("tgts",tgts)
        # print("preds",preds)
        # input("STOP!")
        for i in range(args.n_classes):
            outAUROC.append(roc_auc_score(tgts[:, i], preds[:, i]))

        for i in range(0, len(outAUROC)):
            assert args.n_classes == len(outAUROC)
            classACC[args.labels[i]] = outAUROC[i]
            # print('******:', args.labels[i], metrics[args.labels[i]])

        
        metrics["micro_roc_auc"] = roc_auc_score(tgts, preds, average="micro")
        metrics["macro_roc_auc"] = roc_auc_score(tgts, preds, average="macro")
        metrics["macro_f1"] = f1_score(tgts, preds, average="macro")
        metrics["micro_f1"] = f1_score(tgts, preds, average="micro")
    else:
        tgts = [l for sl in tgts for l in sl]
        preds = [l for sl in preds for l in sl]
        metrics["acc"] = accuracy_score(tgts, preds)

    if store_preds:
        store_preds_to_disk(tgts, preds, args)

    return metrics, classACC, tgts, preds


def model_forward(i_epoch, model, args, criterion, batch, device):
    txt, segment, mask, img, tgt = batch
    
    freeze_img = i_epoch < args.freeze_img
    freeze_txt = i_epoch < args.freeze_txt

    model.to(device)

    if args.model == "bow":
        txt = txt.to(device)
        out = model(txt)
    elif args.model == "img":
        img = img.to(device)
        out = model(img)
    elif args.model == "concatbow":
        txt, img = txt.to(device), img.to(device)
        out = model(txt, img)
    elif args.model == "bert":
        txt, mask, segment = txt.to(device), mask.to(device), segment.to(device)
        out = model(txt, mask, segment)
    elif args.model == "concatbert":
        txt, img = txt.to(device), img.to(device)
        mask, segment = mask.to(device), segment.to(device)
        out = model(txt, mask, segment, img)
    else:
        assert args.model == "mmbt"

        # for param_tensor in model.state_dict():
        #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

        if args.num_image_embeds >0:
            for param in model.module.enc.img_encoder.parameters():
                param.requires_grad = args.freeze_img_all  #not freeze_img

        for param in model.module.enc.encoder.parameters():
            param.requires_grad = args.freeze_txt_all  #not freeze_txt

        txt, img = txt.to(device), img.to(device)
        mask, segment = mask.to(device), segment.to(device)
        out = model(txt, mask, segment, img)

    tgt = tgt.to(device)
    loss = criterion(out, tgt)
    return loss, out, tgt


def train(args):
    wandb.init(config=args, project="classification", entity="mimic-cxr")
    print("Training start!!")
    print(" # PID :", os.getpid())
    set_seed(args.seed)
    args.savedir = os.path.join(args.savedir, os.name)
    os.makedirs(args.savedir, exist_ok=True)

    train_loader, val_loader = get_data_loaders(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args)
    
    print("MMBT model : ")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    input("STOP!")

    criterion = get_criterion(args, device)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    logger = create_logger("%s/logfile.log" % args.savedir, args)
    # logger.info(model)
    torch.save(args, os.path.join(args.savedir, "args.bin"))
    
    start_epoch, global_step, n_no_improve, best_metric = 0, 0, 0, -np.inf

    if os.path.exists(os.path.join(args.loaddir, "pytorch_model.bin")):
        # checkpoint = torch.load(os.path.join(args.loaddir, "pytorch_model.bin"))
        # pretrained_dict = {k: v for k, v in model.load_state_dict(checkpoint).items()}
        # print("pretrained_dict",pretrained_dict)
        # input("STOP!")
        
        model.load_state_dict(torch.load(args.loaddir+"/pytorch_model.bin"), strict=False)
        # chkpoint = torch.load(args.loaddir+"/pytorch_model.bin")
        # for key in list(chkpoint.keys()):
        #     chkpoint[key.replace('module', '')] = chkpoint.pop(key)
        # model.load_state_dict(chkpoint, strict=False)
        
        print("This would load the trained model, then fine-tune the model.")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        input("STOP!")

    else:
        print("");print("")
        print("this option initilize the model with random value. train from scratch.")
        print("Loaded model : ")
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())

        # pretrained_dict = model.load_state_dict(checkpoint, strict=False)
    

    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
    
    # print("pretrained_dict",pretrained_dict)
    
    # new_model_dict.update(pretrained_dict)
    # new_model.load_state_dict(new_model_dict)

    print("freeze image?",args.freeze_img_all)
    print("freeze txt?",args.freeze_txt_all)
    model.to(device)
    logger.info("Training..")

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        
    for i_epoch in range(start_epoch, args.max_epochs):
        train_losses = []
        model.module.train()
        optimizer.zero_grad()

        for batch in tqdm(train_loader, total=len(train_loader)):
            loss, out, target = model_forward(i_epoch, model, args, criterion, batch, device)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            train_losses.append(loss.item())
            loss.backward()
            global_step += 1
            if global_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        model.eval()
        metrics, classACC, tgts, preds = model_eval(i_epoch, val_loader, model, args, criterion, device)
        logger.info("Train Loss: {:.4f}".format(np.mean(train_losses)))
        log_metrics("Val", metrics, args, logger)

        tuning_metric = (
            metrics["micro_f1"] if args.task_type == "multilabel" else metrics["acc"]
        )
        scheduler.step(tuning_metric)
        is_improvement = tuning_metric > best_metric
        if is_improvement:
            best_metric = tuning_metric
            n_no_improve = 0
        else:
            n_no_improve += 1

        wandb.log({
            "micro_roc_auc": metrics["micro_roc_auc"],
            "macro_roc_auc": metrics["macro_roc_auc"],
            "macro_f1 f1 scroe": metrics["macro_f1"],
            "micro f1 score": metrics["micro_f1"],
            "train loss": np.mean(train_losses),
            "class accuracy": classACC,
            # "ROC": wandb.sklearn.plot_roc(tgts, preds, labels=args.labels),
            # "ALL": wandb.sklearn.plot_classifier(metrics)
        })

        save_checkpoint(
            {
                "epoch": i_epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "n_no_improve": n_no_improve,
                "best_metric": best_metric,
            },
            is_improvement,
            args.savedir,
        )

        if n_no_improve >= args.patience:
            logger.info("No improvement. Breaking out of loop.")
            break

    load_checkpoint(model, os.path.join(args.savedir, "args.bin"))
    # model.eval()
    # for test_name, test_loader in test_loaders.items():
    #     test_metrics = model_eval(
    #         np.inf, test_loader, model, args, criterion, store_preds=True
    #     )
    #     log_metrics(f"Test - {test_name}", test_metrics, args, logger)


def cli_main():
    parser = argparse.ArgumentParser(description="Train Models")
    get_args(parser)
    args, remaining_args = parser.parse_known_args()
    assert remaining_args == [], remaining_args
    # print("")
    train(args)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    
    cli_main()