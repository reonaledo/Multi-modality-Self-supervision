"""

class CXRBertEncoder(nn.Module):  # MultimodalBertEncoder, BERT
    def __init__(self, args):
        super().__init__()
        self.args = args
        # self.config = config
        if args.init_model == 'bert-base-uncased':
            bert = BertModel.from_pretrained('bert-base-uncased')
        elif args.init_model == 'BlueBERT':
            config = BertConfig.from_pretrained('BlueBERT')
            model_state_dict = torch.load('BlueBERT/pytorch_model.bin')
            bert = BertModel.from_pretrained('BlueBERT', state_dict=model_state_dict, config=config)
        # elif args.init_model == 'albert-base-v2':
        #     bert = AlbertModel.from_pretrained('albert-base-v2')

        self.txt_embeddings = bert.embeddings
        self.img_embeddings = ImageBertEmbeddings(args, self.txt_embeddings)

        #self.img_encoder = ImageEncoder_pool(args)
        # self.img_encoder = Img_patch_embedding(image_size=256, patch_size=32, dim=2048)  # ViT
        self.img_encoder = ImageEncoder_cnn(args)
        self.encoder = bert.encoder

        self.pooler = bert.pooler

    def forward(self, cls_tok, input_txt, attn_mask, segment, input_img):
        extended_attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # (bsz, 1, 1, len(attn_mask))
        # extended_attn_mask = extended_attn_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attn_mask = extended_attn_mask.to(dtype=torch.float16)  # fp16 compatibility
        extended_attn_mask = (1.0 - extended_attn_mask) * -10000.0

        img_tok = (torch.LongTensor(input_txt.size(0), (self.args.num_image_embeds)).fill_(0).cuda())
        cls_segment = (torch.LongTensor(input_txt.size(0), 1).fill_(0).cuda())

        img = self.img_encoder(input_img)  # BxNx2048
        cls_out = self.txt_embeddings(cls_tok, cls_segment)
        img_embed_out = self.img_embeddings(img, img_tok)  # img_embed_out: torch.Size([32, 5, 768])
        txt_embed_out = self.txt_embeddings(input_txt, segment)  # txt_embed_out: torch.Size([32, 507, 768])
        # print('input_txt', input_txt.size())  # torch.Size([bsz, input_txt_len])
        # print('segment', segment.size())  # torch.Size([bsz, input_txt_len])
        encoder_input = torch.cat([cls_out, img_embed_out, txt_embed_out], 1)  # TODO: Check B x (TXT + IMG) x HID
        # torch.Size([16, 512, 768]), torch.Size([16, 1, 1, 512])
        encoded_layers = self.encoder(
            encoder_input, extended_attn_mask, output_hidden_states=False
        )  # in mmbt: output_all_encoded_layers=False, but the argument was changed in recent Transformers
        # encoded_layers[-1] is encoded_layers[0]

        #return self.pooler(encoded_layers[-1])  # torch.Size([32, 768])
        return encoded_layers[-1]  # torch.Size([32, 512, 768])

"""

import torch
import torch.nn as nn

import os
import datetime
import argparse

from transformers import BertConfig, AutoConfig
from transformers.modeling_bert import BertModel
from transformers.modeling_auto import AutoModel

from models.cxrbert import CXRBERT

def model_load(args):
    bert =BertModel.from_pretrained('bert-base-uncased')

    bert_embedding = bert.embeddings
    bert_encoder = bert.encoder

    config = BertConfig.from_pretrained('/home/ubuntu/HG/cxr-bert/BlueBERT')
    model_state_dict = torch.load('/home/ubuntu/HG/cxr-bert/BlueBERT/pytorch_model.bin')
    bluebert = BertModel.from_pretrained('/home/ubuntu/HG/cxr-bert/BlueBERT', state_dict=model_state_dict, config=config)
    """
    /home/ubuntu/HG/cxr-bert/output/2020-11-09 15:29:19.925411
    """
    c_config = AutoConfig.from_pretrained('/home/ubuntu/HG/cxr-bert/output/2020-11-09 17:06:40.912986')
    # print(c_config)
    print('________________________________________')
    c_model_state_dict = torch.load('/home/ubuntu/HG/cxr-bert/output/2020-11-09 17:06:40.912986/pytorch_model.bin')
    # print(c_model_state_dict)

    # for key, value in c_model_state_dict.items():
    #     print(key, value)
    cxr_bert = CXRBERT.from_pretrained('/home/ubuntu/HG/cxr-bert/output/2020-11-09 17:06:40.912986', state_dict=c_model_state_dict, config=c_config, args=args)
    cxr_bert_encoder = cxr_bert.enc.encoder

    print('________________________________________')

    bluebert_embedding = bluebert.embeddings
    bluebert_encoder = bluebert.encoder
    # print(bluebert_encoder)

    input_txt = torch.rand(3, 10)
    segment = torch.rand(3, 10)

    input_txt = torch.LongTensor([[1,2,3,4,5],[4,5,6,2,1],[7,8,5,2,1]])
    segment = torch.LongTensor([[1,1,1,0,0],[1,1,1,1,0],[1,0,0,0,0]])
    # print(input_txt)
    # print(segment)

    bluebert_embed_out = bluebert_embedding(input_txt, segment)
    bert_embed_out = bert_embedding(input_txt, segment)
    # print(bluebert_embed_out)
    # print(bert_embed_out)
    # print(bluebert_embed_out == bert_embed_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dataset", type=str, default='/home/ubuntu/HG/cxr-bert/dset/img_512/cxr_train.json',
                        help="train dataset for training")
    parser.add_argument("--test_dataset", type=str, default='/home/ubuntu/HG/cxr-bert/dset/img_512/cxr_test.json',
                        help='test dataset for evaluate train set')

    # output_path = 'output/' + str(datetime.now())
    # if not os.path.exists(output_path):
    #     os.mkdir(output_path)
    # parser.add_argument("--output_path", type=str, default=output_path, help="ex)path/to/save/model")

    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n inter: setting n")
    parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: True or False")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")

    parser.add_argument("--epochs", type=int, default=200, help='number of epochs')
    parser.add_argument("--batch_size", type=int, default=32, help="number of batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="dataloader worker size")

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout_prob", type=float, default=0.1)

    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--lr_patience", type=int, default=10)  # lr_scheduler.ReduceLROnPlateau
    parser.add_argument("--lr_factor", type=float, default=0.2)
    # TODO: img-SGD, txt-AdamW
    parser.add_argument("--beta1", type=float, default=0.9, help="adams first beta value")
    parser.add_argument("--beta2", type=float, default=0.999, help="adams first beta value")
    parser.add_argument("--eps", type=float, default=1e-6, help="adams epsilon")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="weight_decay of adam")  # 0.01 , AdamW

    # TODO: loading BlueBERT
    parser.add_argument("--bert_model", type=str, default='bert-base-uncased',
                        choices=["bert-base-uncased", "BlueBERT", "albert-base-v2"])  # for tokenizing ...

    parser.add_argument("--init_model", type=str, default='BlueBERT',
                        choices=["bert-base-uncased", "BlueBERT", "albert-base-v2"])

    parser.add_argument("--max_seq_len", type=int, default=512, help="maximum sequence len")
    parser.add_argument("--vocab_size", type=int, default=30522, choices=[30522, 30000])
    parser.add_argument("--embedding_size", type=int, default=768, choices=[768, 128])
    parser.add_argument("--hidden_size", type=int, default=768)

    parser.add_argument("--img_hidden_sz", type=int, default=2048)
    parser.add_argument("--img_embed_pool_type", type=str, default="max", choices=["max", "avg"])
    parser.add_argument("--num_image_embeds", type=int, default=0)  # TODO: 224x224, output fiber 49...
    #-------------------------------------------------------------------------------------------
    # TODO: ...!
    parser.add_argument("--gradient_accumulation_steps", type=int, default=24)  # loss, optimizer.step() slowly
    parser.add_argument("--warmup", type=float, default=0.1)  # optimizer = BertAdam(warmup=args.warmup)
    parser.add_argument("--seed", type=int, default=123)

    #parser.add_argument("--patience", type=int, default=10)  # n_no_improve > args.patienc: break
    # parser.add_argument("--weight_classes", type=int, default=1)  # for multi-label classification weight

    args = parser.parse_args()

    model_load(args)