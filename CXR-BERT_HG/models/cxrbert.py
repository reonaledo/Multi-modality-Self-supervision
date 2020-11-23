import torch
import torch.nn as nn

from models.image import ImageEncoder_pool, ImageEncoder_cnn, Img_patch_embedding

from transformers import BertConfig, AlbertConfig # BertModel
from transformers.modeling_bert import BertModel, BertForMaskedLM
from transformers.modeling_albert import AlbertModel
from transformers.modeling_utils import PreTrainedModel, ModuleUtilsMixin



class ImageBertEmbeddings(nn.Module):
    def __init__(self, args, embeddings):  # self.img_embeddings = ImageBertEmbeddings(args, self.txt_embeddings)
        super().__init__()
        self.args = args
        # self.img_embeddings = nn.Linear(args.img_hidden_sz, args.hidden_size)
        self.img_embeddings = nn.Linear(args.img_hidden_sz, args.embedding_size)
        self.token_type_embeddings = embeddings.token_type_embeddings
        self.LayerNorm = embeddings.LayerNorm
        self.dropout = nn.Dropout(args.dropout_prob)

    def forward(self, input_imgs, token_type_ids):  # img_embed_out = self.img_embeddings(img, img_tok)
        bsz = input_imgs.size(0)
        seq_len = self.args.num_image_embeds
        # print('input_imgs.size:', input_imgs.size())

        imgs_embeddings = self.img_embeddings(input_imgs)  # torch.Size([32, 5, 768])
        token_type_embeddings = self.token_type_embeddings(token_type_ids)  # torch.Size([32, 5, 768])
        embeddings = imgs_embeddings + token_type_embeddings  # should be tensor
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        #print('embeddings:', embeddings.size())  # torch.Size([32, 5, 768])

        return embeddings

class CXRBertEncoder(nn.Module):  # MultimodalBertEncoder, BERT
# class CXRBertEncoder(PreTrainedModel):  # multi-gpu, save_pretrained problem.. not fixed
    def __init__(self, args):
        super().__init__()
        self.args = args
        # self.config = config
        if args.init_model == 'bert-base-uncased':
            bert = BertModel.from_pretrained('bert-base-uncased')

        elif args.init_model == 'BlueBERT':
            bert = BertModel.from_pretrained('bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12')
            # config = BertConfig.from_pretrained('/home/ubuntu/HG/cxr-bert/BlueBERT')
            # model_state_dict = torch.load('/home/ubuntu/HG/cxr-bert/BlueBERT/pytorch_model.bin')
            # bert = BertModel.from_pretrained('/home/ubuntu/HG/cxr-bert/BlueBERT', state_dict=model_state_dict, config=config)

        elif args.init_model == 'albert-base-v2':
            bert = AlbertModel.from_pretrained('albert-base-v2')

        else:
            bert = BertModel.from_pretrained(args.init_model)
        self.bert = bert
        self.txt_embeddings = self.bert.embeddings
        self.img_embeddings = ImageBertEmbeddings(args, self.txt_embeddings)

        #self.img_encoder = ImageEncoder_pool(args)
        # self.img_encoder = Img_patch_embedding(image_size=256, patch_size=32, dim=2048)  # ViT
        self.img_encoder = ImageEncoder_cnn(args)
        self.encoder = self.bert.encoder

        self.pooler = self.bert.pooler

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


class CXRBERT(PreTrainedModel):
#class CXRBERT(nn.Module):  # BERTLM, MultimodalBertClf
    """
    Multimodal BERT
    Masked Language Model + Image Text Matching
    """
    def __init__(self, config, args):
        super().__init__(config)
        self.enc = CXRBertEncoder(args)
        self.mlm = MaskedLanguageModel(args, args.hidden_size, args.vocab_size)
        self.itm = ImageTextMatching(args.hidden_size)

    def forward(self, cls_tok, input_txt, attn_mask, segment, input_img):
        x = self.enc(cls_tok, input_txt, attn_mask, segment, input_img)  # torch.Size([32, 512, 768]) [bsz, seq_len, hidden]

        return self.mlm(x), self.itm(x)

class MaskedLanguageModel(nn.Module):
    """
    (vocab_size) classification model
    """
    def __init__(self, args, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.linear.weight = CXRBertEncoder(args).txt_embeddings.word_embeddings.weight
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # x: torch.Size([8, 512, 768])
        # return: torch.Size([8, 512, 30522])
        # return self.softmax(self.linear(x))
        return self.linear(x)

class ImageTextMatching(nn.Module):
    """
    2-class classification model : Aligned, Not aligned
    """
    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # x: torch.Size([8, 512, 768])
        # return: torch.Size([8, 2])
        # return self.softmax(self.linear(x[:, 0]))  # [CLS] token, x_size: torch.Size([bsz, 768]) return [bsz, 2]
        return self.linear(x[:, 0])  # [CLS] token, x_size: torch.Size([bsz, 768]) return [bsz, 2]
