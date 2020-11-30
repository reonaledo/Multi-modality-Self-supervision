import torch
import torch.nn as nn

from models.image import ImageEncoder_cnn, ImageEncoder_pool, Img_patch_embedding

from transformers.modeling_auto import AutoModel
from transformers.modeling_bert import BertModel, BertForMaskedLM
from transformers.modeling_albert import AlbertModel
from transformers.modeling_utils import PreTrainedModel

class ImageBertEmbeddings(nn.Module):
    def __init__(self, args, embeddings):
        super().__init__()
        self.args = args
        self.img_embeddings = nn.Linear(args.img_hidden_sz, args.embedding_size)
        self.token_type_embeddings = embeddings.token_type_embeddings
        self.LayerNorm = embeddings.LayerNorm
        self.dropout = nn.Dropout(args.dropout_prob)

    def forward(self, input_imgs, token_type_ids):  # img_embed_out = self.img_embeddings(img, img_tok)
        imgs_embeddings = self.img_embeddings(input_imgs)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = imgs_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)  # bsz, num_img_embeds, hidden_sz
        return embeddings

class CXRBertEncoder(nn.Module):  # MultimodalBertEncoder, BERT
    def __init__(self, args):
        super().__init__()
        self.args = args

        if args.bert_model == "albert-base-v2":
            bert = AlbertModel.from_pretrained(args.bert_model)
        elif args.bert_model == "emilyalsentzer/Bio_ClinicalBERT":
            bert = AutoModel.from_pretrained(args.bert_model)
        elif args.bert_model == "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12":
            bert = AutoModel.from_pretrained(args.bert_model)
        else:
            bert = BertModel.from_pretrained(args.bert_model)  # bert-base-uncased, small, tiny

        self.bert = bert
        self.txt_embeddings = self.bert.embeddings
        self.img_embeddings = ImageBertEmbeddings(args, self.txt_embeddings)

        if args.img_encoder == 'ViT':
            img_size = args.img_size
            patch_sz = 32 if img_size == 512 else 16
            self.img_encoder = Img_patch_embedding(image_size=img_size, patch_size=patch_sz, dim=2048)
        else:
            self.img_encoder = ImageEncoder_cnn(args)

        self.encoder = self.bert.encoder
        self.pooler = self.bert.pooler

    def forward(self, cls_tok, input_txt, attn_mask, segment, input_img):

        extended_attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # (bsz, 1, 1, len(attn_mask))
        # extended_attn_mask = extended_attn_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attn_mask = extended_attn_mask.to(dtype=torch.float16)  # fp16 compatibility
        extended_attn_mask = (1.0 - extended_attn_mask) * -10000.0

        if self.args.cuda_devices == [1]:
            cuda = torch.device('cuda:1')
            img_tok = (torch.LongTensor(input_txt.size(0), (self.args.num_image_embeds)).fill_(0)).to(device=cuda)
            cls_segment = (torch.LongTensor(input_txt.size(0), 1).fill_(0)).to(device=cuda)

        else:
            img_tok = (torch.LongTensor(input_txt.size(0), (self.args.num_image_embeds)).fill_(0).cuda())
            cls_segment = (torch.LongTensor(input_txt.size(0), 1).fill_(0).cuda())

        cls_out = self.txt_embeddings(cls_tok, cls_segment)

        img = self.img_encoder(input_img)  # BxNx2048
        img_embed_out = self.img_embeddings(img, img_tok)  # bsz, num_img_embeds, hsz
        txt_embed_out = self.txt_embeddings(input_txt, segment)  # bsz, seq_len, hsz. inputs: bsz, seq_len
        encoder_input = torch.cat([cls_out, img_embed_out, txt_embed_out], 1)  # B x (TXT + IMG) x HID
        encoded_layers = self.encoder(
            encoder_input, extended_attn_mask, output_hidden_states=False
        )  # torch.Size([16, 512, 768]), torch.Size([16, 1, 1, 512])
        return encoded_layers[-1]  # bsz, max_len, hsz, encoded_layers[0]

class CXRBERT(PreTrainedModel):  # BERTLM, MultimodalBertClf
    """
    Multimodal BERT
    : Masked Language Model + Image Text Matching
    """
    def __init__(self, config, args):
        super().__init__(config)
        self.enc = CXRBertEncoder(args)
        self.mlm = MaskedLanguageModel(args, args.hidden_size, args.vocab_size)
        self.itm = ImageTextMatching(args.hidden_size)

        self.bertformlm = BertForMaskedLM.from_pretrained(args.bert_model).cls

    def forward(self, cls_tok, input_txt, attn_mask, segment, input_img):
        x = self.enc(cls_tok, input_txt, attn_mask, segment, input_img)  # bsz, max_len, hidden
        return self.mlm(x), self.itm(x)
        # return self.bertformlm(x), self.itm(x)

class MaskedLanguageModel(nn.Module):
    """
    (vocab_size) classification model
    """
    def __init__(self, args, hidden, vocab_size):
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.linear.weight = CXRBertEncoder(args).txt_embeddings.word_embeddings.weight
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.linear(x)  # x: torch.Size([8, 512, 768]), return: torch.Size([8, 512, 30522])

class ImageTextMatching(nn.Module):
    """
    2-class classification model : Aligned, Not aligned
    """
    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.linear(x[:, 0])  # [CLS], x_size: [bsz, max_len, hsz] -> [bsz, hsz], return [bsz, 2]
