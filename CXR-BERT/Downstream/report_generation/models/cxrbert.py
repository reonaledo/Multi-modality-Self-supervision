import torch
import torch.nn as nn

from models.image import ImageEncoder

from transformers import BertModel

class ImageBertEmbeddings(nn.Module):
    def __init__(self, args, embeddings):  # self.img_embeddings = ImageBertEmbeddings(args, self.txt_embeddings)
        super().__init__()
        self.args = args
        self.img_embeddings = nn.Linear(args.img_hidden_sz, args.hidden_size)
        self.token_type_embeddings = embeddings.token_type_embeddings
        self.LayerNorm = embeddings.LayerNorm
        self.dropout = nn.Dropout(args.dropout_prob)

    def forward(self, input_imgs, token_type_ids):  # img_embed_out = self.img_embeddings(img, img_tok)
        bsz = input_imgs.size(0)
        seq_len = self.args.num_image_embeds

        imgs_embeddings = self.img_embeddings(input_imgs)  # torch.Size([32, 5, 768])
        token_type_embeddings = self.token_type_embeddings(token_type_ids)  # torch.Size([32, 5, 768])
        embeddings = imgs_embeddings + token_type_embeddings  # should be tensor
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        #print('embeddings:', embeddings.size())  # torch.Size([32, 5, 768])

        return embeddings

class CXRBertEncoder(nn.Module):  # MultimodalBertEncoder, BERT
    def __init__(self, args):
        super().__init__()
        self.args = args
        bert = BertModel.from_pretrained(args.bert_model)
        self.txt_embeddings = bert.embeddings
        self.img_embeddings = ImageBertEmbeddings(args, self.txt_embeddings)
        self.img_encoder = ImageEncoder(args)
        self.encoder = bert.encoder
        self.pooler = bert.pooler

    def forward(self, input_txt, attn_mask, segment, input_img):
        bsz = input_txt.size(0)

        print("input_img",input_img.size())
        print("input_txt",input_txt)
        print("attn_mask",attn_mask)

        print("segment",segment)
        input("SSSSS")

        extended_attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # (bsz, 1, 1, len(attn_mask))
        extended_attn_mask = extended_attn_mask.to(dtype=next(self.parameters()).dtype)
        extended_attn_mask = (1.0 - extended_attn_mask) * -10000.0

        img_tok = (
            torch.LongTensor(input_txt.size(0), self.args.num_image_embeds)
            .fill_(0).cuda()
        )  # TODO: check, img_tok is same as token_type_id, segment_id?


        img = self.img_encoder(input_img)  # BxNx2048
        print("img",img)
        input("SSSSS")

        img_embed_out = self.img_embeddings(img, img_tok)  # img_embed_out: torch.Size([32, 5, 768])
        print("img_embed_out",img_embed_out)
        input("SSSSS")

        txt_embed_out = self.txt_embeddings(input_txt, segment)  # txt_embed_out: torch.Size([32, 507, 768])
        print("txt_embed_out",img_ttxt_embed_outok)
        input("SSSSS")

        encoder_input = torch.cat([txt_embed_out, img_embed_out],1)  # TODO: Check B x (TXT + IMG) x HID

        encoded_layers = self.encoder(
            encoder_input, extended_attn_mask, output_hidden_states=False
        )  # in mmbt: output_all_encoded_layers=False, but the argument was changed in recent Transformers
        # encoded_layers[-1] is encoded_layers[0]

        #return self.pooler(encoded_layers[-1])  # torch.Size([32, 768])
        return encoded_layers[-1]  # torch.Size([32, 512, 768])


class CXRBERT(nn.Module):  # BERTLM, MultimodalBertClf
    """
    Multimodal BERT
    Masked Language Model + Image Text Matching
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.enc = CXRBertEncoder(args)
        self.mlm = MaskedLanguageModel(args.hidden_size, args.vocab_size)
        self.itm = ImageTextMatching(args.hidden_size)

    # def forward(self, x, segment_label):  # TODO: Think x should be (txt+img) and followed segment_label
    #     x = self.enc(x, segment_label)  # TODO: Change Argument: input_txt, attn_mask, segment, input_img
    #     return self.mlm(x), self.itm(x)

    def forward(self, input_txt, attn_mask, segment, input_img):
        x = self.enc(input_txt, attn_mask, segment, input_img)  # torch.Size([32, 512, 768]) [bsz, seq_len, hidden]
        return self.mlm(x), self.itm(x)

class MaskedLanguageModel(nn.Module):
    """
    (vocab_size) classification model
    """
    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # x: torch.Size([8, 512, 768])
        # return: torch.Size([8, 512, 30522])
        return self.softmax(self.linear(x))


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
        return self.softmax(self.linear(x[:, 0]))  # [CLS] token, x_size: torch.Size([bsz, 768])
