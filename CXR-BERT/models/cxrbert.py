import torch
import torch.nn as nn

from models.image import random_sample, Img_patch_embedding, fully_use_cnn

from transformers import BertConfig, AlbertConfig, AutoTokenizer, AutoModel, BertModel, AutoConfig
from transformers.modeling_utils import PreTrainedModel, ModuleUtilsMixin
from transformers.configuration_utils import PretrainedConfig


class ImageBertEmbeddings(nn.Module):
    def __init__(self, args, embeddings):  # self.img_embeddings = ImageBertEmbeddings(args, self.txt_embeddings)
        super().__init__()
        self.args = args
        self.img_embeddings = nn.Linear(args.img_hidden_sz, args.hidden_size)
        # self.img_embeddings = nn.Linear(2048, 512)
        
        if self.args.img_postion: self.position_embeddings = embeddings.position_embeddings

        self.token_type_embeddings = embeddings.token_type_embeddings
        self.LayerNorm = embeddings.LayerNorm
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_imgs, img_pos, token_type_ids):  # img_embed_out = self.img_embeddings(img, img_tok)

        imgs_embeddings = self.img_embeddings(input_imgs)  # torch.Size([32, 5, 768])
        token_type_embeddings = self.token_type_embeddings(token_type_ids)  # torch.Size([32, 5, 768])

        if self.args.img_postion:
            position_embeddings = self.position_embeddings(img_pos)
            embeddings = imgs_embeddings + position_embeddings + token_type_embeddings  # should be tensor
        else: 
            embeddings = imgs_embeddings + token_type_embeddings  # should be tensor
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

class CXRBertEncoder(nn.Module):  # MultimodalBertEncoder, BERT
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.new_segment_ids = args.new_segment_ids

        type_vocab_size = 6 if args.new_segment_ids else 2

        if args.from_scratch:
            config = BertConfig.from_pretrained(args.init_model)#, type_vocab_size = type_vocab_size)
            bert = AutoModel.from_config(config)
            print("the model will be trained from scratch!!")

        else:
            if args.init_model == 'bert-base-uncased':
                bert = BertModel.from_pretrained('bert-base-uncased')

            elif args.init_model == 'ClinicalBERT':
                bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

            elif args.init_model == 'BlueBERT':
                bert = BertModel.from_pretrained('bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12')#, type_vocab_size = type_vocab_size)
                # bert = BertModel.from_pretrained('bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12', type_vocab_size = type_vocab_size)

            elif args.init_model == 'google/bert_uncased_L-4_H-512_A-8':
                bert = AutoModel.from_pretrained("google/bert_uncased_L-4_H-512_A-8")
                # config = AutoConfig.from_pretrained("google/bert_uncased_L-4_H-512_A-8", type_vocab_size = type_vocab_size)
                # bert = AutoModel.from_pretrained(config)
                # bert = AutoModel.from_pretrained("google/bert_uncased_L-4_H-512_A-8", type_vocab_size = type_vocab_size)

            else:
                # bert = BertModel.from_pretrained(args.init_model, type_vocab_size = type_vocab_size)
                bert = BertModel.from_pretrained(args.init_model)#, type_vocab_size = type_vocab_size)

        self.txt_embeddings = bert.embeddings
        self.img_embeddings = ImageBertEmbeddings(args, self.txt_embeddings)
        
        if args.img_encoding == 'random_sample':
            self.img_encoder = random_sample(args)

        elif args.img_encoding == 'Img_patch_embedding':
            self.img_encoder = Img_patch_embedding(image_size=512, patch_size=32, dim=2048)  # ViT
            
        elif args.img_encoding == 'fully_use_cnn':    
            self.img_encoder = fully_use_cnn() 

        self.encoder = bert.encoder
        self.pooler = bert.pooler


    def get_extended_attention_mask(self, attention_mask):
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)

        else:
            raise NotImplementedError
        
        try:
            extended_attention_mask = extended_attention_mask.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
        except StopIteration:
            extended_attention_mask = extended_attention_mask.to(dtype=torch.float16)

        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        return extended_attention_mask

    def forward(self, cls_tok, input_txt, attn_mask, segment, input_img, sep_tok):

        extended_attn_mask = self.get_extended_attention_mask(attn_mask)

        if self.new_segment_ids:
            img_seg = (torch.LongTensor(input_txt.size(0), (self.args.num_image_embeds)).fill_(4).cuda()) #[SEP]
            sep_segment = (torch.LongTensor(input_txt.size(0), 1).fill_(4).cuda())
            cls_segment = (torch.LongTensor(input_txt.size(0), 1).fill_(4).cuda())

        else:
            img_seg = (torch.LongTensor(input_txt.size(0), (self.args.num_image_embeds)).fill_(0).cuda()) #[SEP]
            sep_segment = (torch.LongTensor(input_txt.size(0), 1).fill_(0).cuda())
            cls_segment = (torch.LongTensor(input_txt.size(0), 1).fill_(0).cuda())

        img, img_pos = self.img_encoder(input_img)  # BxNx2048
        sep_out = self.txt_embeddings(sep_tok, sep_segment)
        cls_out = self.txt_embeddings(cls_tok, cls_segment)
        img_embed_out = self.img_embeddings(img, img_pos, img_seg)  # img_embed_out: torch.Size([32, 5, 768])
        txt_embed_out = self.txt_embeddings(input_txt, segment)  # txt_embed_out: torch.Size([32, 507, 768])
        encoder_input = torch.cat([cls_out, img_embed_out, sep_out, txt_embed_out], 1)  # TODO: Check B x (TXT + IMG) x HID

        encoded_layers = self.encoder(
            encoder_input, extended_attn_mask, output_hidden_states=False
        )  # in mmbt: output_all_encoded_layers=False, but the argument was changed in recent Transformers
        # encoded_layers[-1] is encoded_layers[0]

        #return self.pooler(encoded_layers[-1])  # torch.Size([32, 768])
        return encoded_layers[-1], self.pooler(encoded_layers[-1])



class CXRBERT(PreTrainedModel):
#class CXRBERT(nn.Module):  # BERTLM, MultimodalBertClf
    """
    Multimodal BERT
    Masked Language Model + Image Text Matching
    """
    def __init__(self, config, args):
        super().__init__(config)
        self.mlm_task = args.mlm_task
        self.itm_task = args.itm_task
        
        self.enc = CXRBertEncoder(args)        
        if self.mlm_task:
            self.mlm = MaskedLanguageModel(args, args.hidden_size, args.vocab_size)

        if self.itm_task:
            self.itm = ImageTextMatching(args.hidden_size)
    
    def forward(self, cls_tok, input_txt, attn_mask, segment, input_img, sep_tok):
        # x = self.enc(cls_tok, input_txt, attn_mask, segment, input_img, sep_tok)  # torch.Size([32, 512, 768]) [bsz, seq_len, hidden]
        x_mlm, x_itm = self.enc(cls_tok, input_txt, attn_mask, segment, input_img, sep_tok)  # bsz, max_len, hidden
        # return self.mlm(x_mlm), self.itm(x_itm)

        if self.mlm_task:
            output_1 = self.mlm(x_mlm)
        
        if self.itm_task:
            output_2 = self.itm(x_itm)

        if self.itm_task and self.mlm_task:
            return output_1, output_2
        elif self.itm_task and self.mlm_task == False:
            return output_2, output_2
        elif self.mlm_task and self.itm_task == False:
            return output_1, output_1

class MaskedLanguageModel(nn.Module):
    """
    (vocab_size) classification model
    """
    def __init__(self, args, hidden, vocab_size):
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.linear.weight = CXRBertEncoder(args).txt_embeddings.word_embeddings.weight

    def forward(self, x):
        return self.linear(x)

class ImageTextMatching(nn.Module):
    """
    2-class classification model : Aligned, Not aligned
    """
    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        
    def forward(self, x):
        return self.linear(x)