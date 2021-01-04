import torch
import torch.nn as nn

from models.image import ImageEncoder_cnn, ImageEncoder_pool, Img_patch_embedding

from transformers.modeling_auto import AutoConfig, AutoModel
from transformers.modeling_bert import BertConfig, BertModel, BertForMaskedLM, BertPreTrainedModel
from transformers.modeling_albert import AlbertModel
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_lxmert import LxmertForPreTraining
from transformers.configuration_utils import PretrainedConfig

class CXRConfig(PretrainedConfig):
    model_type = "cxrbert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=512,
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=2048,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        gradient_checkpointing=False,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.gradient_checkpointing = gradient_checkpointing

class CXRPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """

    config_class = CXRConfig
    # load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "cxrbert"
    authorized_missing_keys = [r"position_ids"]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

class ImageBertEmbeddings(nn.Module):
    def __init__(self, args, embeddings):
        super().__init__()
        self.args = args
        self.img_embeddings = nn.Linear(args.img_hidden_sz, args.embedding_size)
        self.token_type_embeddings = embeddings.token_type_embeddings
        self.LayerNorm = embeddings.LayerNorm
        self.dropout = nn.Dropout(args.dropout_prob)
        self.position_embeddings = embeddings.position_embeddings

    def forward(self, input_imgs, img_pos, token_type_ids):  # img_embed_out = self.img_embeddings(img, img_tok)

        imgs_embeddings = self.img_embeddings(input_imgs)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        if self.args.img_postion:
            position_embeddings = self.position_embeddings(img_pos)
            embeddings = imgs_embeddings + position_embeddings +token_type_embeddings
        else:
            embeddings = imgs_embeddings + token_type_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)  # bsz, num_img_embeds, hidden_sz
        return embeddings

# class CXRBertEncoder(nn.Module):  # MultimodalBertEncoder, BERT
class CXRBertEncoder(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.args = args

        if args.bert_model == "albert-base-v2":
            bert = AlbertModel.from_pretrained(args.bert_model)
        elif args.bert_model == "emilyalsentzer/Bio_ClinicalBERT":
            bert = AutoModel.from_pretrained(args.bert_model)
        elif args.bert_model == "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12":
            bert = AutoModel.from_pretrained(args.bert_model)
        elif args.bert_model == "bert-small-scratch":
            config = BertConfig.from_pretrained("google/bert_uncased_L-4_H-512_A-8")
            bert = BertModel(config)
        else:
            bert = BertModel.from_pretrained(args.bert_model)  # bert-base-uncased, small, tiny

        self.txt_embeddings = bert.embeddings
        self.img_embeddings = ImageBertEmbeddings(args, self.txt_embeddings)

        if args.img_encoder == 'ViT':
            img_size = args.img_size
            patch_sz = 32 if img_size == 512 else 16
            self.img_encoder = Img_patch_embedding(image_size=img_size, patch_size=patch_sz, dim=2048)
        else:
            self.img_encoder = ImageEncoder_cnn(args)

        self.encoder = bert.encoder
        self.pooler = bert.pooler

        # print('img_enc_weight:', self.img_encoder.state_dict())

    def get_extended_attn_mask(self, attn_mask):
        if attn_mask.dim() == 2:
            # print('attn_mask.dim():', attn_mask.dim())
            extended_attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
        elif attn_mask.dim() == 3:
            # print('attn_mask.dim():', attn_mask.dim())
            extended_attn_mask = attn_mask.unsqueeze(1)
        else:
            raise NotImplementedError
        extended_attn_mask = extended_attn_mask.to(dtype=torch.float16)
        extended_attn_mask = (1.0 - extended_attn_mask) * - 10000.0

        return extended_attn_mask

    def forward(self, cls_tok, input_txt, attn_mask, segment, input_img, sep_tok):

        # extended_attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # (bsz, 1, 1, len(attn_mask))
        # # extended_attn_mask = extended_attn_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        # extended_attn_mask = extended_attn_mask.to(dtype=torch.float16)  # fp16 compatibility
        # extended_attn_mask = (1.0 - extended_attn_mask) * -10000.0

        extended_attn_mask = self.get_extended_attn_mask(attn_mask)

        # if self.args.cuda_devices == [1]:
        #     cuda = torch.device('cuda:1')
        #     img_tok = (torch.LongTensor(input_txt.size(0), self.args.num_image_embeds).fill_(0)).to(device=cuda)
        #     cls_segment = (torch.LongTensor(input_txt.size(0), 1).fill_(0)).to(device=cuda)
        #
        # else:
        #     img_tok = (torch.LongTensor(input_txt.size(0), self.args.num_image_embeds).fill_(0).cuda())
        #     cls_segment = (torch.LongTensor(input_txt.size(0), 1).fill_(0).cuda())

        img_tok = (torch.LongTensor(input_txt.size(0), self.args.num_image_embeds).fill_(0).cuda())

        cls_segment = (torch.LongTensor(input_txt.size(0), 1).fill_(0).cuda())

        cls_out = self.txt_embeddings(cls_tok, cls_segment)
        sep_out = self.txt_embeddings(sep_tok, cls_segment)

        img, position = self.img_encoder(input_img)  # BxNx2048

        img_embed_out = self.img_embeddings(img, position, img_tok)  # bsz, num_img_embeds, hsz
        txt_embed_out = self.txt_embeddings(input_txt, segment)  # bsz, seq_len, hsz. inputs: bsz, seq_len
        encoder_input = torch.cat([cls_out, img_embed_out, sep_out, txt_embed_out], 1)  # B x (TXT + IMG) x HID
        encoded_layers = self.encoder(
            encoder_input, extended_attn_mask, output_hidden_states=False
        )  # torch.Size([16, 512, 768]), torch.Size([16, 1, 1, 512])
        # return encoded_layers[-1]  # bsz, max_len, hsz, encoded_layers[0]
        return encoded_layers[-1], self.pooler(encoded_layers[-1])

class CXRBERT(BertPreTrainedModel):  # BERTLM, MultimodalBertClf
    """
    Multimodal BERT
    : Masked Language Model + Image Text Matching
    """
    def __init__(self, config, args):
        super().__init__(config)
        self.enc = CXRBertEncoder(config, args)
        self.mlm = MaskedLanguageModel(config, args, args.hidden_size, args.vocab_size)
        self.itm = ImageTextMatching(args.hidden_size)

        # self.bertformlm = BertForMaskedLM.from_pretrained(args.bert_model).cls

    def forward(self, cls_tok, input_txt, attn_mask, segment, input_img, sep_tok):
        # x = self.enc(cls_tok, input_txt, attn_mask, segment, input_img)  # bsz, max_len, hidden
        x_mlm, x_itm = self.enc(cls_tok, input_txt, attn_mask, segment, input_img, sep_tok)  # bsz, max_len, hidden
        return self.mlm(x_mlm), self.itm(x_itm)
        # return self.bertformlm(x), self.itm(x)

class MaskedLanguageModel(nn.Module):
    """
    (vocab_size) classification model
    """
    def __init__(self, config, args, hidden, vocab_size):
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.linear.weight = CXRBertEncoder(config, args).txt_embeddings.word_embeddings.weight

    def forward(self, x):
        return self.linear(x)  # x: torch.Size([8, 512, 768]), return: torch.Size([8, 512, 30522])

class ImageTextMatching(nn.Module):
    """
    2-class classification model : Aligned, Not aligned
    """
    def __init__(self, hidden):
        super().__init__()
        self.linear = nn.Linear(hidden, 2)

    def forward(self, x):
        return self.linear(x)
        # return self.linear(x[:, 0])  # [CLS], x_size: [bsz, max_len, hsz] -> [bsz, hsz], return [bsz, 2]