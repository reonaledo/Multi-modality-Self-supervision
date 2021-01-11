import torch
import torch.nn as nn

from models.image import random_sample, Img_patch_embedding, fully_use_cnn

from transformers import BertConfig, AlbertConfig, AutoTokenizer, AutoModel, BertModel, AutoConfig, BertPreTrainedModel
from transformers.modeling_utils import PreTrainedModel, ModuleUtilsMixin
from transformers.configuration_utils import PretrainedConfig
import math

class ImageBertEmbeddings(nn.Module):
    def __init__(self, args, embeddings):  # self.img_embeddings = ImageBertEmbeddings(args, self.txt_embeddings)
        super().__init__()
        self.args = args
        self.img_embeddings = nn.Linear(args.img_hidden_sz, args.hidden_size)
        
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

class CXRBertEncoder_origin(BertPreTrainedModel):  # MultimodalBertEncoder, BERT
    def __init__(self, config, args):
        super().__init__(config)
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
        
        # causal mask generation.
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

class CXRBERT(BertPreTrainedModel):
#class CXRBERT(nn.Module):  # BERTLM, MultimodalBertClf
    """
    Multimodal BERT
    Masked Language Model + Image Text Matching
    """
    def __init__(self, config, args):
        super().__init__(config)        
        self.mlm_task = args.mlm_task
        self.itm_task = args.itm_task
        self.enc = CXRBertEncoder_origin(config, args)        
        self.cls = BertPreTrainingHeads(config, CXRBertEncoder_origin(config, args).txt_embeddings.word_embeddings.weight, num_labels=2) # num_labels not applicable for VLP
    
    def forward(self, cls_tok, input_txt, attn_mask, segment, input_img, sep_tok):
        # x = self.enc(cls_tok, input_txt, attn_mask, segment, input_img, sep_tok)  # torch.Size([32, 512, 768]) [bsz, seq_len, hidden]
        x_mlm, x_itm = self.enc(cls_tok, input_txt, attn_mask, segment, input_img, sep_tok)  # bsz, max_len, hidden
        # return self.mlm(x_mlm), self.itm(x_itm)

        if self.mlm_task:
            output_1 = self.cls(x_mlm)
        
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
    def __init__(self, config, args, hidden, vocab_size):
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.linear.weight = CXRBertEncoder_origin(config, args).txt_embeddings.word_embeddings.weight

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

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}
try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        hid_size = config.hidden_size
        if hasattr(config, 'relax_projection') and (config.relax_projection > 1):
            hid_size *= config.relax_projection
        self.dense = nn.Linear(config.hidden_size, hid_size)
        self.LayerNorm = BertLayerNorm(hid_size, eps=1e-5)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(
            bert_model_embedding_weights.size(0)))
        if hasattr(config, 'relax_projection') and (config.relax_projection > 1):
            self.relax_projection = config.relax_projection
        else:
            self.relax_projection = 0
        self.converted = False

    def forward(self, hidden_states, task_idx=None):
        if not self.converted:
            self.converted = True
            # if self.fp32_embedding:
            #     self.transform.half()
        hidden_states = self.transform(hidden_states)
        if self.relax_projection > 1:
            num_batch = hidden_states.size(0)
            num_pos = hidden_states.size(1)
            # (batch, num_pos, relax_projection*hid) -> (batch, num_pos, relax_projection, hid) -> (batch, num_pos, hid)
            hidden_states = hidden_states.view(
                num_batch, num_pos, self.relax_projection, -1)[torch.arange(0, num_batch).long(), :, task_idx, :]
            hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights, num_labels=2):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(
            config, bert_model_embedding_weights)
        # self.seq_relationship = nn.Linear(config.hidden_size, num_labels)

    def forward(self, sequence_output, task_idx=2):
        prediction_scores = self.predictions(sequence_output, task_idx)
        seq_relationship_score = None
        # if pooled_output is None:
        #     seq_relationship_score = None
        # else:
        #     seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score
