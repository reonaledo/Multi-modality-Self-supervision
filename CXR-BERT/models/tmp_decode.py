from transformers import BertModel, BertConfig, AlbertConfig, AutoConfig, AutoModel


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, prev_embedding=None, prev_encoded_layers=None, output_all_encoded_layers=True):
        assert (prev_embedding is None) == (prev_encoded_layers is None), \
                "history embedding and encoded layer must be simultanously given."
        all_encoder_layers = []
        if (prev_embedding is not None) and (prev_encoded_layers is not None):
            history_states = prev_embedding
            for i, layer_module in enumerate(self.layer):
                hidden_states = layer_module(
                    hidden_states, attention_mask, history_states=history_states)
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
                if prev_encoded_layers is not None:
                    history_states = prev_encoded_layers[i]
        else:
            for layer_module in self.layer:
                hidden_states = layer_module(hidden_states, attention_mask)
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class CXRBertEncoder(nn.Module):  # MultimodalBertEncoder, BERT
    def __init__(self, args):
        super().__init__()
        self.args = args
        # self.config = config
        if args.init_model == 'bert-base-uncased':
            bert = BertModel.from_pretrained('bert-base-uncased')
        elif args.init_model == 'ClinicalBERT':
            bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        elif args.init_model == 'BlueBERT':
            bert = BertModel.from_pretrained('bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12')
        elif args.init_model == 'google/bert_uncased_L-4_H-512_A-8':
            bert = AutoModel.from_pretrained("google/bert_uncased_L-4_H-512_A-8")
        else:
            bert = BertModel.from_pretrained(args.init_model)
        
        if args.from_scratch:
            config = BertConfig.from_pretrained(args.init_model)
            model = BertModel.from_config(config)
            print("the model will be trained from scratch!!")

        else: pass

        self.bert = bert
        self.txt_embeddings = self.bert.embeddings
        self.img_embeddings = ImageBertEmbeddings(args, self.txt_embeddings)
        
        if args.img_encoding == 'random_sample':
            self.img_encoder = random_sample(args)

        elif args.img_encoding == 'Img_patch_embedding':
            self.img_encoder = Img_patch_embedding(image_size=512, patch_size=32, dim=2048)  # ViT
            
        elif args.img_encoding == 'fully_use_cnn':    
            self.img_encoder = fully_use_cnn() 

        self.encoder = bert.encoder
        self.pooler = self.bert.pooler

    def get_extended_attention_mask(self, attention_mask):
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        extended_attention_mask = extended_attention_mask.to(dtype=torch.float16)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask


    def forward(self, cls_tok, input_txt, attn_mask, segment, input_img):
        
        extended_attn_mask = self.get_extended_attention_mask(attn_mask)
        img_tok = (torch.LongTensor(input_txt.size(0), (self.args.num_image_embeds)).fill_(0).cuda())
        cls_segment = (torch.LongTensor(input_txt.size(0), 1).fill_(0).cuda())
        img = self.img_encoder(input_img)  # BxNx2048
        cls_out = self.txt_embeddings(cls_tok, cls_segment)
        img_embed_out = self.img_embeddings(img, img_tok)  # img_embed_out: torch.Size([32, 5, 768])
        txt_embed_out = self.txt_embeddings(input_txt, segment)  # txt_embed_out: torch.Size([32, 507, 768])
        encoder_input = torch.cat([cls_out, img_embed_out, txt_embed_out], 1)  # TODO: Check B x (TXT + IMG) x HID
        encoded_layers = self.encoder(
            encoder_input, extended_attn_mask, output_hidden_states=False
        )  # in mmbt: output_all_encoded_layers=False, but the argument was changed in recent Transformers
        # encoded_layers[-1] is encoded_layers[0]

        #return self.pooler(encoded_layers[-1])  # torch.Size([32, 768])
        return encoded_layers[-1]  # torch.Size([32, 512, 768])

class CXRBertDecoder(nn.Module):  # MultimodalBertEncoder, BERT
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.from_scratch:
            config = BertConfig.from_pretrained(args.init_model)
            bert = AutoModel.from_config(config)
            print("the model will be trained from scratch!!")

        else:
            if args.init_model == 'bert-base-uncased':
                bert = BertModel.from_pretrained('bert-base-uncased')
            elif args.init_model == 'ClinicalBERT':
                bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            elif args.init_model == 'BlueBERT':
                bert = BertModel.from_pretrained('bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12')
            elif args.init_model == 'google/bert_uncased_L-4_H-512_A-8':
                bert = AutoModel.from_pretrained("google/bert_uncased_L-4_H-512_A-8")
            else:
                bert = BertModel.from_pretrained(args.init_model)
                
        self.bert = bert
        self.txt_embeddings = self.bert.embeddings
        self.img_embeddings = ImageBertEmbeddings(args, self.txt_embeddings)
        
        if args.img_encoding == 'random_sample':
            self.img_encoder = random_sample(args)

        elif args.img_encoding == 'Img_patch_embedding':
            self.img_encoder = Img_patch_embedding(image_size=512, patch_size=32, dim=2048)  # ViT
            
        elif args.img_encoding == 'fully_use_cnn':    
            self.img_encoder = fully_use_cnn() 

        self.encoder = bert.encoder
        self.pooler = self.bert.pooler

    def get_extended_attention_mask(self, attention_mask):
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        extended_attention_mask = extended_attention_mask.to(dtype=torch.float16)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask


    def forward(self, input_txt, attn_mask, segment, input_img):
        
        extended_attn_mask = self.get_extended_attention_mask(attn_mask)
        img_tok = (torch.LongTensor(input_txt.size(0), (self.args.num_image_embeds)).fill_(0).cuda())
        img = self.img_encoder(input_img)  # BxNx2048
        
        img_embed_out = self.img_embeddings(img, img_tok)  # img_embed_out: torch.Size([32, 5, 768])
        txt_embed_out = self.txt_embeddings(input_txt, segment)  # txt_embed_out: torch.Size([32, 507, 768])
        embedding_output = torch.cat([img_embed_out, txt_embed_out], 1)  # TODO: Check B x (TXT + IMG) x HID
        encoded_layers = self.encoder(embedding_output, extended_attn_mask, prev_embedding=prev_embedding,
            prev_encoded_layers=prev_encoded_layers,
            output_hidden_states=True)  # in mmbt: output_all_encoded_layers=False, but the argument was changed in recent Transformers
        # encoded_layers[-1] is encoded_layers[0]

        #return self.pooler(encoded_layers[-1])  # torch.Size([32, 768])
        return embedding_output, encoded_layers[-1]  # torch.Size([32, 512, 768])


class BertModelIncr(BertModel):
    def __init__(self, config):
        super(BertModelIncr, self).__init__(config)
    def get_extended_attention_mask(self, attention_mask):
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else: raise NotImplementedError
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float16)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def forward(self, vis_feats, vis_pe, input_ids, token_type_ids, position_ids, attention_mask,
                prev_embedding=None, prev_encoded_layers=None, output_all_encoded_layers=True,
                len_vis_input=None):
        if args.init_model == 'bert-base-uncased':
            bert = BertModel.from_pretrained('bert-base-uncased',output_hidden_states=True)
        elif args.init_model == 'ClinicalBERT':
            bert = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        elif args.init_model == 'BlueBERT':
            bert = BertModel.from_pretrained('bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12',output_hidden_states=True)
        elif args.init_model == 'google/bert_uncased_L-4_H-512_A-8':
            bert = AutoModel.from_pretrained("google/bert_uncased_L-4_H-512_A-8", output_hidden_states=True)
        else:
            bert = BertModel.from_pretrained(args.init_model,output_hidden_states=True)
        
        if args.from_scratch:
            config = BertConfig.from_pretrained(args.init_model)
            bert = BertModel.from_config(config)
            
        self.bert = bert
        self.txt_embeddings = self.bert.embeddings
        self.img_embeddings = ImageBertEmbeddings(args, self.txt_embeddings)
        
        if args.img_encoding == 'random_sample':
            self.img_encoder = random_sample(args)

        elif args.img_encoding == 'Img_patch_embedding':
            self.img_encoder = Img_patch_embedding(image_size=512, patch_size=32, dim=2048)  # ViT
            
        elif args.img_encoding == 'fully_use_cnn':    
            self.img_encoder = fully_use_cnn() 

        self.encoder = bert.encoder
        self.pooler = self.bert.pooler

        extended_attention_mask = self.get_extended_attention_mask(attn_mask)
        embedding_output = self.embeddings(
            vis_feats, vis_pe, input_ids, token_type_ids, position_ids,
            vis_input=(prev_encoded_layers is None), len_vis_input=len_vis_input)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      prev_embedding=prev_embedding,
                                      prev_encoded_layers=prev_encoded_layers,
                                      output_all_encoded_layers=output_all_encoded_layers)

        input("STOP!!!")
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return embedding_output, encoded_layers, pooled_output


""" for VLP, based on UniLM """
class BertForSeq2SeqDecoder(PreTrainedBertModel):
    """refer to BertForPreTraining"""

    def __init__(self, config, args, mask_word_id=0, num_labels=2,
                 search_beam_size=1, length_penalty=1.0, eos_id=0,
                 forbid_duplicate_ngrams=False, forbid_ignore_set=None,
                 ngram_size=3, min_len=0, len_vis_input=None):
        super(BertForSeq2SeqDecoder, self).__init__(config)
        
        self.bert = CXRBertDecoder(args)
        input("현재 여기는 통과 했냐?!!")

        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight, num_labels=num_labels)

        self.apply(self.init_bert_weights)
        self.crit_mask_lm = nn.CrossEntropyLoss(reduce=False, reduction='none')
        self.mask_word_id = mask_word_id
        self.num_labels = num_labels
        self.len_vis_input = len_vis_input
        self.search_beam_size = search_beam_size
        self.length_penalty = length_penalty
        self.eos_id = eos_id
        self.forbid_duplicate_ngrams = forbid_duplicate_ngrams
        self.forbid_ignore_set = forbid_ignore_set
        self.ngram_size = ngram_size
        self.min_len = min_len

        # will not be initialized when loading BERT weights
        
        self.vis_embed = nn.Sequential(nn.Linear(2048, config.hidden_size*2),
                                    nn.ReLU(),
                                    nn.Linear(config.hidden_size*2, config.hidden_size),
                                    nn.ReLU(),
                                    nn.Dropout(config.hidden_dropout_prob))
        
        self.vis_pe_embed = nn.Sequential(nn.Linear(2048, config.hidden_size),
                                    nn.ReLU(),
                                    nn.Dropout(config.hidden_dropout_prob))

    def forward(self, vis_feats, vis_pe, input_ids, token_type_ids, position_ids, attention_mask, task_idx=None, sample_mode='greedy'):
        vis_feats = self.vis_embed(vis_feats) # image region features
        vis_pe = self.vis_pe_embed(vis_pe) # image region positional encodings

        if self.search_beam_size > 1:
            return self.beam_search(vis_feats, vis_pe, input_ids, token_type_ids, position_ids, attention_mask, task_idx)
        input_shape = list(input_ids.size())
        batch_size = input_shape[0]
        input_length = input_shape[1]
        output_shape = list(token_type_ids.size())
        output_length = output_shape[1]

        output_ids = []
        output_probs = []
        prev_embedding = None
        prev_encoded_layers = None
        curr_ids = input_ids
        mask_ids = input_ids[:, :1] * 0 + self.mask_word_id
        next_pos = input_length

        while next_pos < output_length:
            curr_length = list(curr_ids.size())[1]
            start_pos = next_pos - curr_length
            x_input_ids = torch.cat((curr_ids, mask_ids), dim=1)
            curr_token_type_ids = token_type_ids[:, start_pos:next_pos + 1]
            curr_attention_mask = attention_mask[:,
                start_pos:next_pos + 1, :next_pos + 1]
            curr_position_ids = position_ids[:, start_pos:next_pos + 1]
            new_embedding, new_encoded_layers, _ = \
                self.bert(vis_feats, vis_pe, x_input_ids, curr_token_type_ids, curr_position_ids,
                curr_attention_mask, prev_embedding=prev_embedding,
                prev_encoded_layers=prev_encoded_layers,
                output_all_encoded_layers=True, len_vis_input=self.len_vis_input)

            last_hidden = new_encoded_layers[-1][:, -1:, :]
            prediction_scores, _ = self.cls(
                last_hidden, None, task_idx=task_idx)
            if sample_mode == 'greedy':
                max_probs, max_ids = torch.max(prediction_scores, dim=-1)
            elif sample_mode == 'sample':
                prediction_scores.squeeze_(1)
                prediction_probs = F.softmax(prediction_scores, dim=-1).detach()
                max_ids = torch.multinomial(prediction_probs, num_samples=1,
                    replacement=True)
                max_probs = torch.gather(F.log_softmax(prediction_scores, dim=-1),
                    1, max_ids) # this should be logprobs
            else:
                raise NotImplementedError
            output_ids.append(max_ids)
            output_probs.append(max_probs)
            if prev_embedding is None:
                prev_embedding = new_embedding[:, :-1, :]
            else:
                prev_embedding = torch.cat(
                    (prev_embedding, new_embedding[:, :-1, :]), dim=1)
            if prev_encoded_layers is None:
                prev_encoded_layers = [x[:, :-1, :]
                                       for x in new_encoded_layers]
            else:
                prev_encoded_layers = [torch.cat((x[0], x[1][:, :-1, :]), dim=1)
                                       for x in zip(prev_encoded_layers, new_encoded_layers)]
            curr_ids = max_ids
            next_pos += 1
        return torch.cat(output_ids, dim=1), torch.cat(output_probs, dim=1)


    def beam_search(self, vis_feats, vis_pe, input_ids, token_type_ids, position_ids, attention_mask, task_idx=None):

        input_shape = list(input_ids.size())

        # print("input_shape",input_shape)

        batch_size = input_shape[0]
        input_length = input_shape[1]
        # print("input_length",input_length)

        output_shape = list(token_type_ids.size())
        output_length = output_shape[1]
        # print("output_length",output_length)

        output_ids = []
        prev_embedding = None
        prev_encoded_layers = None
        curr_ids = input_ids
        mask_ids = input_ids[:, :1] * 0 + self.mask_word_id
        next_pos = input_length

        K = self.search_beam_size

        total_scores = []
        beam_masks = []
        step_ids = []
        step_back_ptrs = []
        partial_seqs = []
        forbid_word_mask = None
        buf_matrix = None

        while next_pos < output_length:
            curr_length = list(curr_ids.size())[1]
            start_pos = next_pos - curr_length
            x_input_ids = torch.cat((curr_ids, mask_ids), dim=1)
            curr_token_type_ids = token_type_ids[:, start_pos:next_pos + 1]
            curr_attention_mask = attention_mask[:, start_pos:next_pos + 1, :next_pos + 1]
            curr_position_ids = position_ids[:, start_pos:next_pos + 1]

            new_embedding, new_encoded_layers, _ = \
                self.bert(vis_feats, vis_pe, x_input_ids, curr_token_type_ids, curr_position_ids,
                curr_attention_mask, prev_embedding=prev_embedding,
                prev_encoded_layers=prev_encoded_layers,
                output_all_encoded_layers=True, len_vis_input=self.len_vis_input)

            last_hidden = new_encoded_layers[-1][:, -1:, :]
            prediction_scores, _ = self.cls(
                last_hidden, None, task_idx=task_idx)
            log_scores = torch.nn.functional.log_softmax(
                prediction_scores, dim=-1)
            if forbid_word_mask is not None:
                log_scores += (forbid_word_mask * -10000.0)
            if self.min_len and (next_pos-input_length+1 <= self.min_len):
                log_scores[:, :, self.eos_id].fill_(-10000.0)
            kk_scores, kk_ids = torch.topk(log_scores, k=K)
            if len(total_scores) == 0:
                k_ids = torch.reshape(kk_ids, [batch_size, K])
                back_ptrs = torch.zeros(batch_size, K, dtype=torch.long)
                k_scores = torch.reshape(kk_scores, [batch_size, K])
            else:
                last_eos = torch.reshape(
                    beam_masks[-1], [batch_size * K, 1, 1])
                last_seq_scores = torch.reshape(
                    total_scores[-1], [batch_size * K, 1, 1])
                kk_scores += last_eos * (-10000.0) + last_seq_scores
                kk_scores = torch.reshape(kk_scores, [batch_size, K * K])
                k_scores, k_ids = torch.topk(kk_scores, k=K)
                back_ptrs = torch.div(k_ids, K)
                kk_ids = torch.reshape(kk_ids, [batch_size, K * K])
                k_ids = torch.gather(kk_ids, 1, k_ids)
            step_back_ptrs.append(back_ptrs)
            step_ids.append(k_ids)
            beam_masks.append(torch.eq(k_ids, self.eos_id).float())
            total_scores.append(k_scores)

            def first_expand(x):
                input_shape = list(x.size())
                expanded_shape = input_shape[:1] + [1] + input_shape[1:]
                x = torch.reshape(x, expanded_shape)
                repeat_count = [1, K] + [1] * (len(input_shape) - 1)
                x = x.repeat(*repeat_count)
                x = torch.reshape(x, [input_shape[0] * K] + input_shape[1:])
                return x

            def select_beam_items(x, ids):
                id_shape = list(ids.size())
                id_rank = len(id_shape)
                assert len(id_shape) == 2
                x_shape = list(x.size())
                x = torch.reshape(x, [batch_size, K] + x_shape[1:])
                x_rank = len(x_shape) + 1
                assert x_rank >= 2
                if id_rank < x_rank:
                    ids = torch.reshape(
                        ids, id_shape + [1] * (x_rank - id_rank))
                    ids = ids.expand(id_shape + x_shape[1:])
                y = torch.gather(x, 1, ids)
                y = torch.reshape(y, x_shape)
                return y

            is_first = (prev_embedding is None)

            if prev_embedding is None:
                prev_embedding = first_expand(new_embedding[:, :-1, :])
            else:
                prev_embedding = torch.cat(
                    (prev_embedding, new_embedding[:, :-1, :]), dim=1)
                prev_embedding = select_beam_items(prev_embedding, back_ptrs)
            if prev_encoded_layers is None:
                prev_encoded_layers = [first_expand(
                    x[:, :-1, :]) for x in new_encoded_layers]
            else:
                prev_encoded_layers = [torch.cat((x[0], x[1][:, :-1, :]), dim=1)
                                       for x in zip(prev_encoded_layers, new_encoded_layers)]
                prev_encoded_layers = [select_beam_items(
                    x, back_ptrs) for x in prev_encoded_layers]

            curr_ids = torch.reshape(k_ids, [batch_size * K, 1])

            if is_first:
                token_type_ids = first_expand(token_type_ids)
                position_ids = first_expand(position_ids)
                attention_mask = first_expand(attention_mask)
                mask_ids = first_expand(mask_ids)

            if self.forbid_duplicate_ngrams:
                wids = step_ids[-1].tolist()
                ptrs = step_back_ptrs[-1].tolist()
                if is_first:
                    partial_seqs = []
                    for b in range(batch_size):
                        for k in range(K):
                            partial_seqs.append([wids[b][k]])
                else:
                    new_partial_seqs = []
                    for b in range(batch_size):
                        for k in range(K):
                            new_partial_seqs.append(
                                partial_seqs[ptrs[b][k] + b * K] + [wids[b][k]])
                    partial_seqs = new_partial_seqs

                def get_dup_ngram_candidates(seq, n):
                    cands = set()
                    if len(seq) < n:
                        return []
                    tail = seq[-(n-1):]
                    if self.forbid_ignore_set and any(tk in self.forbid_ignore_set for tk in tail):
                        return []
                    for i in range(len(seq) - (n - 1)):
                        mismatch = False
                        for j in range(n - 1):
                            if tail[j] != seq[i + j]:
                                mismatch = True
                                break
                        if (not mismatch) and not(self.forbid_ignore_set and (seq[i + n - 1] in self.forbid_ignore_set)):
                            cands.add(seq[i + n - 1])
                    return list(sorted(cands))

                if len(partial_seqs[0]) >= self.ngram_size:
                    dup_cands = []
                    for seq in partial_seqs:
                        dup_cands.append(
                            get_dup_ngram_candidates(seq, self.ngram_size))
                    if max(len(x) for x in dup_cands) > 0:
                        if buf_matrix is None:
                            vocab_size = list(log_scores.size())[-1]
                            buf_matrix = np.zeros(
                                (batch_size * K, vocab_size), dtype=float)
                        else:
                            buf_matrix.fill(0)
                        for bk, cands in enumerate(dup_cands):
                            for i, wid in enumerate(cands):
                                buf_matrix[bk, wid] = 1.0
                        forbid_word_mask = torch.tensor(
                            buf_matrix, dtype=log_scores.dtype)
                        forbid_word_mask = torch.reshape(
                            forbid_word_mask, [batch_size * K, 1, vocab_size]).cuda()
                    else:
                        forbid_word_mask = None
            next_pos += 1

        # [(batch, beam)]
        total_scores = [x.tolist() for x in total_scores]
        step_ids = [x.tolist() for x in step_ids]
        step_back_ptrs = [x.tolist() for x in step_back_ptrs]
        # back tracking
        traces = {'pred_seq': [], 'scores': [], 'wids': [], 'ptrs': []}
        for b in range(batch_size):
            # [(beam,)]
            scores = [x[b] for x in total_scores]
            wids_list = [x[b] for x in step_ids]
            ptrs = [x[b] for x in step_back_ptrs]
            traces['scores'].append(scores)
            traces['wids'].append(wids_list)
            traces['ptrs'].append(ptrs)
            # first we need to find the eos frame where all symbols are eos
            # any frames after the eos frame are invalid
            last_frame_id = len(scores) - 1
            for i, wids in enumerate(wids_list):
                if all(wid == self.eos_id for wid in wids):
                    last_frame_id = i
                    break
            max_score = -math.inf
            frame_id = -1
            pos_in_frame = -1

            for fid in range(last_frame_id + 1):
                for i, wid in enumerate(wids_list[fid]):
                    if wid == self.eos_id or fid == last_frame_id:
                        s = scores[fid][i] + self.length_penalty * (fid + 1)
                        if s > max_score:
                            max_score = s
                            frame_id = fid
                            pos_in_frame = i
            if frame_id == -1:
                traces['pred_seq'].append([0])
            else:
                seq = [wids_list[frame_id][pos_in_frame]]
                for fid in range(frame_id, 0, -1):
                    pos_in_frame = ptrs[fid][pos_in_frame]
                    seq.append(wids_list[fid - 1][pos_in_frame])
                seq.reverse()
                traces['pred_seq'].append(seq)

        def _pad_sequence(sequences, max_len, padding_value=0):
            trailing_dims = sequences[0].size()[1:]
            out_dims = (len(sequences), max_len) + trailing_dims

            out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
            for i, tensor in enumerate(sequences):
                length = tensor.size(0)
                # use index notation to prevent duplicate references to the tensor
                out_tensor[i, :length, ...] = tensor
            return out_tensor

        # convert to tensors for DataParallel
        for k in ('pred_seq', 'scores', 'wids', 'ptrs'):
            ts_list = traces[k]
            if not isinstance(ts_list[0], torch.Tensor):
                dt = torch.float if k == 'scores' else torch.long
                ts_list = [torch.tensor(it, dtype=dt) for it in ts_list]
            traces[k] = _pad_sequence(
                ts_list, output_length, 0).to(input_ids.device)

        return traces