from transformers import convert_bert_original_tf_checkpoint_to_pytorch

convert_bert_original_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch(
    '/home/edlab-hglee/NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12/bert_model.ckpt',
    '/home/edlab-hglee/NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12/bert_config.json',
    '/home/edlab-hglee/NCBI_BERT_pubmed_mimic_uncased_L-12_H-768_A-12/pytorch_model.bin')