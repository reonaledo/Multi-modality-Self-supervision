from transformers import BertConfig, BertTokenizer, BertModel, BertForTokenClassification
import torch

tokenizer = BertTokenizer.from_pretrained('/home/edlab-hglee/bluebert_pretrained/')
model = BertModel.from_pretrained('/home/edlab-hglee/bluebert_pretrained/')

# tokenizer = BertTokenizer.from_pretrained('bluebert')
# model = BertModel.from_pretrained('bluebert')

input = 'This is sample sentence'
input_ids = torch.tensor(tokenizer.encode(input)).unsqueeze(0)

outputs = model(input_ids)


config = BertConfig.from_pretrained('/home/edlab-hglee/Multimodal/bluebert_model/')
model_state_dict = torch.load('/home/edlab-hglee/Multimodal/bluebert_model/pytorch_model.bin')
#model = BertForTokenClassification.from_pretrained('bert-base-uncased', return_dict=True)
model = BertForTokenClassification.from_pretrained('/home/edlab-hglee/Multimodal/bluebert_model/', state_dict=model_state_dict, config=config)