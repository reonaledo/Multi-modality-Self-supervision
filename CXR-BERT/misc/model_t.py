from data.data import *
from transformers import BertModel
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_eval_batch_size=16,
    per_device_train_batch_size=32,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./log',
    logging_steps=0,
)

model = BertModel.from_pretrained('/home/edlab-hglee/bluebert/bluebert_model/')
trainier = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
)

trainier.train()