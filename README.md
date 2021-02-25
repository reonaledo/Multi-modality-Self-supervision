# Multi-modal Understanding and Generation for Medical Images and Text via Vision-Language Pre-Training (KDD2021 Under Review)
This repository provides the code for fine-tuning MedViLL(Medical Vision Language Learner), a medical vision language pre-training model with a novel self-attention mask.  During the pre-training, our model conducts masked language modeling and label-conditioned image text matching. Then the pre-trained model is fine-tuned for disease classification, medical visual question answering, label-conditioned image-text retrieval, and radiology report generation. Please refer to our paper **Multi-modal Understanding and Generation for Medical Images and Text via Vision-Language Pre-Training** for more details.


# Our Contributions.
The main contributions of this paper can be summarized as follows:

(1) We propose Medical Vision Language Learner (MedViLL), a multi-modal pre-training model for medical images and text with a novel self-attention scheme.

(2) We demonstrate the effectiveness of our approach with detailed ablation study on extensive vision language understanding and generation-based downstream tasks, including disease classification, image-text retrieval, visual question answering, and report generation.

(3) We demonstrate the generalization ability of our approach under the transfer learning setting using two separate chest X-ray datasets, where we pre-train a model on one dataset and perform diverse downstream tasks on another.


# Our Approach.
Our proposed architecture MedViLL is a single BERT-based model that learns unified contextualized vision-language (VL) representation for both Vision Language Understanding(VLU) and Vision Language Generation (VLG). MedViLL performs pre-training with a CNN-based visual encoder and a cross-modal Transformer for VL joint representation learning. After pre-training, our model can be easily used for VLU and VLG tasks with task-specific finetuning. MedViLL can be divided into four main components (visual feature embedding, language feature embedding, joint embedding, and pre-training objectives).


## Download.
### Pre-training model.
We are releasing five versions of BERT-based pre-trained weights with different types of self-attention masks. Pre-training for the joint embedding was built on the BERT-base architecutre(12 hidden layers, 12 attention heads, 768 hidden size), and training details are described in our paper. Currently avaliable versions of pre-trained weights are as follows:

<Pre-trained model will be updated>
  
- MedViLL - BERT-Base model with Bidirectional Auto-regressive attention mask.

- Bi & Seq2Seq - BERT-Base model with Seq2Seq attention mask(75% chance) and Bidirectional attention mask(25% chance) in every mini-batch.

- Bidirectional - BERT-Base model with Bidirectional attention mask.

- Seq2Seq - BERT-Base model with Seq2Seq attention mask.

- Non-cross - BERT-Base model with Non-cross modality attention mask.

### Datasets.
We provide a pre-processed version of multiple datasets for each task as follows:
- MIMIC-CXR: (xx MB), Unique study of the AP view imaging and radiology report pair.
- OPEN-I: (xx MB), Unique study of the frontal view(AP and PA view) imaging and radiology report pair.
- VQA-RAD: (xx MB), 


## Reproducing results on BERT.
### Section A. Installation
Sections below describe the virtual env installation and the fine-training process of MedviLL based on pytorch version 1.7, python version 3.8. 

To fine-tune MedViLL, you need to download the pre-trained weights of MedViLL. After downloading the pre-trained weights, use environment.yml to install conda based virtual env as follows:

```
$ git clone https://github.com/SuperSupermoon/Multi-modality-Self-supervision.git
$ cd Multi-modality-Self-supervision; conda env create --file environment.yml
```

Note that this repository is based on the BERT repository by Google. All fine-tuning models were conducted on 8 Geforce RTX-3090 GPU machines, each of which has 24GB of VRAM. 


### Section B. Pre-training model
Example:
```
askdfksdflksd
```



### Section C. Downstream model
Disease Classification
Example:
```
askdfksdflksd
```

Image-Text Retrieval
Example:
```
askdfksdflksd
```

Medical Visual Qestion Answering
Example:
```
python sc/finetune.py --tasks vqa --s2s_prob 0 --bi_prob 1 --mask_prob 0
```

Report Generation
Example:
```
python sc/finetune.py --tasks report_generation --mask_prob 0.15 --s2s_prob 1 --bi_prob 0
```
