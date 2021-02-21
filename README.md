# Multi-modal Understanding and Generation for Medical Images and Text via Vision-Language Pre-Traini
We introduce a vision-and-language pre-training model with a novel self-attention mask in the biomedical domain. During the pre-training, our model conducts masked language modeling and label-conditioned imagetext matching. Then the pre-trained model is fine-tuned for disease classification, medical visual question answering, label-conditioned image-text retrieval, and radiology report generation.


#### • Original Data Structure.
> *MIMIC-CXR <jpg version>
        ⎿ Dataset (xxx)                
            ├ subject_id 001         
            │   ⎿ study_id                   
            │       ├ 00000001_000.jpg           
            │       ⎿ 00001335_006.jpg (1~3개)           
            │
            │   ⎿ study_id 
            │       ⎿ 00001335_006.jpg  
            │
            ├ subject_id 002       
            │   ⎿ study_id                   
            │       ├ 00001336_000.jpg                 
            │       ⎿ 00003923_013.jpg                      
            │                                       
            ├ mimic_cxr_chexpert.csv (Label info)
            ├ mimic_cxr_nebio.csv (Label info)
            ├ mimic_cxr_metadata.csv (View info, subject id, anonymous dicom id, etc ..)
            ⎿ mimic_cxr_split.csv (Train, Valid, Test set)


> * MIMIC-CXR <dicom version>
        ⎿ Dataset (xxx)                
            ├ subject_id 001         
            │   ⎿ study_id 01                 
            │       ├ 00000001_000.dcm           
            │       ⎿ 00001335_006.dcm (1~3개)           
            │   ⎿ study_id 01.txt
            │   
            │   ⎿ study_id 02
            │       ⎿ 00001335_006.dcm  
            │   ⎿ study_id 02.txt
            │   
            │
            ├ subject_id 002       
            │   ⎿ study_id 01                
            │       ├ 00001336_000.dcm                 
            │       ⎿ 00003923_013.dcm                      
            │   ⎿ study_id 01.txt
            │   
            ├ cxr-record-list.csv (subject id, study id, dicom id, path)
            ├ cxr-study-list.csv (subject id, study id, text path)
            ⎿ cxr-reports folder (only contain txt files)


# Our Contributions.
The main contributions of this paper can be summarized as follows:
(1) We propose Medical Vision Language Learner (MedViLL), a multi-modal pre-training model for medical images and text with a novel self-attention scheme.

(2) We demonstrate the effectiveness of our approach with detailed ablation study on extensive vision language understanding and generation-based downstream tasks, including disease classification, image-text retrieval, visual question answering, and report generation.

(3) We demonstrate the generalization ability of our approach under the transfer learning setting using two separate chest X-ray datasets, where we pre-train a model on one dataset and perform diverse downstream tasks on another.


# Our Approach.
Our proposed architecture MedViLL is a single BERT-based model that learns unified contextualized vision-language (VL) representation for both Vision Language Understanding(VLU) and Vision Language Generation (VLG). MedViLL performs pre-training with a CNN-based visual encoder and a cross-modal Transformer for VL joint representation learning. After pre-training, our model can be easily used for VLU and VLG tasks with task-specific finetuning. MedViLL can be divided into four main components (visual feature embedding, language feature embedding, joint embedding, and pre-training objectives).


## Code
Detailed script you can check those files(main.py, data.transformer, data.).

How to run the model?
1. Conda activate simclr 
2. python main.py
