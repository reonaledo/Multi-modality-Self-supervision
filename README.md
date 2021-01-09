
# Multi-modality Self-supervision
1st KAIST project.
Let's get started!


#### • Target Class.   
* Atelectasis  
* Cardiomegaly  
* Consolidation
* Edema
* Enlarged Cardiomediastinum
* Fracture
* Lung Lesion
* Lung Opacity
* No finding
* Pleural Effusion
* Effusion          
* Pneumonia    
* Pneumothorax 
* Support Devices

#### • Directory Structure.

> *  KAIST Multi-modality self-supervision <jpg version>
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


> *  KAIST Multi-modality self-supervision <dicom version>
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
 
## Related work (Dataset pre-processing)

#### 1> TieNet: Text-Image Embedding Network for Common Thorax Disease Classification and Reporting in Chest X-rays

Text : Findings & Impression
- critical diagnosis information is often presented in the ‘impression’ section by considering all findings, patient history, and previous studies. ChestX-ray14 consists of 14 disease labels that can be observed in chest X-ray, i.e., Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, and Hernia. The NLP-mined labels are used as ‘ground truth’ for model training and testing throughout the experiments. We adopt the patient level data splits published with the data.

#### 2> Baselines for Chest X-Ray Report Generation

All experiments in this work used the MIMIC-CXR dataset. MIMIC-CXR consists of 473,057 chest x-ray images and 206,563 reports from 63,478 patients. Of these images, 240,780 are of anteroposterior (AP) views, which we focus on in this work. Further, we eliminate all duplicated radiograph images with adjusted brightness or contrast2, leaving a total of 95,242/87,353 images/reports, which we subdivide into a train set of 75,147/69,171 and a test set of 19,825/18,182 images/reports, with no overlap of patients between the two. Radiological reports are parsed into sections and we use the findings section. See Figure 1 for an example chest x-ray and report.

#### 3> Clinically Accurate Chest X-Ray Report Generation

MIMIC-CXR is the largest radiology dataset to date and consists of 473, 057 chest X-ray images and 206,563 reports from 63,478 patients. Among these images, 240,780 are of anteroposterior (AP), 101, 379 are of posteroanterior (PA), and 116, 023 are of lateral (LL) views. Furthermore, we eliminate duplicated radiograph images with adjusted brightness level or contrast as they are commonly produced for clinical needs, after which we are left with 327,281 images and 141,783 reports. The radiological reports are parsed into sections, among which we extract the findings section. We then apply tokenization and keep tokens with at least 5 occurrences in the corpus, resulting in 5, 571 tokens in total.



# Our Approach.

our baseline model architecture is Unified VLP for Natural Language Genration and Natural Language Understanding tasks.
BERT based image-text model can help joint embedding of img-txt information and can generate captions autoregressively through a modified converter encoder structure.
Our strategy for single modality is that train the model utilizing BYOL method to get image feature (randomly pick 100 img feature fibers), and also train the model utilizing BlueBERT method to make the txt only model.
After train each model, we make the Unified VLP based model to jointly train image and text features for the multi-modality model.

1. Pre-train tasks
- CLOZE tasks (Masked Seq-to-Seq masking n-gram using multi mask)
- Image-txt matching using <CLS> token.

2. Downstream task
- Classification: image, txt, img-txt, img-generated txt (optional)
- NLU : Caption based Image Retrieval (Recall@1, Recall@5, Recall@10), Text Retrieval optional.
- NLG : Image captioning. -> To evaluate the clinical accuracy of the generated report, we apply the labeler method (CheXpert).

## Code

Now Byol is being trained on Kubernetes. The entire data of the MIMIC-CXR is still being downloaded, so we can run the model using subset data.
Detailed script you can check those files(main.py, data.transformer, data.).

how to run the model?
1. Conda activate simclr 
2. python main.py
