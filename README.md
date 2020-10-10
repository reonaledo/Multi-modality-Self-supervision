
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



# < Oct.7.2020 Today's meeting >
오늘의 미팅 내용 정리.

About : Report parsing
Clinical BERT paper 좀 더 살펴 볼 것 -> . Wordpieace 는 적절해 보이지 않음.
토큰 length 256면 적당해 보임.
Impression + finding > finding > impression
Finding을 제일 우선시해서, 둘다 있을 경우는 긴 것으로 하는게 좀더 description이 기니까 좋지 않나

네트워크
1. ROI 없이 이미지를 학습하는 모델 기반 (PixelBERT, 16x16 patch)으로 베이스라인으로 잡고 구현, Masked Object Modeling을 개선 할 수 있는 방법을 고민해보자. 
2. Method. 
1) MLM 1번 구조, ITM task는 fix. (가장 기본적인것만 confirmed)
2) Input에 attention을 걸어 인풋 임베딩 하는 것은 좋지 않은 방법이다. -> Text-conditioned로 하는 것과 image-conditioned로 하는 방식은 뭔가 방향성이 있어 보이고, 좋은 방법같진 않다.
3) Masked Object 모델링을 위해 Attention based로 인풋을 넣는 방법 말고, 2,3번 대신 LXMERT 방식처럼 각각 인풋 이미지에 노이즈를 섞는 방식을 고민해보자. 
4) Out-of-domain 테스트를 해야하는 명분을 모르겠다 In domain 테스트로만 수행하면 안되는 것인가? out-of-domain을 하는 이유를 명확히 찾자.

object를 prediction 또는 generation을 하기 위해 16x16 grid patch를 randomly maksing한 뒤, regression 하거나 pixel value를 채워 원본 이미지와 비교하는 방식이 좋을 것 같고,
cnn을 통한 feature extraion된 결과에 대해 수행하게 되면 입력까지 backward gradient를 하기 때문에, (자기 자신을 regression함으로서 collapse 될 위험이 있다? -> 명확하게 하기 위해 BYOL 을 살펴볼것). 따라서,
image feature extraction을 위한 CNN layer는 결과적으로 freeze 한 뒤, feature extraction된 fiber로 부터 원본 이미지 pixel value를 채워가는 방식으로 가능은 하겠으나 intuitive 하지 않은 것 같다.
cnn으로부터 visual feature를 extraction하게 되면, 


* 최종, <A COMPARISON OF PRE-TRAINED VISION-AND-LANGUAGE MODELS FOR MULTIMODAL REPRESENTATION LEARNING ACROSS MEDICAL IMAGES AND REPORTS> 
논문에서 각 모델별로 차이를 분석, 
e.g., PIXELBERT가 왜 잘 안됐는지? 우수한 모델이 어떤 것 이었는지? 각 방법론의 장점들은 뭔지? 여러 모델의 장단점을 비교해서 고안하자. 방향은 각 모달리티 인풋을 각각 넣는 식이 좋을 것 같다.

내가 내린 결론 
->좋은 방법은 입력을 16x16 patch로 cnn layer없이 인풋을 grid로 짤라 쓰고, 이를 object masking된 영역을 regression을 하든, pixel value를 채우든 하는 방식으로 가는 것이 좋을 것 같다.

## Code

Now Byol is being trained on Kubernetes. The entire data of the MIMIC-CXR is still being downloaded, so we can run the model using subset data.
Detailed script you can check those files(main.py, data.transformer, data.).

how to run the model?
1. Conda activate simclr 
2. python main.py
