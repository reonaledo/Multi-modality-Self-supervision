### Multi-modality Self-supervision
1st KAIST project.
Let's get started!

## Dataset pre-processing

# 1> TieNet: Text-Image Embedding Network for Common Thorax Disease Classification and Reporting in Chest X-rays

Text : Findings & Impression
- critical diagnosis information is often presented in the ‘impression’ section by considering all findings, patient history, and previous studies. ChestX-ray14 consists of 14 disease labels that can be observed in chest X-ray, i.e., Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural Thickening, and Hernia. The NLP-mined labels are used as ‘ground truth’ for model training and testing throughout the experiments. We adopt the patient level data splits published with the data.

# 2> Baselines for Chest X-Ray Report Generation

All experiments in this work used the MIMIC-CXR dataset. MIMIC-CXR consists of 473,057 chest x-ray images and 206,563 reports from 63,478 patients. Of these images, 240,780 are of anteroposterior (AP) views, which we focus on in this work. Further, we eliminate all duplicated radiograph images with adjusted brightness or contrast2, leaving a total of 95,242/87,353 images/reports, which we subdivide into a train set of 75,147/69,171 and a test set of 19,825/18,182 images/reports, with no overlap of patients between the two. Radiological reports are parsed into sections and we use the findings section. See Figure 1 for an example chest x-ray and report.

# 3> Clinically Accurate Chest X-Ray Report Generation

MIMIC-CXR is the largest radiology dataset to date and consists of 473, 057 chest X-ray images and 206,563 reports from 63,478 patients. Among these images, 240,780 are of anteroposterior (AP), 101, 379 are of posteroanterior (PA), and 116, 023 are of lateral (LL) views. Furthermore, we eliminate duplicated radiograph images with adjusted brightness level or contrast as they are commonly produced for clinical needs, after which we are left with 327,281 images and 141,783 reports. The radiological reports are parsed into sections, among which we extract the findings section. We then apply tokenization and keep tokens with at least 5 occurrences in the corpus, resulting in 5, 571 tokens in total.
