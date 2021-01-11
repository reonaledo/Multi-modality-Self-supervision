import csv
from tqdm import tqdm
import pickle
import pandas as pd
import json
import os
import re
import shutil
import sys
import tqdm
from os.path import join
from sklearn.model_selection import train_test_split
from tqdm.autonotebook import tqdm


def format_cxr_dataset(txt_path, img_path, label_path, json_save_path):
    print("Parsing data...")
    data = parse_data(txt_path, img_path, label_path)
    print("data")
    print("Saving everything into format...")
    save_in_format(data, json_save_path)


# def format_txt_file(content):
#     for c in '<>/\\+=-_[]{}\'\";:.,()*&^%$#@!~`':
#         content = content.replace(c, ' ')
#     content = re.sub("\s\s+" , ' ', content)
#     return content.lower().replace("\n", " ")


def parse_data(txt_path, img_path, label_path):
    splits = ["Train", "Valid","Test"]
    data = {split: [] for split in splits}
    label = pd.read_csv(label_path)
   
    count_no_matched_img = 0
    count_matched_img = 0
    for split in (splits):
        print("split",split)
        for sub_id in tqdm(os.listdir(join(txt_path, split))):
            dobj = {}
            std_id = sub_id.split('.')[0]
        
#             print("img", str('/home/ubuntu/byol_/CXR_BYOL/vit_github/examples/3ch/'+str(split)+'/'+str(sub_id.split('.')[0])+'.jpg'))            
            img_ = join(img_path, split, "{}.jpg".format(sub_id.split('.')[0]))

            match_label = label.loc[label['study_id'].isin([std_id])]
            matched_label = match_label['label'].tolist()
                       
            if len(matched_label) == 0:
                print("there is no matched label!!")
                input("STOP!! ERROR!!")
            
            if not os.path.exists(img_):
                    count_no_matched_img += 1
                    print("id", std_id)
                    print("There is no matched image here! :", count_no_matched_img)

            else:
                count_matched_img += 1
                print("\n\n")
                print("id", std_id)
                print("split", split)
                print("label", matched_label[0][1:-1])
                print("text", open(txt_path+'/'+str(split)+'/'+str(sub_id)).read())
                dobj["id"] = std_id
                dobj["split"] = split
                dobj["label"] = matched_label[0][1:-1]
                dobj["text"] = open(txt_path+'/'+str(split)+'/'+str(sub_id)).read()
                dobj["img"] = str(img_)
                data[split].append(dobj)
                print("There are matched image here! :", count_matched_img)

            
    return data


def save_in_format(data, target_path):
    """
    Stores the data to @target_dir. It does not store metadata.
    """

    for split_name in data:
        jsonl_loc = join(target_path, split_name + ".jsonl")
        with open(jsonl_loc, "w") as jsonl:
            for sample in tqdm(data[split_name]):
                jsonl.write("%s\n" % json.dumps(sample))


if __name__ == "__main__":
    chexpert = pd.read_csv('/home/ubuntu/mimic_jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv')
    stack_study_name = []
    stack_label = []
    count = 1
    for row in chexpert.loc():
        positive = row.isin(['1.0'])
        negative = row.isin(['-1.0'])
        ambiguous = row.isin(['0.0'])

        positive = row.index[positive].tolist()
        negative = row.index[negative].tolist()
        ambiguous = row.index[ambiguous].tolist()

        study_name = 's'+str(int(row[1]))
        stack_study_name.append(study_name)

        if len(positive)>0:
            label = positive
            stack_label.append(label)
        else:
            label = []
            stack_label.append(label)

        count += 1
    #     input("Stop")
        if count == 227828:
            break

    new = pd.DataFrame(data={'study_id':stack_study_name, 'label':stack_label}, columns=['study_id','label'])
    new.to_csv("/home/ubuntu/image_preprocessing/positive_label_extract.csv", mode='w')
    
  #-------------------------------------------------------------------------------------------------------------------------#
  
    txt_path = "/home/ubuntu/Multi-modality-Self-supervision/F_TXT/F_TXT_split"
    img_path = '/home/ubuntu/image_preprocessing/re_512_3ch'
    label_path = "/home/ubuntu/image_preprocessing/positive_label_extract.csv"
#     save_path = "/home/ubuntu/simclr_/Multi-modality-Self-supervision"
    save_path = "/home/ubuntu/image_preprocessing"

    
    # Path to the directory for Food101
    format_cxr_dataset(txt_path, img_path, label_path, save_path)