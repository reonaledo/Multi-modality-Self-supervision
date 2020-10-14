# -*- coding: utf-8 -*- 

import pandas as pd
from glob import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageEnhance
import math
import random
import tqdm as tqdm
import collections
from tqdm import tqdm
from datetime import date
from torchvision.utils import save_image
import torchvision.transforms as transforms
import torch
import os

pd.set_option('display.max_colwidth', -1)
jpg = glob("/home/ubuntu/mimic_jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/files/*/*/*/*.jpg")
free_text = glob("/home/ubuntu/mimic_jpg/files/*/*/*.txt")

jpg = pd.DataFrame({"dir" : jpg})
free_text = pd.DataFrame({"dir" : free_text})

print("# of jpg files", len(jpg))
print("# of free_text files",len(free_text))

meta_info = pd.read_csv("/home/ubuntu/mimic_jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/modified_mimic-cxr-2.0.0-metadata.csv")
unique_study_list = meta_info['study_id'].value_counts()[lambda x: x== 1].index.tolist()
unique_std = meta_info[meta_info.study_id.isin(unique_study_list)]
apview_std = unique_std[unique_std.ViewCodeSequence_CodeMeaning.isin(['antero_posterior'])]
# apview_std

re_dataframe= apview_std.drop(['PerformedProcedureStepDescription','study_id2','Height','Width','ViewPosition','StudyDate','StudyTime','ProcedureCodeSequence_CodeMeaning','PatientOrientationCodeSequence_CodeMeaning'],axis=1)
re_dataframe.to_csv("/home/ubuntu/apview_dataframe.csv", mode='w')

# a,b = np.unique(apview_info.study_id, return_counts=True)
train_set = re_dataframe['split'] == 'train'
valid_set = re_dataframe['split'] == 'validate'
test_set = re_dataframe['split'] == 'test'
train = re_dataframe[train_set]
valid = re_dataframe[valid_set]
test = re_dataframe[test_set]
total = len(train)+len(valid)+len(test)
print("total %d, train ratio %.2f, valid ratio %.2f, test ratio %.2f" % (total, len(train)/total*100,len(valid)/total*100,len(test)/total*100))



train_dataframe = re_dataframe[re_dataframe.split.isin(['train'])]
valid_dataframe = re_dataframe[re_dataframe.split.isin(['validate'])]
test_dataframe = re_dataframe[re_dataframe.split.isin(['test'])]

#path matching
jpg["study"] = jpg["dir"].str.split(pat = "/", expand = True)[11]
jpg["num_of_study"] = pd.to_numeric(jpg.study.str.slice(start=1))

free_text["study"] = free_text["dir"].str.split(pat = "/", expand = True)[7]
a = free_text["study"].str.split(pat = ".", expand = True)
free_text["num_of_study"] = pd.to_numeric(a[0].str.slice(start=1))

train_match_image_path = jpg[jpg.num_of_study.isin(train_dataframe['study_id'])]
train_match_txt_path = free_text[free_text.num_of_study.isin(train_dataframe['study_id'])]

valid_match_image_path = jpg[jpg.num_of_study.isin(valid_dataframe['study_id'])]
valid_match_txt_path = free_text[free_text.num_of_study.isin(valid_dataframe['study_id'])]

test_match_image_path = jpg[jpg.num_of_study.isin(test_dataframe['study_id'])]
test_match_txt_path = free_text[free_text.num_of_study.isin(test_dataframe['study_id'])]

total_match_image_path = jpg[jpg.num_of_study.isin(re_dataframe['study_id'])]
total_match_txt_path = free_text[free_text.num_of_study.isin(re_dataframe['study_id'])]


train_img_list = train_match_image_path.dir.tolist() 
train_txt_list = train_match_txt_path.dir.tolist() 

valid_img_list = valid_match_image_path.dir.tolist() 
valid_txt_list = valid_match_txt_path.dir.tolist() 

test_img_list = test_match_image_path.dir.tolist() 
test_txt_list = test_match_txt_path.dir.tolist() 

total_img_list = total_match_image_path.dir.tolist() 
total_txt_list = total_match_txt_path.dir.tolist() 

print("train loaded image list in ap view : ",len(train_img_list))
print("train loaded text list in ap view : ",len(train_txt_list))
print("\n")
print("valid loaded image list in ap view : ",len(valid_img_list))
print("valid loaded text list in ap view : ",len(valid_txt_list))
print("\n")
print("test loaded image list in ap view : ",len(test_img_list))
print("test loaded text list in ap view : ",len(test_txt_list))
print("\n")
print("total loaded image list in ap view : ",len(total_img_list))
print("total loaded text list in ap view : ",len(total_txt_list))
print("\n")

def cut_edge(image, keep_margin):
    '''
    function that cuts zero edge
    '''
    H, W = image.shape
    
    H_s, H_e = 0, H - 1
    W_s, W_e = 0, W - 1

    while H_s < H:
        if image[H_s, :].sum() != 0:
            break
        H_s += 1
    while H_e > H_s:
        if image[H_e, :].sum() != 0:
            break
        H_e -= 1
    while W_s < W:
        if image[:, W_s].sum() != 0:
            break
        W_s += 1
    while W_e > W_s:
        if image[:, W_e].sum() != 0:
            break
        W_e -= 1
    if keep_margin != 0:
        H_s = max(0, H_s - keep_margin)
        H_e = min(H - 1, H_e + keep_margin)
        W_s = max(0, W_s - keep_margin)
        W_e = min(W - 1, W_e + keep_margin)
        
    return int(H_s), int(H_e) + 1, int(W_s), int(W_e) + 1

## 모든 이미지에 블랙 자르기

def image_cut(img_list):
    for itr in img_list:
        img1 = cv2.imread(itr)
        plt.imshow(img1)
        plt.show()
        img2 = cv2.imread(itr, cv2.IMREAD_GRAYSCALE)
        print("Original",img2.shape)
        min_H_s, max_H_e, min_W_s, max_W_e = cut_edge(img2, 0)
        image = img2[min_H_s: max_H_e, min_W_s:max_W_e]
        plt.imshow(image,cmap="gray")
        # plt.show()
        # input("ssssssssss")

def rescale(img_list):
    for itr, impath in enumerate (img_list):
        size = 256, 256
        im = Image.open(impath)
        print(im.size)
        im.thumbnail(size, Image.ANTIALIAS)
        im.save('/home/ubuntu/image_preprocessing/'+str(impath.split('/')[-2])+ str(im.size)+".jpg")
        print(im.size)

def image_cut_and_rescale(img_list):

    size = 256, 256
#     array_list = []
    include_img = []
    include_ap_ratio = []
    exclude_img = []
    exclude_ap_ratio = []
    
    for itr, impath in enumerate (tqdm(img_list)):        
        img2 = Image.open(impath)
        im2arr2 = np.array(img2)
        study_id = str(impath.split('/')[-2])
        # print("Original image size",(im2arr2.shape))
#         plt.imshow(im2arr2,cmap="gray")
#         plt.show()
        
        min_H_s, max_H_e, min_W_s, max_W_e = cut_edge(im2arr2, 0)
        image = im2arr2[min_H_s: max_H_e, min_W_s:max_W_e]

        # print("After cutting out balck :",image.shape)
#         plt.imshow(image,cmap="gray")
#         plt.show()
        
#         image = image.transpose(1, 2, 0)
        arr2im = Image.fromarray(image)    
        
        arr2im.thumbnail(size, Image.ANTIALIAS)
#         arr2im.save('/home/ubuntu/image_preprocessing/'+str(impath.split('/')[-2])+ str(arr2im.size)+".jpg")

        # print("Maintain the aspect ratio", np.array(arr2im).shape)

#         plt.imshow(arr2im,cmap="gray")
#         plt.show()
        
        img_H = (np.array(arr2im).shape[0])
        img_W = (np.array(arr2im).shape[1])

        if 0.8 <= round(img_W/img_H,2) <=1.2:
            include_ap_ratio.append(round(img_W/img_H,2))
            include_img.append([arr2im,study_id])

        else :
            exclude_ap_ratio.append(round(img_W/img_H,2))
            exclude_img.append([arr2im,study_id])
    print("include_ap_ratio",len(include_ap_ratio))
    print("exclude_ap_ratio",len(exclude_ap_ratio))
    
    return include_ap_ratio, include_img, exclude_ap_ratio, exclude_img



def aspect_ratio_histogram(include_ap_ratio, exclude_ap_ratio, bins=None):
    ##### Aspect Ratio = W/H
    #     img_H = [each_shape[0] for each_shape in img_size]
    #     img_W = [each_shape[1] for each_shape in img_size]
    #     img_AR = [round(img_W[i]/img_H[i],2) for i in range(len(img_H)) if 0.8 <= round(img_W[i]/img_H[i],2) <=1.2]
    img_AR = sorted(include_ap_ratio, reverse=False)
    #     abandon_img_AR = [round(img_W[i]/img_H[i],2) for i in range(len(img_H)) if round(img_W[i]/img_H[i],2) < 0.8 or round(img_W[i]/img_H[i],2)>1.2]
    abandon_img_AR = sorted(exclude_ap_ratio, reverse=False)
    
    #     [i for i in v if i==12]
    
    # img_AR.sort()
    # print(img_AR)
    
    print('len(included img AR) : ', len(img_AR))
    print('len(abandon img AR) : ', len(abandon_img_AR))

    #     print('img_AR : ', img_AR[:10])
    #print('img_AR sort', img_AR.sort())

    counter = collections.Counter(img_AR)

    keys_of_counter = counter.keys()
    values_of_counter = counter.values()

    labels, values = zip(*collections.Counter(img_AR).items())
    indexes = np.arange(len(labels))

    labels = np.array(labels)[:][:]
    
    ################################################################################################################

    plt.figure(figsize=(25,10))
    bar_list = plt.bar(indexes, values, width=1, color='#374274',edgecolor='black')

    for idx, value in enumerate(values):
        plt.text(idx, value, str(value), color='r', fontsize=15,
                 horizontalalignment='center', verticalalignment='bottom')

    plt.xticks(indexes, labels, rotation=90, fontsize=15)
    plt.yticks(fontsize=15)
    plt.title('Image Aspect Ratio Distribution of Dataset', fontsize=20)
    plt.figtext(.15,.75,'< Numerical Values >\n    - total # of dataset : %d \n\
        - length of indexes : %d \n' %(len(img_AR),len(indexes)), fontsize=13)
    plt.savefig('/home/ubuntu/Aspect_ratio_dataset_total.png', dpi=300)
    # plt.show()

    ################################################################################################################
    
    if len(abandon_img_AR) > 1:
        ab_counter = collections.Counter(abandon_img_AR)

        keys_of_counter = ab_counter.keys()
        values_of_counter = ab_counter.values()

        ab_labels, ab_values = zip(*collections.Counter(abandon_img_AR).items())
        ab_indexes = np.arange(len(ab_labels))

        ab_labels = np.array(ab_labels)[:][:]
        
        plt.figure(figsize=(25,10))
        bar_list = plt.bar(ab_indexes, ab_values, width=1, color='#374274',edgecolor='black')

        for idx2, value2 in enumerate(ab_values):
            plt.text(idx2, value2, str(value2), color='r', fontsize=15,
                    horizontalalignment='center', verticalalignment='bottom')

        plt.xticks(ab_indexes, ab_labels, rotation=90, fontsize=15)
        plt.yticks(fontsize=15)
        plt.title('** Abandoned Image Aspect Ratio Distribution of Dataset', fontsize=20)
        plt.figtext(.15,.75,'< Numerical Values >\n    - total # of dataset : %d \n\
            - length of indexes : %d \n' %(len(abandon_img_AR),len(ab_indexes)), fontsize=13)
        plt.savefig('/home/ubuntu/Aspect_ratio_dataset_abandon.png', dpi=300)
        # plt.show()
    else: pass



def image_resize_with_interpolation(img_list, include_ap_ratio, mode = None):

    print("length of interpolated images :", len(img_list))
    modified_img_save_path = '/home/ubuntu/image_preprocessing/'

    modified_img_save_path = modified_img_save_path + mode
    if not os.path.exists(modified_img_save_path):
        os.makedirs(modified_img_save_path, mode=0o777)

    size = 256, 256
    interpolated_image_list = []
#     array_list = []
    as_ratio = []
    
    count = 1
    for itr, img in enumerate (tqdm(img_list)):        
        im2arr2 = np.array(img[0])
        study_id = str(img[1])
        interpolated_image = np.array(cv2.resize(im2arr2, size, interpolation=cv2.INTER_NEAREST))
        interpolated_image_list.append(interpolated_image)
        as_ratio.append(include_ap_ratio[itr])
        arr2im = Image.fromarray(interpolated_image)   
        

        if any(str(study_id) in s for s in glob(modified_img_save_path+"/*.jpg")):
            study_id = study_id + '_'+str(count)
            count += 1
        else :
            count = 0

        arr2im.save(modified_img_save_path + '/'+study_id+".jpg", dpi=(300, 300))


    return interpolated_image_list, as_ratio



def random_crop(preprocessed_img, crop_h, crop_w):
#     img = np.array(preprocessed_img)
    
    cropped_image = []
    crop_img_size = []
    print("crop standard", crop_h, crop_w)

    for idx, img in enumerate(tqdm(preprocessed_img)):
        img = np.array(img)
        a,b = img.shape
        print("shape of img", a,b)        
        w1, h1 = random.choice(value_of)
        print("w, h", w1, h1)

        random_crop = img[h1:h1 + crop_h, w1:w1 + crop_w]
        
        crop_img_size.append(random_crop.shape)
        cropped_image.append(random_crop)

        plt.imshow(random_crop,cmap="gray")
        plt.show()    
    print("len(crop_img_size)",len(crop_img_size))
    return cropped_image, crop_img_size

# Brightness only

def aumentation(cropped_image):
        ##### Augmentation instance lists for Randomizing! #####
#     brightness_list = [0.4]
    
    augmented_img = []
    ##### Augmentation! #####
    for idx, img in enumerate(tqdm(cropped_image)):
        
        # color_jitter = transforms.ColorJitter(0.8 * 1, 0.8 * 1, 0.8 * 1, 0.2 * 1)
        # data = transforms.RandomApply([color_jitter], p=0.8)
        # save_image(data, '/home/ubuntu/image_preprocessing/'+str(idx)+'.png')
        # input("sssksksksksks")        
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl1 = clahe.apply(np.array(img))
        
        augmented_img.append(cl1)
        
        arr2im = Image.fromarray(cl1)    
        arr2im.save('/home/ubuntu/image_preprocessing/augmented_image/'+str(idx+1)+".jpg", dpi=(300, 300))

    return augmented_img



# image_cut(img_list)
# rescale(img_list)

# train_img_list 
# valid_img_list 
# test_img_list 

# print("total_img_list",total_img_list)
# prepro_img, prepro_img_size, min_value = image_cut_and_rescale(total_img_list[:10], size = (512,512))

value_of = [(0, 0), (0, 32), (32, 0), (16, 16), (32, 32)]


include_ap_ratio, include_img, exclude_ap_ratio, exclude_img = image_cut_and_rescale(train_img_list)
aspect_ratio_histogram(include_ap_ratio, exclude_ap_ratio)
rescaled_img_list, as_ratio = image_resize_with_interpolation(include_img, include_ap_ratio, mode ='Train')


include_ap_ratio, include_img, exclude_ap_ratio, exclude_img = image_cut_and_rescale(valid_img_list)
aspect_ratio_histogram(include_ap_ratio, exclude_ap_ratio)
rescaled_img_list, as_ratio = image_resize_with_interpolation(include_img, include_ap_ratio, mode ='Valid')


include_ap_ratio, include_img, exclude_ap_ratio, exclude_img = image_cut_and_rescale(test_img_list)
aspect_ratio_histogram(include_ap_ratio, exclude_ap_ratio)
rescaled_img_list, as_ratio = image_resize_with_interpolation(include_img, include_ap_ratio, mode ='Test')

# cropped_image, crop_img_size = random_crop(rescaled_img_list, 224, 224)

# augmented_img_list = aumentation(cropped_image)

# print(min_value)


