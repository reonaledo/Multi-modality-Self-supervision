"""
====================================================================
	File name: mimic3_cxr_data_matching.py
====================================================================
---------------------------*[Description]*--------------------------
	-Comments: This is the mimic 3 and mimic cxr data matching process
               to connect both dataset for futher research.
               Made by Jong Hak Moon(jhak.moon@gmail.com), Department of AI, KAIST.

    # Reference
	- [MIMIC-CXR Database]
	    (https://physionet.org/content/mimic-cxr/2.0.0/)

	- [MIMIC-III Clinical Database]
	    (https://physionet.org/content/mimiciii/1.4/)

====================================================================
-------------------------*[Version record]*-------------------------
--------------------------------------------------------------------
	  Date	      ｜	     Name	    ｜ 	 Version description       ｜
--------------------------------------------------------------------
    2020.10.06       Jong Hak Moon       v0.1 - start

====================================================================
TODO | MJH :

"""

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
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import datasets
from glob import glob
from PIL import Image
import os
%matplotlib inline

pd.set_option('display.max_colwidth', -1)

free_text = glob("/home/ubuntu/mimic_jpg/files/*/*/*.txt")
free_text = pd.DataFrame({"dir" : free_text})
print("# of free_text files",len(free_text))


# meta_info = pd.read_csv("/home/ubuntu/mimic_jpg/physionet.org/files/mimic-cxr-jpg/2.0.0/modified_mimic-cxr-2.0.0-metadata.csv")
meta_info = pd.read_csv("/home/ubuntu/NOTEEVENTS.csv")
meta_info.head()

meta_info['CATEGORY'].unique()
radio_only = meta_info['CATEGORY'] == 'Radiology'
print("total data column number :",len(meta_info))
print("total Radiology number :",len(meta_info[radio_only]))
print("\n")
radio_report = meta_info[radio_only]
radio_report.to_csv('radio_report.csv')

portable_ap = meta_info['DESCRIPTION'] == 'CHEST (PORTABLE AP)'
pa_lat = meta_info['DESCRIPTION'] == 'CHEST (PA & LAT)'
len_ap = len(meta_info[portable_ap])
len_pa_lat = len(meta_info[pa_lat])
print("Length of MIMIC3 AP view",len_ap)
print("Length of MIMIC3 PA & LAT view",len_pa_lat)
print("Total data of MIMIC3 X-ray",len_ap+len_pa_lat)

portable_ap = meta_info[portable_ap]
pa_lat = meta_info[pa_lat]

ap_report_list = []
for itr in portable_ap['TEXT']:
    tmp_list = []
    for line in itr.splitlines():
        tmp_list.append(line)
        
    ap_report_list.append(tmp_list)

report_list = []
for itr in pa_lat['TEXT']:
    tmp_list = []
    for line in itr.splitlines():
        tmp_list.append(line)
        
    report_list.append(tmp_list)
    
import csv


# MIMIC 3 Parsing and save the result as csv file.

Report_analysis = open('MIMIC3_xray_report_AP.csv','w')
# Report_analysis = open('MIMIC3_xray_report_PA_LAT.csv','w')
wr = csv.writer(Report_analysis)

count = 0
# for data in pa_lat['TEXT']:
for data in portable_ap['TEXT']:
    #ap
    for idx, line in enumerate(ap_report_list[count]):
    #pa LAT
#     for idx, line in enumerate(report_list[count]):
        if 'FINDINGS' not in data and 'IMPRESSION' not in data:
            wr.writerow([count,'None'])
            print(count, 'None')
            break
        elif 'FINDINGS' in line:
            #ap
            report = ap_report_list[count][idx:]
#             #LAT_PA
#             report = report_list[count][idx:]
            wr.writerow([count,report])
            print('FINDINGS',count, report)
            break
        elif 'IMPRESSION' in line:
            #ap
            report = ap_report_list[count][idx:]
#             #LAT_PA
#             report = report_list[count][idx:]
            wr.writerow([count,report])
            print('IMPRESSION',count, report)
            break
    count += 1
    print("finish count : ",count)

Report_analysis.close()

# MIMIC-CXR Parsing and save the result as csv file.
# Report -> Findings & Impressions

Path = glob("/home/ubuntu/mimic_jpg/files/*/*/*.txt")
def Report_to_lines(Report):
    Report_analysis = open('MIMIC_cxr_report.csv','w')
    wr = csv.writer(Report_analysis)
    for i in range(len(Report)):

        f, f_re = open(Report[i],'r'), open(Report[i],'r')
        lines, data = f.readlines(), f_re.read() # 1) list of lines in txt
        for idx, line in enumerate(lines):
            if 'FINDINGS' not in data and 'IMPRESSION' not in data:
                wr.writerow([Report[i][-13:],'None'])
                print(Report[i][-13:], 'None')
                break

            elif 'FINDINGS' in line:
                report = lines[idx:]
                wr.writerow([Report[i][-13:],report])
                print(Report[i][-13:], report)
                break

            elif 'IMPRESSION' in line:
                report = lines[idx:]
                wr.writerow([Report[i][-13:],report])
                print(Report[i][-13:], report)
                break
        f.close()
        f_re.close()
    Report_analysis.close()

Report_to_lines(Path)

#AP view
mic3_parsing = pd.read_csv("/home/ubuntu/simclr_/MIMIC3_xray_report_AP.csv")

#PA_LAT view
# mic3_parsing = pd.read_csv("/home/ubuntu/simclr_/MIMIC3_xray_report_PA_LAT.csv")

#CXR
cxr_parsing = pd.read_csv("/home/ubuntu/simclr_/MIMIC_cxr_report.csv")


# Comparing the parsing results 
from string_grouper import match_strings, match_most_similar, group_similar_strings, StringGrouper

# Create all matches:
matches = match_strings(mic3_parsing.iloc[:, 1],cxr_parsing.iloc[:, 1], ngram_size = 1)

# #AP_view
matches.to_csv('/home/ubuntu/simclr_/AP_view_text_matching_result.csv')

# #PA_LAT
# matches.to_csv('/home/ubuntu/simclr_/LAT_PA_view_text_matching_result.csv')

#AP view
match_read = pd.read_csv("/home/ubuntu/simclr_/AP_view_text_matching_result.csv")

#PA_LAT view
# match_read = pd.read_csv("/home/ubuntu/simclr_/LAT_PA_view_text_matching_result.csv")

b2w = match_read[(match_read.similarity > 0.99) & (match_read.similarity < 1)].sort_values(['similarity'], ascending = [True])
print("0.99< sim <1 : ",len(b2w))
# print("\n")
under1 = match_read[match_read.similarity > 0.999].sort_values(['similarity'], ascending = [True])
print("sim = 1 :",len(under1))
# print("\n")

# print("\n")
total_matched = match_read.sort_values(['similarity'], ascending = [True])
print("co sim over 0.8:",len(total_matched))
print("--------------------------------------")
print("\n")


for itr in range(len(b2w)):
    print("MIMIC3 : \n",b2w.iloc[itr+1,1])
    print("\n")
    print("MIMIC-CXR : \n",b2w.iloc[itr+1,2])
    print("\n")
    print("similarity",b2w.iloc[itr+1,3])
    print("-------------------------------------------------------------------------")
    
# for itr in range(len(a)):
#     print("MIMIC3 : \n",a.iloc[itr+1,1])
#     print("\n")
#     print("MIMIC-CXR : \n",a.iloc[itr+1,2])
#     print("\n")
#     print("similarity",a.iloc[itr+1,3])
#     print("-------------------------------------------------------------------------")
#     input()

# only save high similiar result of AP
# match_read.to_csv('/home/ubuntu/simclr_/over0.99_AP_view_text_matching_result.csv')

# only save high similiar result of PA_LAT
# match_read.to_csv('/home/ubuntu/simclr_/over0.99_LAT_PA_view_text_matching_result.csv')
