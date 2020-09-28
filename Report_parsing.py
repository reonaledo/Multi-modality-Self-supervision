# Report -> Findings & Impressions

import os
import re
import csv
import pickle
import shutil
import pandas as pd
import numpy as np
from openpyxl import Workbook
import matplotlib.pyplot as plt

# with open(Report[0],'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         print(line)

# # return file into sequences(문자열)
# f = open(Report[0],'r')
# data = f.read()
#
# if 'FINDINGS' in data:
#     print('ttt')
# f.close()



# test_path = 'C:\\Users\\HG_LEE\\Desktop\\MMIC-CXR\\files\\p11\\p11000011\\s51029426_test.txt'
# f = open(test_path,'r')

Path = 'C:\\Users\\HG_LEE\\Downloads\\files' # Original reports path
AP_path = 'C:\\Users\\HG_LEE\\PycharmProjects\\MIMIC-CXR\\AP_report\\Original'

def get_report_path(Path):
    path_list = []
    for (path, dir, files) in os.walk(Path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.txt':
                report_path = os.path.join(path,filename)
                path_list.append(report_path)
    return path_list

def Report_to_lines(Report):
    Report_analysis = open('Report_analysis_p11.csv','w')
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

def Report_to_sequence(Report):
    for i in range(len(Report)):
        Report_name = Report[i][-13:]
        f = open(Report[i], 'r')
        sequence = f.read()
        parsing = open('C:\\Users\\HG_LEE\\PycharmProjects\\MIMIC-CXR\\Parsed_report\\' + Report_name, 'w')

        if 'FINDING' not in sequence and 'IMPRESSION' not in sequence:
            parsing.write(sequence)
            print(Report[i])

        elif sequence.find('FINDING') != -1:
            idx = sequence.find('FINDING')
            parsing.write(sequence[idx:])
            print(Report[i])

        elif sequence.find('IMPRESSION') != -1:
            idx = sequence.find('IMPRESSION')
            parsing.write(sequence[idx:])
            print(Report[i])

        f.close()
        parsing.close()

def Words_counting(Report):
    cnt_none, cnt_both, cnt_OF, cnt_OI = 0,0,0,0
    cnt = 0

    for i in range(len(Report)):
        f = open(Report[i], 'r')
        sequence = f.read()

        # if 'FINDING' not in sequence and 'IMPRESSION' not in sequence:
        #     cnt_none +=1
        #
        # elif 'FINDING' in sequence and 'IMPRESSION' in sequence:
        #     cnt_both += 1
        #
        # elif 'FINDING' in sequence and 'IMPRESION' not in sequence:
        #     cnt_OF += 1
        #
        # elif 'FINDING' not in sequence and 'IMPRESSION' in sequence:
        #     cnt_OI += 1
        # if 'FINAL REPORT' in sequence:
        #     cnt += 1
        #     print(cnt)

    # print('Num of \'FINAL REPORT\' : ', cnt, '/', len(Report))

    # print('Num of no \'FINDINGS\' and \'IMPRESSIONS\' : ', cnt_none,'/',len(Report))
    # print('Num of both \'FINDINGS\' and \'IMPRESSIONS\' : ', cnt_both, '/', len(Report))
    # print('Num of only \'FINDINGS\' : ', cnt_OF, '/', len(Report))
    # print('Num of only \'IMPRESSIONS\' : ', cnt_OI, '/', len(Report))

def AP_view_parsing():

    metadata = pd.read_csv('C:\\Users\\HG_LEE\\PycharmProjects\\MIMIC-CXR\\mimic-cxr-2.0.0-metadata.csv')
    m_s_id = metadata.study_id.values.tolist()
    m_view = metadata.ViewCodeSequence_CodeMeaning.values.tolist()
    meta_dict = {}

    # for i in range(len(m_s_id)):
    #     if m_view[i] == 'antero-posterior':
    #         meta_dict[m_s_id[i]] = m_view[i]
    # print(len(meta_dict))

    # for m_s_id,m_view in zip(m_s_id,m_view):
    #     meta_dict[m_s_id] = m_view
    #meta_dict = dict(zip(m_s_id,m_view))

    study_list = pd.read_csv('C:\\Users\\HG_LEE\\PycharmProjects\\MIMIC-CXR\\cxr-study-list.csv')
    s_s_id = study_list.study_id.values.tolist()
    s_path = study_list.path.values.tolist()
    study_dict = dict(zip(s_s_id, s_path)) # 57540554: file/p11/...
    #print(len(study_dict)) # 227835 ==
    t_list = []
    cnt = 0
    for i in range(len(m_s_id)):
        if m_view[i] == 'antero-posterior':
            for j in range(len(s_s_id)):
                if m_s_id[i] == s_s_id[j]:
                    t_list.append([m_view[i], s_s_id[j], s_path[j]])
                    cnt +=1
                    print(m_view[i],cnt)
    print('Finished')

    return t_list


#Report = get_report_path(Path)
#AP_origin_report = get_report_path(AP_path)

#Report_to_lines(Report) # to csv, separate with \n

#Report_to_sequence(Report) # to exel, sequences

#Words_counting(Report)

#AP_view_parsing()

'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
# 일단 AP view 이미지만 사용하기로 가정했기에, AP view가 존재하는 study_id 추출하기,
# AP view 만 존재하는 study_id로 해당 report path 추출해서 pickle 파일로 덤핑함
# 이떄 생긴 이슈, 한 study_id에 여러 AP view가 존재하는 경우가 있다는 것을 알게 됨
#AP_list = []
# metadata = pd.read_csv('C:\\Users\\HG_LEE\\PycharmProjects\\MIMIC-CXR\\mimic-cxr-2.0.0-metadata.csv')
# m_s_id = metadata.study_id.values.tolist()
# m_view = metadata.ViewCodeSequence_CodeMeaning.values.tolist()
#
# study_list = pd.read_csv('C:\\Users\\HG_LEE\\PycharmProjects\\MIMIC-CXR\\cxr-study-list.csv')
# s_s_id = study_list.study_id.values.tolist()
# s_path = study_list.path.values.tolist()
#
# cnt = 0
# for i in range(len(m_s_id)):
#     if m_view[i] == 'antero-posterior':
#         for j in range(len(s_s_id)):
#             if m_s_id[i] == s_s_id[j]:
#                 AP_list.append([m_view[i], s_s_id[j], s_path[j]])
#                 print(cnt,AP_list[cnt])
#                 cnt +=1
#
# print(AP_list)

# with open('AP_list.pickle','wb') as f:
#     pickle.dump(AP_list, f)
'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
# 덤핑한 pickle 파일 읽어오기
# with open('AP_list.pickle','rb') as f:
#     AP_list = pickle.load(f)

# print(AP_list[:2])
# print(AP_list[0][2])
'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
# 추려진 AP_path로 .txt파일 복사, 총 AP view 146,448 에서 중복된거 제외하면 매칭될 수 있는 report 갯수는 131,920
# 어차피 copy할땐 중복된건 알아서 덮어써지므로 상관 없음
# AP 뷰가 여러장 있는 study_id가 14,528개임을 알게 되었고, 최대 10개도 존재함
# for i in AP_list:
#     AP_path = os.path.abspath('C:\\Users\\HG_LEE\\PycharmProjects\\MIMIC-CXR\\Parsed_report\\' + i[2][-13:])
#     print(AP_path)
#
#     dst = 'C:\\Users\\HG_LEE\\PycharmProjects\\MIMIC-CXR\\AP_report\\Parsed'
#     shutil.copy2(AP_path, dst)
'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
# 하나의 study_id 에 AP view 가 다수 존재하는 경우있음, 각 study_id 당 AP view 갯수 카운트 후 value가 1이 아닌 즉 여러장의 AP view가 있는 애들만 추린후 pickle로 덤핑
# metadata = pd.read_csv('C:\\Users\\HG_LEE\\PycharmProjects\\MIMIC-CXR\\mimic-cxr-2.0.0-metadata.csv')
# m_s_id = metadata.study_id.values.tolist()
# m_view = metadata.ViewCodeSequence_CodeMeaning.values.tolist()
# count = {}
# for i in range(len(m_view)):
#     if m_view[i] == 'antero-posterior':
#         j = m_s_id[i]
#         try: count[j] += 1
#         except: count[j] = 1
# #print(len(count))
# a = dict({key:value for key,value in count.items() if value !=1})
# print(a)
#
# with open('Duplicate_cnt.pickle','wb') as f:
#     pickle.dump(a, f)
'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
# duplicated_cnt = {}
# with open('Duplicate_cnt.pickle','rb') as f:
#     a = pickle.load(f)
# a = list(a.values())
# for i in a:
#     try: duplicated_cnt[i] +=1
#     except: duplicated_cnt[i] =1
# print(duplicated_cnt)

def Unique_AP_report(Report):
    #AP view 가 1개만 있는 리포트 가져오기
    study_list = pd.read_csv('C:\\Users\\HG_LEE\\PycharmProjects\\MIMIC-CXR\\mimic\\cxr-study-list.csv')
    s_s_id = study_list.study_id.values.tolist()
    s_path = study_list.path.values.tolist()

    Unique_AP_list = pd.read_csv('C:\\Users\\HG_LEE\\PycharmProjects\\MIMIC-CXR\\data_analysis\\Unique_AP_list.csv')
    U_AP_id = Unique_AP_list.study_id.values.tolist()
    for U_var in U_AP_id:
        U_path = s_path[s_s_id.index(U_var)]
        U_path = os.path.abspath('C:\\Users\\HG_LEE\\Downloads\\' + U_path)

        dst = os.path.abspath('C:\\Users\\HG_LEE\\PycharmProjects\\MIMIC-CXR\\Unique_AP_report\\Original')
        shutil.copy2(U_path, dst)

        print(U_path)

def Unique_AP_sectioned_report(Report):
    #AP view가 1개만 있는 study_id의 sectioned report 가져오기
    Unique_AP_list = pd.read_csv('C:\\Users\\HG_LEE\\PycharmProjects\\MIMIC-CXR\\data_analysis\\Unique_AP_list.csv')
    U_AP_id = Unique_AP_list.study_id.values.tolist()

    sectioned_report = pd.read_csv('C:\\Users\\HG_LEE\\PycharmProjects\\MIMIC-CXR\\mimic-cxr-sections\\mimic_cxr_sectioned.csv').replace(np.nan, 0)
    study_id = sectioned_report.study.values.tolist()
    impression = sectioned_report.impression.values.tolist()
    findings = sectioned_report.findings.values.tolist()
    last_paragraph = sectioned_report.last_paragraph.values.tolist()
    comparison = sectioned_report.comparison.values.tolist()

    error_list = []
    parsed_report_section_list = []
    last_paragraph_list = [] # finding, impression X
    content_list = [] # sectioned_report에 comparison part만 있는 study_id

    parsed_path = 'C:\\Users\\HG_LEE\\PycharmProjects\\MIMIC-CXR\\Unique_AP_report\\Parsed\\'
    for U_var in U_AP_id:

        try:
            U_var = str(U_var)
            idx = study_id.index('s'+ U_var)
            with open(parsed_path + 's' + U_var +'.txt', 'w') as file:
                if impression[idx] == 0.0 and findings[idx] == 0.0: #impression, findings 둘 다 없는 경우
                    if last_paragraph[idx] != 0.0:
                        file.write(last_paragraph[idx])
                        print(U_var, 'last_paragraph')
                        last_paragraph_list.append(U_var)
                        parsed_report_section_list.append([U_var,': last_paragraph'])
                    elif last_paragraph[idx] == 0.0:
                        file.write(comparison[idx])
                        print(U_var, 'comparison')
                        content_list.append(U_var)
                        parsed_report_section_list.append([U_var, ': comparison'])

                elif impression[idx] != 0.0 and findings[idx] == 0.0: # impression 있고, finding 없는 경우
                    if last_paragraph[idx] != 0.0:
                        file.write(impression[idx]+last_paragraph[idx])
                        print(U_var, 'impression, last_paragraph')
                        parsed_report_section_list.append([U_var, ': impression, last_paragraph'])
                    elif last_paragraph[idx] == 0.0:
                        file.write(impression[idx])
                        print(U_var, 'impression')
                        parsed_report_section_list.append([U_var, ': impression'])

                elif impression[idx] == 0.0 and findings[idx] != 0.0: # impression 없고, finding 있는 경우
                    if last_paragraph[idx] != 0.0:
                        file.write(findings[idx] + last_paragraph[idx])
                        print(U_var, 'findings, last_paragraph')
                        parsed_report_section_list.append([U_var, ': findings, last_paragraph'])
                    elif last_paragraph[idx] == 0.0:
                        file.write(findings[idx])
                        print(U_var, 'findings')
                        parsed_report_section_list.append([U_var, ': findings'])

                elif impression[idx] != 0.0 and findings[idx] != 0.0: # 둘 다 있는 경우
                    file.write(impression[idx]+findings[idx])
                    print(U_var, 'impression, findings')
                    parsed_report_section_list.append([U_var, ': impression, findings'])

        except ValueError as e:
            error_list.append(e)

    print('Error_list:', error_list)
    print('!!!!!!!!!!!!!!!!!')
    print('Content_list :', content_list)
    print('!!!!!!!!!!!!!!!!!')
    print('Last_paragraph_list :', last_paragraph_list)
    print('last_paragraph_lenght:', len(last_paragraph_list))

    with open('parsing_error_list.pickle', 'wb') as f:
        pickle.dump(error_list, f)

    with open('parsed_report_section_list.pickle', 'wb') as f:
        pickle.dump(parsed_report_section_list, f)


#Unique_AP_report(Report)
#Unique_AP_sectioned_report(Report)

left_list = ['s50058765','s50196495','s50743547','s50913680','s51491012',
             's51966317','s53202765','s53356173','s53514462','s54654948',
             's54875119','s54986978','s55157853','s55258338','s56412866',
             's56451190','s56482935','s56724958','s57564132','s57936451',
             's58235663','s59067458','s59087630','s59215320','s59330497','s59505494']

def Space_parsed_report(Reports):
    # \t. \n 제거하기
    for report in Reports:
        with open(report, 'r') as f:
            data = f.read()
        with open('C:\\Users\\HG_LEE\\PycharmProjects\\MIMIC-CXR\\Unique_AP_report\\Space_parsed\\' + report[-13:],'w') as space_parsing:
            parsed = ' '.join(data.split())
            space_parsing.write(parsed)
        print(report)

# Unique_parsed_path = 'C:\\Users\\HG_LEE\\PycharmProjects\\MIMIC-CXR\\Unique_AP_report\\Parsed'
# Unique_report = get_report_path(Unique_parsed_path)
# Space_parsed_report(Unique_report)

def Merge_text_files(Reports):
    with open('merged_reports.txt','w') as output:
        for report in Reports:
            with open(report,'r') as input:
                output.write(input.read())
                output.write('\n')

# Space_parsed_path = 'C:\\Users\\HG_LEE\\PycharmProjects\\MIMIC-CXR\\Unique_AP_report\\Space_parsed'
# Space_report = get_report_path(Space_parsed_path)
# Merge_text_files(Space_report)


# chexpert, negbio label data에서 겹치는 case 추려내기
def id_labels(label_path):
    # label.csv에서 리스트로 정보들 가져오기
    rows = []
    file = csv.reader(open(label_path))

    for row in file:
        rows.append(row)
    return rows[1:]

def intersection_labels():

    chexpert = id_labels('C:\\Users\\HG_LEE\\PycharmProjects\\MIMIC-CXR\\mimic\\mimic-cxr-2.0.0-chexpert.csv')
    negbio = id_labels('C:\\Users\\HG_LEE\\PycharmProjects\\MIMIC-CXR\\mimic\\mimic-cxr-2.0.0-negbio.csv')

    for row in negbio:
        for idx,value in enumerate(row):
            if value == '1.0':
                row[idx] = '1'
            if value == '-1.0':
                row[idx] = '-1'
            if value == '0.0':
                row[idx] = '0'

    result = [x for x in chexpert if x in negbio]
    print(result)
    print('num of intersection :', len(result)) # num of intersection : 205,668 / 227,835
    with open('intersection_labels.pickle','wb') as f:
        pickle.dump(result, f)


# all case에서 intersection_label 중에서 Unique_ap_view인 case 추려내기
def inter_U_labels():
    with open('intersection_labels.pickle','rb') as f:
        inter_labels = pickle.load(f)

    Unique_AP_list = pd.read_csv('C:\\Users\\HG_LEE\\PycharmProjects\\MIMIC-CXR\\data_analysis\\Unique_AP_list.csv')
    U_AP_id = Unique_AP_list.study_id.values.tolist()
    U_AP_id = list(map(str,U_AP_id))

    # print(inter_labels[:2])
    # print(U_AP_id[:2])

    inter_U_AP_labels = [row for row in inter_labels if row[1] in U_AP_id]
    with open('inter_U_AP_labels.pickle', 'wb') as f:
        pickle.dump(inter_U_AP_labels, f)

    print(len(inter_U_AP_labels)) # 88628/99849


# pickel에 저장된 list 들 .csv 파일로 저장
# with open('./data_analysis/inter_U_AP_labels.pickle','rb') as f:
#     inter_U_AP_labels = pickle.load(f)

# inter_U_AP_case = pd.DataFrame(inter_U_AP_labels)
# inter_U_AP_case.to_csv('inter_U_AP_labels.csv', index=False, header=False) # to .csv

# inter_U_AP_labels에서 오직 positive(1)만 있는 case 추려내기 34354/88628
# P_inter_u_labels = [row for row in inter_U_AP_labels if '-1' not in row and '0' not in row] # 34354/88628, Postive만 있는 경우
# p_labels = [row for row in inter_U_AP_labels if '1' in row] # 86817/88628, positive(1)이 하나라도 있는 경우


def U_inter_Nofinding_parsing():
    # No_finding label에 해당하는 report의 평균 token length
    # 1. inter_U_AP_labels.csv에서 No Finding==1 인 study_id 찾기, No_finding, index = 10
    inter_U_AP_list = id_labels('./data_analysis/inter_U_AP_labels.csv')
    NO_finding_inter_U_AP = []
    for case in inter_U_AP_list:
        if case[10] == '1':
            NO_finding_inter_U_AP.append(case)

    #print(len(NO_finding_inter_U_AP)) # 17893/88628
    # 2. ./space_parsed/~.txt에서 1.에 해당하는 report parsing
    U_parsed_reports = get_report_path('C:\\Users\\HG_LEE\\PycharmProjects\\MIMIC-CXR\\Unique_AP_report\\Space_parsed')
    dst = 'C:\\Users\\HG_LEE\\PycharmProjects\\MIMIC-CXR\\Unique_AP_report\\Inter_U_AP_Nofinding'
    for N_case in NO_finding_inter_U_AP:
        N_study_id = 's' + N_case[1]
        for path in U_parsed_reports:
            U_study_id = path[-13:-4]
            if N_study_id == U_study_id:
                print(path) # notice 용
                shutil.copy2(path, dst)

#U_inter_Nofinding_parsing()

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

def Inter_U_AP_Nofindig_reports_tokenizing():
    inter_U_AP_Nofinding_reports = get_report_path('C:\\Users\\HG_LEE\\PycharmProjects\\MIMIC-CXR\\Unique_AP_report\\Inter_U_AP_Nofinding')
    total = len(inter_U_AP_Nofinding_reports)
    len_subwords_list = []
    for report in inter_U_AP_Nofinding_reports:
        with open(report,'r') as f:
            sequence = f.read()
            len_subwords = len(tokenizer.encode(sequence)) - 2
            len_subwords_list.append(len_subwords)

    print('sum_subwords:', sum(len_subwords_list)) # 1368476
    print('avg_subwords:', sum(len_subwords_list)/total) # 76.48108198736936
    print('num_total:', len(len_subwords_list)) # 17893
    print('max_:', max(len_subwords_list)) # 432
    print('min_:', min(len_subwords_list)) # 2

    with open('Inter_U_AP_Nofinding_len_subwords_list.pickle','wb') as f:
        pickle.dump(len_subwords_list, f)

#Inter_U_AP_Nofindig_reports_tokenizing()

def Plotting_analysis():
    # parsing된 report subword length 분포 그리기
    with open('./Inter_U_AP_Nofinding_len_subwords_list.pickle','rb') as f:
        I_U_N_len_lst = pickle.load(f)

    I_U_N_len_lst.sort()

    x_axis_num = list(range(0,101, 10))
    x_axis = list(map(str,x_axis_num))
    y_axis = [0]*len(x_axis)

    for i in I_U_N_len_lst[:13446]:
        if i < x_axis_num[1]:
            y_axis[0] += 1
        else:
            for j in x_axis:
                if str(i)[0] == j[0]:
                    y_axis[int(str(i)[0])] += 1

    plt.bar(x_axis_num, y_axis, width=0.7, color='blue')

    plt.xlabel('Token_length')
    plt.ylabel('Counts')

    plt.title('Token length analysis of CXR')
    plt.show()

#Plotting_analysis()

# with open('./Inter_U_AP_Nofinding_len_subwords_list.pickle','rb') as f:
#     I_U_N_len_lst = pickle.load(f)
#
# I_U_N_len_lst.sort()
# print(I_U_N_len_lst)
# print(I_U_N_len_lst.index(101)) #13446
# print(I_U_N_len_lst[13445]) # subword len 의 마지막 100

# 최종적으로 사용 할 Intersection_Unique_AP_view Reports parsing, 88,628
def U_inter_parsing():
    inter_U_AP_list = id_labels('./data_analysis/inter_U_AP_labels.csv')
    U_parsed_reports = get_report_path('C:\\Users\\HG_LEE\\PycharmProjects\\MIMIC-CXR\\Unique_AP_report\\Space_parsed')
    dst = 'C:\\Users\\HG_LEE\\PycharmProjects\\MIMIC-CXR\\Unique_AP_report\\Inter_U_AP'

    for case in inter_U_AP_list:
        I_study_id = 's' + case[1]
        for path in U_parsed_reports:
            U_study_id = path[-13:-4]
            if I_study_id == U_study_id:
                print(path) # notice 용
                shutil.copy2(path, dst)

#U_inter_parsing()

# 각 report 별 subwords 분포
def Inter_U_AP_reports_tokenizing():
    inter_U_AP_reports = get_report_path('C:\\Users\\HG_LEE\\PycharmProjects\\MIMIC-CXR\\Unique_AP_report\\Inter_U_AP')
    total = len(inter_U_AP_reports)
    len_subwords_list = []
    cnt=0
    for report in inter_U_AP_reports:
        cnt +=1
        print(cnt,'/88628', report) # for notice
        with open(report,'r') as f:
            sequence = f.read()
            len_subwords = len(tokenizer.encode(sequence)) - 2 # [CLS],[SEP]
            len_subwords_list.append(len_subwords)

    print('sum_subwords:', sum(len_subwords_list))
    print('avg_subwords:', sum(len_subwords_list)/total) # 8512928
    print('num_total:', len(len_subwords_list)) # 96.05235365798619
    print('max_:', max(len_subwords_list)) # 732
    print('min_:', min(len_subwords_list)) # 2

    with open('Inter_U_AP_len_subwords_list.pickle','wb') as f:
        pickle.dump(len_subwords_list, f)

#Inter_U_AP_reports_tokenizing()

with open('./Inter_U_AP_len_subwords_list.pickle','rb') as f:
    I_U_len_list = pickle.load(f)

I_U_len_list.sort()
# print(I_U_len_list)

print(I_U_len_list.index(395))

Under_100 = I_U_len_list[:54968]
Under_200 = I_U_len_list[54968:85986]
Under_300 = I_U_len_list[85986:88381]
Left = I_U_len_list[88381:]
total = [Under_100, Under_200, Under_300, Left]

for idx, case in enumerate(total):
    n, bins, _ = plt.hist(case)
    plt.show()
    print(idx)
    print(n)
    print(bins)
    print('________________________________')


