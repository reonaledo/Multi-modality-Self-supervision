import json
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def get_labels_and_frequencies(path):
    label_freqs = Counter()
    data_labels = [json.loads(line)["label"] for line in open(path)]
    if type(data_labels) == list:
        for label_row in data_labels:
            if label_row == '':
                label_row = ["'Others'"]
            else:
                label_row=label_row.split(', ')  
                
            label_freqs.update(label_row)
    else: pass
    return list(label_freqs.keys()), label_freqs

    
def get_label_accuracy(target_path, reference_path='/home/mimic-cxr/dataset/image_preprocessing/Test.jsonl'):
    data = [json.loads(l) for l in open(reference_path)]
    labels, _ = get_labels_and_frequencies(reference_path)
    #ans = np.zeros((len(data), len(labels)))
    
    ans = []
    for index in range(len(data)):
        if data[index]["label"] == '':
            data[index]["label"] = "'Others'"
        label = np.zeros(len(labels), dtype=int)
        label[
                [labels.index(tgt) for tgt in data[index]["label"].split(', ')]
            ] = 1
        ans.append(label)
    ans = np.array(ans)
    df=pd.read_csv(target_path)
    df['Others'] = 0
    for i in range(len(df)):
        if (df.iloc[i] == 1).sum() == 0:
            df.loc[i, 'Others'] = 1.0
    df = df[ [label[1:-1] for label in labels] ]
    preds = df.to_numpy()
    preds = (preds == 1).astype(int)
    
    # accuracy 
    temp = ((ans + preds) == 2)
    correct_num = temp.sum(axis = 1)
    acc_per_instance = correct_num / ans.sum(axis = 1)
    acc = np.mean(acc_per_instance)
    # Precision
    precision = precision_score(ans, preds, average="micro")
    # Recall
    recall = recall_score(ans, preds, average="micro")
    # F1
    f1 = f1_score(ans, preds, average="micro")
    return acc, precision, recall, f1
    

def get_label_accuracy_v2(hypothesis, reference):
    df_hyp = pd.read_csv(hypothesis)
    df_ref = pd.read_csv(reference)
    
    df_result = (df_hyp == df_ref)
    acc_list = []
    for row in range(len(df_result)):
        if df_ref.iloc[row].isnull().sum() == 14:
            continue
        acc = df_result.iloc[row].sum() / (14 - df_ref.iloc[row].isnull().sum())
        acc_list.append(acc)
    acc_array = np.array(acc_list)
    
    return np.mean(acc_array), acc_array

def get_label_accuracy_v3(target, reference):
    df_tgt = pd.read_csv('/home/jhmoon/NegBio/chexpert-labeler/'+target)
    df_ref = pd.read_csv('/home/jhmoon/NegBio/chexpert-labeler/'+reference)    
    positive_tgt = df_tgt.isin([1.0])
    negative_tgt = df_tgt.isin([0.0])
    ambi_tgt = df_tgt.isin([-1.0])
    positive_ref = df_ref.isin([1.0])
    negative_ref = df_ref.isin([0.0])
    ambi_ref = df_ref.isin([-1.0])
    all_result = (df_tgt == df_ref)
    acc_list = []
    pos_precision = []
    neg_precision = []
    amb_precision = []
    
    pos_recall = []
    neg_recall = []
    amb_recall = []
    all_precision_lt = []
    all_recall_lt = []
    for row in range(len(df_tgt)):
        if len(positive_ref.loc[row].unique()) != 1:
            positive_precision = precision_score(positive_ref.loc[row],positive_tgt.loc[row], average="binary", pos_label=True)
            positive_recall = recall_score(positive_ref.loc[row],positive_tgt.loc[row], average="binary", pos_label=True)
            pos_precision.append(positive_precision)
            pos_recall.append(positive_recall)

        if len(negative_ref.loc[row].unique()) != 1:
            negative_precision = precision_score(negative_ref.loc[row],negative_tgt.loc[row], average="binary", pos_label=True)
            negative_recall = recall_score(negative_ref.loc[row],negative_tgt.loc[row], average="binary", pos_label=True)
            neg_precision.append(negative_precision)
            neg_recall.append(negative_recall)

        if len(ambi_ref.loc[row].unique()) != 1:
            ambi_precision = precision_score(ambi_ref.loc[row],ambi_tgt.loc[row], average="binary", pos_label=True)
            ambi_recall = recall_score(ambi_ref.loc[row],ambi_tgt.loc[row], average="binary", pos_label=True)
            amb_precision.append(ambi_precision)
            amb_recall.append(ambi_recall)

        acc_for_every_class = accuracy_score(df_ref.iloc[row,1:].fillna(4).values, df_tgt.iloc[row,1:].fillna(4).values)
        all_precision = precision_score(df_ref.iloc[row,1:].fillna(4).values, df_tgt.iloc[row,1:].fillna(4).values, average='macro')
        all_recall = recall_score(df_ref.iloc[row,1:].fillna(4).values, df_tgt.iloc[row,1:].fillna(4).values, average='macro')
        
        acc_list.append(acc_for_every_class)
        all_precision_lt.append(all_precision)
        all_recall_lt.append(all_recall)
    acc_array = np.mean(acc_list)
    pos_precision = np.mean(pos_precision)
    pos_recall = np.mean(pos_recall)
    neg_precision = np.mean(neg_precision)
    neg_recall = np.mean(neg_recall)
    amb_precision = np.mean(amb_precision)
    amb_recall = np.mean(amb_recall)
    all_precision_lt = np.mean(all_precision_lt)
    all_recall_lt = np.mean(all_recall_lt)

    return acc_array, pos_precision, pos_recall, neg_precision, neg_recall, amb_precision, amb_recall, all_precision_lt, all_recall_lt

def get_label_accuracy_v4(hypothesis, reference):
    df_hyp = pd.read_csv('/home/jhmoon/NegBio/chexpert-labeler/'+hypothesis)
    df_ref = pd.read_csv('/home/jhmoon/NegBio/chexpert-labeler/'+reference)
    df_hyp_pos1 = (df_hyp == 1).astype(int)
    del df_hyp_pos1["Reports"]
    df_hyp_pos1 = np.array(df_hyp_pos1)
    
    df_ref_pos1 = (df_ref == 1).astype(int)
    del df_ref_pos1["Reports"]
    df_ref_pos1 = np.array(df_ref_pos1)
    df_hyp_0 = (df_hyp == 0).astype(int)
    del df_hyp_0["Reports"]
    df_hyp_0 = np.array(df_hyp_0)
    df_ref_0 = (df_ref == 0).astype(int)
    del df_ref_0["Reports"]
    df_ref_0 = np.array(df_ref_0)
    df_hyp_neg1 = (df_hyp == -1).astype(int)
    del df_hyp_neg1["Reports"]
    df_hyp_neg1 = np.array(df_hyp_neg1)
    df_ref_neg1 = (df_ref == -1).astype(int)
    del df_ref_neg1["Reports"]
    df_ref_neg1 = np.array(df_ref_neg1)
    df_hyp_all = df_hyp_pos1 + df_hyp_0 + df_hyp_neg1
    df_ref_all = df_ref_pos1 + df_ref_0 + df_ref_neg1

    # Accuarcy
    accuracy_pos1 = (df_ref_pos1 == df_hyp_pos1).sum() / df_ref_pos1.size
    accuracy_0 = (df_ref_0 == df_hyp_0).sum() / df_ref_0.size
    accuracy_neg1 = (df_ref_neg1 == df_hyp_neg1).sum() / df_ref_neg1.size
    accuracy_all = (df_ref_all == df_hyp_all).sum() / df_ref_all.size

    # Precision
    precision_pos1 = precision_score(df_ref_pos1, df_hyp_pos1, average="micro")
    precision_0 = precision_score(df_ref_0, df_hyp_0, average="micro")
    precision_neg1 = precision_score(df_ref_neg1, df_hyp_neg1, average="micro")
    precision_all = precision_score(df_ref_all, df_hyp_all, average="micro")

    # Recall
    recall_pos1 = recall_score(df_ref_pos1, df_hyp_pos1, average="micro")
    recall_0 = recall_score(df_ref_0, df_hyp_0, average="micro")
    recall_neg1 = recall_score(df_ref_neg1, df_hyp_neg1, average="micro")
    recall_all = recall_score(df_ref_all, df_hyp_all, average="micro")

    # F1
    f1_pos1 = f1_score(df_ref_pos1, df_hyp_pos1, average="micro")
    f1_0 = f1_score(df_ref_0, df_hyp_0, average="micro")
    f1_neg1 = f1_score(df_ref_neg1, df_hyp_neg1, average="micro")
    f1_all = f1_score(df_ref_all, df_hyp_all, average="micro")

    return (accuracy_pos1, precision_pos1, recall_pos1, f1_pos1), (accuracy_0, precision_0, recall_0, f1_0), (accuracy_neg1, precision_neg1, recall_neg1, f1_neg1), (accuracy_all, precision_all, recall_all, f1_all)
    
if __name__ == '__main__':

    metric_pos1, metric_0, metric_neg1, metric_all = get_label_accuracy_v4(hypothesis = 'small_sc_50ep_4baem.csv', reference = 'base_sc_30ep_4beam_gt.csv')
    print("(micro) accuracy, precision, recall, f1 for all : {}, {}, {}, {}".format(round(metric_all[0], 4), round(metric_all[1], 4), round(metric_all[2], 4), round(metric_all[3], 4)))
    print("(micro) accuracy, precision, recall, f1 for postive: {}, {}, {}, {}".format(round(metric_pos1[0], 4), round(metric_pos1[1], 4), round(metric_pos1[2], 4), round(metric_pos1[3], 4)))
    print("(micro) accuracy, precision, recall, f1 for negative: {}, {}, {}, {}".format(round(metric_0[0], 4), round(metric_0[1], 4), round(metric_0[2], 4), round(metric_0[3], 4)))
    print("(micro) accuracy, precision, recall, f1 for ambi: {}, {}, {}, {}".format(round(metric_neg1[0], 4), round(metric_neg1[1], 4), round(metric_neg1[2], 4), round(metric_neg1[3], 4)))