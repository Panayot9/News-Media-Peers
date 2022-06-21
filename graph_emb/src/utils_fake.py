import datetime
import dgl
import errno
import numpy as np
import os
import pickle
import random
import torch

from pprint import pprint
from scipy import sparse
from scipy import io as sio

from sklearn.metrics import f1_score, roc_auc_score

import pandas as pd
import numpy

def score(pred, labels):
    labels = labels.cpu().numpy()
    pred = pred.cpu().numpy()
    #pred = np.array([x.argmax() for x in pred])

    #print("pred: ", pred)
    #print("labels: ", labels)
    accuracy = (pred == labels).sum() / len(pred)
    micro_f1 = f1_score(labels, pred, average='micro')
    macro_f1 = f1_score(labels, pred, average='macro')
    try:
        f1 = f1_score(labels, pred, average='weighted')
    except:
        print('Exception occurred while calculating F1 Score', labels, pred)
        f1 = 0
#     auc = roc_auc_score(labels, pred, multi_class='ovo', average='weighted')
    
    # print('SCORE', accuracy, f1, auc)
    return accuracy, f1, accuracy


def gen_xavier(xavier_file = './data/city/city_xavier.csv',
                         node_count=263148, feature_count=32):
    x = torch.empty(node_count, feature_count)
    torch.nn.init.xavier_normal_(x)

    x = x.t()
    df_feat = pd.DataFrame()

    for i in range (0,feature_count):
        df_feat[f'x{i}'] = x[i].tolist()

    df_feat.to_csv(xavier_file, index=False)