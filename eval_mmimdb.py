
import os
import pickle
from sklearn.metrics import (f1_score,
                             multilabel_confusion_matrix)
import torch
import sys
import numpy as np

label_to_class = ['Drama', 'Comedy', 'Romance', 'Thriller', 'Crime', 'Action', 'Adventure', 'Horror', 'Documentary', 'Mystery', 'Sci-Fi', 'Fantasy', 'Family', 'Biography', 'War', 'History', 'Music', 'Animation', 'Musical', 'Western', 'Sport', 'Short', 'Film-Noir']

with open('1.pkl', 'rb') as f:
    data_list = pickle.load(f)
    
y_true = []
y_pred = []
for data in data_list:
    pred_score = data['pred_score']
    pred_score = torch.where(pred_score > 0.5, torch.tensor(1), torch.tensor(0))
    gt_score = data['gt_score']
    y_true.append(gt_score)
    y_pred.append(pred_score)

y_pred = torch.stack(y_pred).numpy()
y_true = torch.stack(y_true).numpy()

_dev_f1_weighted = f1_score(y_true, y_pred, average="weighted")
_dev_f1_micro = f1_score(y_true, y_pred, average="micro")
_dev_f1_macro = f1_score(y_true, y_pred, average="macro")
_dev_f1_samples = f1_score(y_true, y_pred, average="samples")

print('_dev_f1_weighted:', _dev_f1_weighted)
print('_dev_f1_micro:',_dev_f1_micro)
print('_dev_f1_macro', _dev_f1_macro)
print('_dev_f1_samples',_dev_f1_samples)


sample_num = np.sum(y_true, axis=0)
for i, conf_matrix in enumerate(multilabel_confusion_matrix(y_true, y_pred)):
    tn, fp, fn, tp = conf_matrix.ravel()
    f1 = 2 * tp / (2 * tp + fp + fn + sys.float_info.epsilon)
    recall = tp / (tp + fn + sys.float_info.epsilon)
    print(f'Label: {label_to_class[i]} f1={f1:.5f} sample_num={sample_num[i]} recall={recall:.5f}')



