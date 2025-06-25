
import scanpy as sc

import warnings
import numpy as np
warnings.filterwarnings("ignore")
import pandas as pd
####################################Settings#################################


test_data = sc.read_h5ad('data/ms/0/ms_test0.h5ad')
train_data = sc.read_h5ad('data/ms/0/ms_train0.h5ad')

unique_labels = np.unique(train_data.obs["celltype"])
label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
test_label = test_data.obs["celltype"].map(label_to_int).to_numpy()
train_label = train_data.obs["celltype"].map(label_to_int).to_numpy()

ref_cell_embeddings =np.load('results/train_deepcdr_01B-resolution_singlecell_cell_embedding_none_max_resolution.npy')
test_emebd =np.load('results/test_deepcdr_01B-resolution_singlecell_cell_embedding_none_max_resolution.npy')

def l2_sim(a, b):
    sims = -np.linalg.norm(a - b, axis=1)
    return sims


def get_similar_vectors(vector, ref, top_k=10):
    # sims = cos_sim(vector, ref)
    sims = l2_sim(vector, ref)

    top_k_idx = np.argsort(sims)[::-1][:top_k]
    return top_k_idx, sims[top_k_idx]

top_k = 10
idx_list=[i for i in range(test_emebd.shape[0])]
preds = []
for k in idx_list:
    idx, sim = get_similar_vectors(test_emebd[k][np.newaxis, ...], ref_cell_embeddings, top_k)
    pred = pd.Series(train_label)[idx].value_counts()
    preds.append(pred.index[0])
gt = test_label

from sklearn.metrics import precision_score, recall_score, f1_score, balanced_accuracy_score
celltypes_labels = gt
predictions = np.array(preds)


balanced_accuracy = balanced_accuracy_score(celltypes_labels, predictions)
f1 = f1_score(celltypes_labels, predictions, average="macro")
precision = precision_score(celltypes_labels, predictions, average="macro")
recall = recall_score(celltypes_labels, predictions, average="macro")

print(
    f"macro Accuracy: {balanced_accuracy:.3f}, macro Precision: {precision:.3f},macro Recall: {recall:.3f}, "f"macro F1: {f1:.3f}",
    flush=True)
micro_f1 = f1_score(celltypes_labels, predictions, average="micro")
micro_precision = precision_score(celltypes_labels, predictions, average="micro")
micro_recall = recall_score(celltypes_labels, predictions, average="micro")
print(
    f"micro Accuracy: {balanced_accuracy:.3f}, micro Precision: {micro_precision:.3f},micro Recall: {micro_recall:.3f}, "f"micro F1: {micro_f1:.3f}",
    flush=True)








