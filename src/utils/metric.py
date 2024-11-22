import numpy as np
from scipy.spatial import KDTree
from functools import reduce
import math
from scipy.sparse import csr_matrix

def compute_M(data):
    cols = np.arange(data.size)
    return csr_matrix((cols, (data.ravel(), cols)), shape=(data.max() + 1, data.size))


def maximise_binary(arr):
    results = []
    # Perform rolling and find the maximum binary value
    if len(arr.shape) == 1: 
        arr = np.expand_dims(arr, axis=0)

    for j in arr:
        max_dec = 0
        for i in range(len(j)):
            rolled_arr = np.roll(j, i)
            decimal_value = reduce(lambda a,b: 2*a+b, rolled_arr)
            if decimal_value > max_dec: max_dec = decimal_value
        results.append(max_dec)
    # make shape [n, 1]
    results = np.array(results)
    results = np.expand_dims(results, axis=1)
    return results


def evaluate_metrics():
    pass


def recall_accuracy(emb_a, emb_b, gt_ori):
    est_ori = gt_ori 

    # Concatenate all batches
    db = np.concatenate(emb_a, axis=0)
    query = np.concatenate(emb_b, axis=0)
    est_ori = np.concatenate(est_ori, axis=0)
    gt_ori = np.concatenate(gt_ori, axis=0)

    # Get maximum binary value to decimal value
    rolled_gt_ori = maximise_binary(gt_ori)
    rolled_query_ori = maximise_binary(est_ori)
    db_length = db.shape[0]

    # Get hamming distance from each est_ori to gt_ori
    hamming = np.zeros((len(est_ori), len(gt_ori)))
    for i, est in enumerate(est_ori):
        for j, gt in enumerate(gt_ori):
            hamming[i, j] = np.sum(est != gt)

    gt_ori = gt_ori.tolist()

    tree = KDTree(db)
    percent = math.ceil(db_length/100)
    ks = [1, 5, 10, percent]
    hammings = [1, 2, 3]
    metrics = {}
    for k in ks: 
        metrics[k], metrics[f'{k}_y'], metrics[f'{k}_yb'] = 0, 0, 0
        # for h in hammings:
        #     metrics[f'{k}_yb_{h}'] = 0

    _, retrievals = tree.query(query, k=db_length)

    for gt_ind, ret_inds in enumerate(retrievals):
        # Single Image Retrieval Recall Accuracies
        for k in filter(lambda k: len(np.intersect1d(ret_inds[:k], gt_ind)) > 0, ks):
            metrics[k] += 1

        # Yaw Estimation Recall Accuracies - known bearing
        bear_yaw_ind = ret_inds[np.all((gt_ori[gt_ind]  == est_ori[ret_inds]), axis=1)]
        for k in filter(lambda k: len(np.intersect1d(bear_yaw_ind[:k], gt_ind)) > 0, ks):
            metrics[f'{k}_yb'] += 1

        # Yaw Estimation Recall Accuracies - unknown bearing
        max_dec_ind = ret_inds[np.all((np.expand_dims(rolled_gt_ori[gt_ind], 0) == rolled_query_ori[ret_inds]), axis=1)]
        for k in filter(lambda k: len(np.intersect1d(max_dec_ind[:k], gt_ind)) > 0, ks):
            metrics[f'{k}_y'] += 1

    for m in metrics:
        metrics[m] = round((metrics[m]/db_length)*100, 4)

    return metrics