import torch
import numpy as np
from scipy.optimize import linear_sum_assignment


def split_cluster_acc_v2(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    mask: np.ndarray, 
    use_balanced: bool = False
) -> tuple[float, float, float]:
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how 
    good the accuracy is on subsets

    Args:
        y_true (`np.ndarray)`: labels, numpy.array with shape `(n_samples,)`
        y_pred (`np.ndarray)`: predicted labels, numpy.array with shape `(n_samples,)`
        mask (`np.ndarray)`: Which instances come from old classes (True) and which ones come 
            from new classes (False)
        use_balanced (`bool`): Whether to compute accuracy for each class 
            separately and average
    Returns:
        `float`: accuracy, in [0,1]
    """
    
    if y_pred.size != y_true.size:
        raise ValueError("Number of predicted labels does not match the number of true labels")
    
    y_true = y_true.astype(int)
    
    old_classes_gt, new_classes_gt = set(y_true[mask]), set(y_true[~mask])
    D = max(y_pred.max(), y_true.max()) + 1
    W = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        W[y_pred[i], y_true[i]] += 1
    
    res = linear_sum_assignment(W.max() - W) # Hungarian algorithm, maximize
    res = np.vstack(res).T # tuple([id], [worker]) -> array[(id, worker)]
    
    index_map = {j: i for i, j in res}
    
    old_acc = np.zeros(len(old_classes_gt) if use_balanced else 1)
    old_samples = np.zeros(len(old_classes_gt) if use_balanced else 1)
    new_acc = np.zeros(len(new_classes_gt) if use_balanced else 1)
    new_samples = np.zeros(len(new_classes_gt) if use_balanced else 1)
    
    for idx, i in enumerate(old_classes_gt):
        idx = idx if use_balanced else 0
        old_acc[idx] += W[index_map[i], i]
        old_samples[idx] += W[:, i].sum()
    
    for idx, i in enumerate(new_classes_gt):
        idx = idx if use_balanced else 0
        new_acc[idx] += W[index_map[i], i]
        new_samples[idx] += W[:, i].sum()
    
    # Avoid division by zero 
    if any(old_samples == 0):
        raise ValueError("No instances from some old classe(s)")
    if any(new_samples == 0):
        raise ValueError("No instances from some new classe(s)")
    
    total_acc = (
        np.concatenate([old_acc, new_acc])
        / np.concatenate([old_samples, new_samples])
    )
    
    old_acc /= old_samples
    new_acc /= new_samples
    
    (
        total_acc, old_acc, new_acc
    ) = total_acc.mean(), old_acc.mean(), new_acc.mean()
    
    return total_acc, old_acc, new_acc
    
    
    