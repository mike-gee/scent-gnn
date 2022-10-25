import numpy as np
import pandas as pd


def get_eval_metrics(actual_y: np.Array, pred_y: np.Array, class_labels: np.Array):
    """Get the precision and recall metrics for input and output metrics

    Args:
        actual_y: Numpy array of ground truth results.
        pred_y: Numpy array of predicted results.
        class_labels: Numpy array of class labels.

    Returns:
        Dataframe with metrics.

    """

    def eval_i(i):
        TP = ((actual_y[:, i] == 1) & (pred_y[:, i] == 1)).sum()
        FP = ((actual_y[:, i] == 1) & (pred_y[:, i] == 0)).sum()
        TN = ((actual_y[:, i] == 0) & (pred_y[:, i] == 0)).sum()
        FN = ((actual_y[:, i] == 0) & (pred_y[:, i] == 1)).sum()
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        data = {
            "class": class_labels[i],
            "precision": precision.item(),
            "recall": recall.item(),
            "TP": TP.item(),
            "FP": FP.item(),
            "TN": TN.item(),
            "FN": FN.item(),
        }
        return data

    return pd.DataFrame([eval_i(i) for i in range(actual_y.shape[1])])
