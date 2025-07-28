# src/utils.py

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

def compute_metrics(eval_pred):
    """
    Computes accuracy, precision, recall, and f1-score for binary classification.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # For binary classification, 'binary' average is appropriate
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

