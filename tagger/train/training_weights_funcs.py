import numpy as np
import tensorflow as tf

def equal_classes(labels, bkg_weights=1):
    """
    Returns weights for each sample to balance the classes.
    Works for both integer binary/multiclass labels (shape (N,))
    and one-hot labels (shape (N, n_classes)).
    """
    # Convert one-hot to integer class indices if needed
    if labels.ndim > 1 and labels.shape[-1] > 1:
        class_idx = np.argmax(labels, axis=-1)
    else:
        class_idx = labels.ravel().astype(int)

    class_counts = np.bincount(class_idx)
    total_samples = len(class_idx)
    class_weights = total_samples / (len(class_counts) * class_counts)
    weights = class_weights[class_idx]
    weights = np.where(class_idx == 0, weights * bkg_weights, weights)

    return weights
