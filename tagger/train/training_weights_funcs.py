import numpy as np
import tensorflow as tf
import inspect

# Define training functions here
def effective_equal_classes(labels, region_weights, class_labels, bkg_weights=1.0):
    """
    Returns weights for each sample to balance classes, using *effective*
    (region-weighted) class populations rather than raw counts -- so class
    balancing accounts for kinematic-region weights already applied
    (e.g. from kinematic_region_weights), instead of being computed
    independently and then multiplied in afterward.

    Works for both integer binary/multiclass labels (shape (N,))
    and one-hot labels (shape (N, n_classes)).

    Parameters
    ----------
    labels : array
        Integer class labels or one-hot labels.
    region_weights : array, shape (N,)
        Per-sample weights from kinematic-region weighting. Used both to
        compute effective class counts and folded into the returned weight.
    bkg_weights : float
        Extra scalar multiplier applied to the background class (label 0).
    """
    if labels.ndim > 1 and labels.shape[-1] > 1:
        class_idx = np.argmax(labels, axis=-1)
    else:
        class_idx = labels.ravel().astype(int)

    n_classes = class_idx.max() + 1

    # effective (region-weighted) count per class, not raw event count
    effective_counts = np.bincount(class_idx, weights=region_weights, minlength=n_classes)
    total_effective = effective_counts.sum()
    effective_counts = np.clip(effective_counts, 1e-12, None)  # guard near-empty classes

    class_balance_weights = total_effective / (n_classes * effective_counts)
    weights = class_balance_weights[class_idx] * region_weights

    minbias_label = class_labels['MinBias']
    weights = np.where(class_idx == minbias_label, weights * bkg_weights, weights)

    # renormalize so mean weight is 1 (keeps effective LR stable, see earlier point)
    weights *= len(weights) / weights.sum()

    return weights

def kinematic_region(data, ht_cut, pt_cuts, labels, class_labels, model, signal_floor=0.1, minbias_floor=0.1):
    """
    Returns weights for signal and background depending on the kinematic region.
    Events already accepted through independet kinemtic cuts are downweighted.
    """
    jet_ht = data['jet_ht']
    ht_passed = jet_ht >= ht_cut
    pt_idx = model.inputs_config['collections']['sc4jets'][0].index('pt')
    leading_jet_passed = data['sc4jets'][:, :, pt_idx][:, 0] >= pt_cuts[0]
    subleading_jet_passed = data['sc4jets'][:, :, pt_idx][:, 1] >= pt_cuts[1]

    # combine the conditions to find events that pass any of the cuts
    already_covered = ht_passed | leading_jet_passed | subleading_jet_passed

    minbias_label = class_labels['MinBias']
    is_signal = (labels != minbias_label)
    is_minbias = ~is_signal

    weights = np.ones_like(labels, dtype=float)
    weights[is_signal]  = np.where(already_covered[is_signal],  signal_floor,  1.0)
    weights[is_minbias] = np.where(already_covered[is_minbias], minbias_floor, 1.0)

    return weights

# Wrapper to call training functions, must enter all training funtions here
# all function argumennts must be included in context dictionary passed to get_weights
WEIGHT_FUNCTIONS = {
    "kinematic_region": kinematic_region,
    "effective_equal_classes": effective_equal_classes,
}

def get_weights(name, **context):
    """
    Dispatch to a registered weighting function by name, passing only the
    subset of `context` that function actually accepts.

    Parameters
    ----------
    name : str
        Key into WEIGHT_FUNCTIONS.
    **context : dict
        Superset of all possible arguments any registered weighting function
        might need (e.g. data, ht_cut, pt_cuts, labels, class_labels,
        region_weights, signal_floor, minbias_floor, bkg_weights). Each
        function pulls out only what it declares in its signature.
    """
    if name not in WEIGHT_FUNCTIONS:
        raise ValueError(f"Unknown weighting function '{name}'. "
                          f"Available: {list(WEIGHT_FUNCTIONS)}")

    fn = WEIGHT_FUNCTIONS[name]
    sig = inspect.signature(fn)

    kwargs = {}
    for pname, param in sig.parameters.items():
        if pname in context:
            kwargs[pname] = context[pname]
        elif param.default is inspect.Parameter.empty:
            raise TypeError(f"'{name}' requires '{pname}' but it wasn't provided in context")
        # else: has a default, fine to omit

    return fn(**kwargs)
