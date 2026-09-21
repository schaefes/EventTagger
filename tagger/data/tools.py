# Python
import gc
import json
import os
import shutil

import awkward as ak

# Third party
import numpy as np
import tensorflow as tf
import uproot
import yaml
from tqdm import tqdm

# Dataset configuration
from .config import EXTRA_FIELDS, FILTER_PATTERN, INPUT_TAG, N_PARTICLES
from tagger.data.processes import DATASETS, PROC_PATHS, DATA_PATH

gc.set_threshold(0)

def extract_array(tree, field, entry_stop):
    """
    Extracts an array from the tree with a limit on the number of entries.
    """
    return tree[field].array(entry_stop=entry_stop)


def extract_nn_inputs(data, input_vars, n_parts=16, n_entries=None):
    """
    Extract nn inputs based on the input_vars list
    """

    # Concatenate all the inputs
    inputs_list = []

    for field in input_vars:

        field_array = extract_array(data, f"jet_puppicand_{field}", n_entries)

        padded_filled_array = _pad_fill(field_array, n_parts)
        inputs_list.append(padded_filled_array[:, :, np.newaxis])

    # batch_size, n_particles, n_features
    inputs = ak.concatenate(inputs_list, axis=2)

    return inputs


def to_ML(data, class_labels):
    """
    Take in the data from make_data (loaded by load_data) and make them ready for training.
    """

    X = np.asarray(data['nn_inputs'])
    y = tf.keras.utils.to_categorical(np.asarray(data['class_label']), num_classes=len(class_labels))
    pt_target = np.asarray(data['target_pt'])
    truth_pt = np.asarray(data['target_pt_phys'])
    reco_pt = np.asarray(data['jet_pt_phys'])

    return X, y, pt_target, truth_pt, reco_pt


def load_data(processes, model, percentage=100, test_ratio=0.2):
    """
    Load a specified percentage of the dataset using uproot.concatenate.

    Parameters:
        processes (list): List of process names to load.
        model (str): Model name to determine the input variables.
        percentage (int): Percentage of the dataset to load (default is 100).

    Returns:
        awkward.Array: Concatenated data arrays split for train and test.
    """
    processes = sorted(processes, key=lambda p: (p != "MinBias", p))
    label_map = {proc: i for i, proc in enumerate(processes)}
    collection_fields = list(model.inputs_config["collections"].keys())
    event_fields = model.inputs_config["event_features"]
    fields = collection_fields + event_fields
    labels_dict = {}
    data_by_field = {f: [] for f in fields}
    labels = []
    for proc in processes:
        proc_paths = DATASETS[proc]
        for path in proc_paths:
            p = np.load(os.path.join(path, "data.npz"))
            for k in model.inputs_config['collections']:
                n_objects = model.inputs_config['collections'][k][1]
                n_fields = len(model.inputs_config['collections'][k][0])
                collection_object = np.empty((len(p[f"{k}_{model.inputs_config['collections'][k][0][0]}"]), n_fields * n_objects))
                for i, f in enumerate(model.inputs_config['collections'][k][0]):
                    collection_object[:, i::n_fields] = p[f"{k}_{f}"]
                data_by_field[k].append(collection_object.reshape(collection_object.shape[0], n_objects, n_fields))

            for f in event_fields:
                arr = p[f]
                data_by_field[f].append(arr)

            labels.append(np.full(len(arr), label_map[proc]))
        labels_dict[proc] = label_map[proc]
    data = {f: np.concatenate(arrs, axis=0) for f, arrs in data_by_field.items()}
    labels = np.concatenate(labels, axis=0)

    # convert labels to one-hot encoding if more than 2 classes
    if len(processes) > 2:
        labels = np.eye(len(processes))[labels]

    # Shuffle the data
    n_total = len(labels)
    rng = np.random.default_rng(seed=42)  # set a seed for reproducibility, or drop it
    perm = rng.permutation(n_total)

    data = {f: arr[perm] for f, arr in data.items()}
    labels = labels[perm]

    # split into train and test sets
    split_idx = int((1 - test_ratio) * n_total)
    train_data = {f: arr[:split_idx] for f, arr in data.items()}
    test_data = {f: arr[split_idx:] for f, arr in data.items()}
    train_labels = labels[:split_idx]
    test_labels = labels[split_idx:]

    return train_data, test_data, train_labels, test_labels, labels_dict

def get_input_mask(jets_ds):
    n_features = 10
    input_mask = np.where(np.sum(jets_ds, axis=2) == 0, 0, 1)
    input_mask = np.repeat(input_mask[:, :,np.newaxis], n_features, axis=2)

    return input_mask

def totalMinBiasRate(nCollBunch = 2808):
    LHCfreq = 11245.6 # in Hz
    return LHCfreq * nCollBunch / 1e3 # in kHz
