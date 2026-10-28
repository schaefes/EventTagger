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
import zlib
from collections import Counter

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


def split_raw(processes, model, test_ratio, seed=42, default_ratio=0.2):
    """
    test_ratio: dict {process: ratio}, e.g. {"MinBias": 0.4, "VBF": 0.2}.
                If it is not a dict, every process uses 0.2.
                Processes missing from the dict use default_ratio.

    Returns {proc: {"train": {key: array}, "test": {key: array}}} with ALL arrays,
    concatenated over the files of the process. Each split also has
    "file_id" (index into DATASETS[proc]) and "event_id" (row in that file).
    """
    if not isinstance(test_ratio, dict):
        test_ratio = {proc: 0.2 for proc in processes}

    # a key known to be per-event, used to get the number of events
    _, (first_names, _, first_obj) = next(iter(model.inputs_config['objects'].items()))
    ref_key = f"{first_obj}_{first_names[0]}"

    split = {"train": {}, "test": {}}
    for proc in processes:
        ratio = test_ratio.get(proc, default_ratio)
        parts = {"train": {}, "test": {}}              # key -> list of arrays, one per file
        static = {}                                     # non-event arrays, kept once

        for i_file, path in enumerate(DATASETS[proc]):
            p = np.load(os.path.join(path, "data.npz"))
            n_events = len(p[ref_key])

            # independent, reproducible seed per process and per file
            rng = np.random.default_rng([seed, zlib.crc32(proc.encode()), i_file])
            perm = rng.permutation(n_events)
            n_test = int(round(ratio * n_events))
            evs = {"test": np.sort(perm[:n_test]), "train": np.sort(perm[n_test:])}

            for k in p.files:                          # one array in memory at a time
                arr = p[k]
                if arr.ndim >= 1 and arr.shape[0] == n_events:
                    for name, ev in evs.items():
                        parts[name].setdefault(k, []).append(arr[ev])
                else:
                    static.setdefault(k, arr)          # keep the first file's copy
            for name, ev in evs.items():
                parts[name].setdefault("event_id", []).append(ev)
                parts[name].setdefault("file_id", []).append(np.full(len(ev), i_file))

        for name in ("train", "test"):
            split[name][proc] = {k: np.concatenate(v, axis=0) for k, v in parts[name].items()}
            split[name][proc].update(static)
    return split


def load_test_split(processes, save_dir):
    """
    Load the files written by save_test_split into the structure load_data expects:
    {"test": {proc: arrays}}.
    """
    return {"test": {proc: np.load(os.path.join(save_dir, f"{proc}_test.npz"))
                     for proc in processes}}


def load_data(processes, model, split, name, extras=[]):
    """
    Format one part (train or test) of the split returned by split_raw.

    Returns:
        data, labels, labels_dict (+ extra_fields if extras is non-empty)
    """
    processes = sorted(processes, key=lambda p: (p != "MinBias", p))
    label_map = {proc: i for i, proc in enumerate(processes)}
    fields = model.inputs_config["collections"] + model.inputs_config["event_features"]
    labels_dict = {}
    data_by_field = {f: [] for f in fields}
    labels = []
    extra_fields = {e: [] for e in extras} if extras else None

    for proc in processes:
        p = split[name][proc]
        for extra in extras:
            extra_fields[extra].append(p[extra])
        for k in model.inputs_config['objects']:
            n_objects = model.inputs_config['objects'][k][1]
            coll_name = model.inputs_config['objects'][k][2]
            n_fields = len(model.inputs_config['objects'][k][0])
            n_events = len(p[f"{coll_name}_{model.inputs_config['objects'][k][0][0]}"])
            loaded_object = np.empty((n_events, n_fields * n_objects))
            for i, f in enumerate(model.inputs_config['objects'][k][0]):
                loaded_object[:, i::n_fields] = p[f"{coll_name}_{f}"][:, :n_objects]
            data_by_field[k].append(loaded_object.reshape(loaded_object.shape[0], n_objects, n_fields))

        for f in model.inputs_config["event_features"]:
            if f not in model.inputs_config['objects']:
                arr = p[f]
                data_by_field[f].append(arr)

        labels.append(np.full(n_events, label_map[proc]))
        labels_dict[proc] = label_map[proc]

    data = {f: np.concatenate(arrs, axis=0) for f, arrs in data_by_field.items()}
    labels = np.concatenate(labels, axis=0)
    extra_fields = {e: np.concatenate(arrs, axis=0) for e, arrs in extra_fields.items()} if extras else None

    # convert labels to one-hot encoding if more than 2 classes
    if len(processes) > 2:
        labels = np.eye(len(processes))[labels]

    # Shuffle the training data only (the test order is kept)
    if name == "train":
        rng = np.random.default_rng(seed=42)
        perm = rng.permutation(len(labels))
        data = {f: arr[perm] for f, arr in data.items()}
        labels = labels[perm]
        extra_fields = {e: arr[perm] for e, arr in extra_fields.items()} if extras else None

    if extras:
        return data, labels, labels_dict, extra_fields
    return data, labels, labels_dict


def get_input_mask(jets_ds):
    n_features = 10
    input_mask = np.where(np.sum(jets_ds, axis=2) == 0, 0, 1)
    input_mask = np.repeat(input_mask[:, :,np.newaxis], n_features, axis=2)

    return input_mask

def totalMinBiasRate(nCollBunch = 2808):
    LHCfreq = 11245.6 # in Hz
    return LHCfreq * nCollBunch / 1e3 # in kHz
