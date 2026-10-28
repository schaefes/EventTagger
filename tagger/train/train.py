import tensorflow as tf
import numpy as np
import awkward as ak
import argparse
import os
import json
from tagger.train.training_weights_funcs import get_weights
from argparse import ArgumentParser
from tagger.model.common import fromFolder, fromYaml
from tagger.data.tools import load_data, split_raw
from tagger.plot.basic import basic

tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

def save_test_split(split, save_dir):
    """Save the test events of each process as <save_dir>/<proc>_test.npz (all arrays)."""
    os.makedirs(save_dir, exist_ok=True)
    for proc, arrays in split["test"].items():
        np.savez_compressed(os.path.join(save_dir, f"{proc}_test.npz"), **arrays)

def train(model, model_path, processes):
    # load data
    test_ratios = {'MinBias': 0.3, 'GGFHHTo4B': 0.2, 'GGFHHTo2B2Tau': 0.2}
    split = split_raw(processes, model, test_ratios)
    train_data, train_labels, labels_dict, extras = load_data(processes, model, split, "train", extras=["central_jet_ht"])
    test_data, test_labels, _ = load_data(processes, model, split, "test")
    train_inputs, train_shapes = model.prepare_inputs(train_data)
    test_inputs, _ = model.prepare_inputs(test_data)
    model.set_labels(
        model.inputs_config["collections"],
        model.inputs_config["event_features"],
        labels_dict,
    )

    # save data
    os.makedirs(os.path.join(model_path, "testing_data"), exist_ok=True)
    save_test_split(split, os.path.join(model_path, "testing_data"))
    np.savez_compressed(os.path.join(model_path, "testing_data/test_data.npz"), **test_inputs)
    np.savez(os.path.join(model_path, "testing_data/test_labels.npz"), labels=test_labels)
    np.savez(os.path.join(model_path, 'testing_data/labels_dict.npz'), **labels_dict)

    # identify output shape and build model
    n_procs = len(processes)
    output_shape = 1 if model.training_config["binary"] else n_procs
    if (n_procs > 2 and model.training_config["binary"]) or (n_procs <= 2 and not model.training_config["binary"]):
        raise ValueError("Binary type not matching the number of processes. Please check the model config.")
    model.build_model(train_shapes, output_shape)

    # Train it with a pruned model
    num_samples = train_labels.shape[0] * (1 - model.training_config['validation_split'])

    model.compile_model(num_samples)

    context = dict(
        data=train_data, jet_ht=extras['central_jet_ht'], labels=train_labels, class_labels=labels_dict,
        ht_cut=350, pt_cuts=[np.inf, np.inf], signal_floor=0.9, minbias_floor=0.9,
        bkg_weights=8.0, region_weights=np.ones(train_labels.shape[0]),
        model=model, alpha=0.1, max_w=20.0
    )

    # Training weights, function defined in training_weights.py
    training_weights = np.ones(train_labels.shape[0])
    for w in model.training_config["weight_method"]:
        training_weights *= get_weights(w, **context)
        context["region_weights"] = training_weights
    model.fit(train_inputs, train_labels, training_weights)

    # Hard negative mining weights and training
    hnm_weights = np.ones(train_labels.shape[0])
    for w in ["hnm_weights", "effective_equal_classes"]:
        hnm_weights *= get_weights(w, **context)
        context["region_weights"] = hnm_weights
    model.fit(train_inputs, train_labels, hnm_weights)

    # Finished training, save model
    model.save()

    model.plot_loss()
    return

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--processes", nargs="+", default=["MinBias", "VBF"], help="List of processes")
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--model-config', type=str, default='ffbinary.yaml')
    parser.add_argument('--model-name', type=str, default='ffbinary')
    args = parser.parse_args()
    args.processes.sort()


    procs_sorted = sorted(args.processes)
    model_dir = os.path.join(f"output/{args.model_name}", "_".join(procs_sorted))
    # train the model
    if args.train:
        model_config = os.path.join("tagger/model/configs/", args.model_config)
        model = fromYaml(model_config, model_dir)
        train(model, model_dir, args.processes)

    if args.evaluate:
        model = fromFolder(model_dir)
        results = basic(model)
