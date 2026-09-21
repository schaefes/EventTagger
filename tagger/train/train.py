import tensorflow as tf
import numpy as np
import awkward as ak
import argparse
import os
import json
import tagger.train.training_weights_funcs as weighting_funcs
from argparse import ArgumentParser
from tagger.model.common import fromFolder, fromYaml
from tagger.data.tools import load_data
from tagger.plot.basic import basic

tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

def train(model, model_path, processes):
    # load data
    train_data, test_data, train_labels, test_labels, labels_dict = load_data(processes, model)

    train_inputs, train_shapes = model.prepare_inputs(train_data)
    test_inputs, _ = model.prepare_inputs(test_data)
    model.set_labels(
        model.inputs_config["collections"],
        model.inputs_config["event_features"],
        labels_dict,
    )

    # save data
    os.makedirs(os.path.join(model_path, "testing_data"), exist_ok=True)
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

    # Training weights, function defined in training_weights.py
    training_weights = np.ones(train_labels.shape[0])
    for w in model.training_config["weight_method"]:
        training_weights *= getattr(weighting_funcs, w)(train_labels)
    model.fit(train_inputs, train_labels, training_weights)

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
    parser.add_argument('--model-name', type=str, default='/eos/user/s/stella/EventTagger/output/ffbinary')
    args = parser.parse_args()
    args.processes.sort()

    os.makedirs(args.model_name, exist_ok=True)

    # train the model
    if args.train:
        model_config = os.path.join("tagger/model/configs/", args.model_config)
        procs_soreted = sorted(args.processes)
        model_name = os.path.join(args.model_name, "_".join(procs_soreted))
        model = fromYaml(model_config, args.model_name)
        train(model, args.model_name, args.processes)

    if args.evaluate:
        model = fromFolder(args.model_name)
        results = basic(model)
