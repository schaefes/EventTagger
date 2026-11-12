# flake8: noqa
import collections
import os

import awkward as ak
import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep

# Third parties
import pandas
import tensorflow as tf
from matplotlib.pyplot import cm
from scipy.stats import norm

setattr(collections, "MutableMapping", collections.abc.MutableMapping)
import histbook
import numpy as np
import shap
from sklearn.metrics import auc, roc_curve

from tagger.data.tools import load_data, load_test_split, totalMinBiasRate
from tagger.plot import style

from .common import PT_BINS, plot_histo

matplotlib.use('Agg')

plt.rcParams.update({'figure.max_open_warning': 0})

# some custom imports for efficiency plots
np.bool = np.bool_


style.set_style()

# DEFINE ALL THE PLOTTING FUNCTIONS HERE!!!! THEY WILL BE CALLED IN basic() function >>>>>>>


def loss_history(plot_dir, loss_names, history):
    for metric in loss_names:
        metric = 'loss'

        fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
        hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, ax=ax, fontsize=style.CMSHEADER_SIZE)
        ax.plot(history.history[metric], label='Train Loss', linewidth=style.LINEWIDTH)
        ax.plot(history.history['val_' + metric], label='Validation Loss', linewidth=style.LINEWIDTH)
        ax.grid(True)
        # ax.set_ylabel('Loss')
        ax.set_ylabel('Loss ' + metric)
        ax.set_xlabel('Epoch')
        ax.legend(loc='upper right')

        save_path = os.path.join(plot_dir, "loss_" + metric + "_history")
        plt.savefig(f"{save_path}.png", bbox_inches='tight')
        plt.savefig(f"{save_path}.pdf", bbox_inches='tight')

        fig.clf()


def ROC_binary(y_pred, y_true, class_labels, save_path):
    # calculate ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    max_rate = 150

    # fpr to rate
    hep.cms.label(
        llabel=style.CMSHEADER_LEFT,
        rlabel=style.CMSHEADER_RIGHT,
        fontsize=style.CMSHEADER_SIZE
    )
    fpr_rate = fpr * totalMinBiasRate()
    rate_mask = fpr_rate < max_rate
    rate = fpr_rate[rate_mask]
    tpr_rate = tpr[rate_mask]
    fig, ax = plt.subplots(figsize=style.FIGURE_SIZE)
    fig2, ax2 = plt.subplots(figsize=style.FIGURE_SIZE)
    ax.plot(fpr, tpr, color='cyan', lw=5, label='ROC curve (area = {:.2f})'.format(roc_auc))
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve ({style.PROCESS_STYLE[class_labels[1]]} vs {style.PROCESS_STYLE[class_labels[0]]})')
    ax.legend(loc='lower right')
    ax.grid()

    ax2.plot(rate, tpr_rate, color='cyan', lw=5, label='ROC curve (area = {:.2f})'.format(roc_auc))
    ax2.set_xlim([0, max_rate])
    ax2.set_ylim([0.0, 1])
    ax2.set_xlabel('Rate (kHz)')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curve with Rate')
    ax2.legend(loc='lower right')
    ax2.grid()

    fig.tight_layout()
    fig2.tight_layout()
    fig.savefig(os.path.join(save_path, 'roc_curve.png'))
    fig2.savefig(os.path.join(save_path, 'roc_curve_with_rate.png'), transparent=True)
    return


def ROC_ovo(y_pred, y_test, class_labels, plot_dir, class_pair, signal_proc=None):
    """
    Generate ROC curves comparing between two specific class labels.
    """
    save_dir = os.path.join(plot_dir, 'roc_ovo')
    os.makedirs(save_dir, exist_ok=True)

    # Ensure class_pair exists in class_labels
    assert (
        class_pair[0] in class_labels and class_pair[1] in class_labels
    ), "Both class_pair labels must exist in class_labels"

    # Get indices of the classes to compare
    idx1, idx2 = class_labels[class_pair[0]], class_labels[class_pair[1]]

    # Select true labels and predicted probabilities for the selected classes
    y_true1, y_true2 = y_test[:, idx1], y_test[:, idx2]
    y_score1, y_score2 = y_pred[:, idx1], y_pred[:, idx2]

    # Combine the labels and scores for binary classification
    selection = (y_true1 == 1) | (y_true2 == 1)
    y_true_binary = y_true1[selection]
    # Normalized probabilities
    y_score_binary = y_score1[selection] / (y_score1[selection] + y_score2[selection])

    # Compute FPR, TPR, and AUC
    fpr, tpr, _ = roc_curve(y_true_binary, y_score_binary)
    roc_auc = auc(fpr, tpr)

    fpr_rate = fpr * totalMinBiasRate()
    max_rate = 150
    rate_mask = fpr_rate < max_rate
    rate = fpr_rate[rate_mask]
    tpr_rate = tpr[rate_mask]

    # Plot the ROC curve
    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, ax=ax, fontsize=style.CMSHEADER_SIZE)
    ax.plot(
        fpr,
        tpr,
        label=f'{style.PROCESS_STYLE[class_pair[0]]} vs {style.PROCESS_STYLE[class_pair[1]]} (AUC = {roc_auc:.2f})',
        color='blue',
        linewidth=5,
    )
    ax.grid(True)
    ax.set_xlabel('Mistag Rate')
    ax.set_ylabel('Signal Efficiency')
    leg = ax.legend(loc='lower right', fontsize=style.SMALL_SIZE + 3, title=signal_proc)
    leg._legend_box.align = "left"

    # Save the plot
    save_path = os.path.join(save_dir, f"ROC_{class_pair[0]}_vs_{class_pair[1]}")
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.savefig(f"{save_path}.png", bbox_inches='tight')
    plt.close()

    # Plot the ROC curve with rate on the x-axis
    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, ax=ax, fontsize=style.CMSHEADER_SIZE)
    ax.plot(
        rate,
        tpr_rate,
        label=f'{style.PROCESS_STYLE[class_pair[0]]} vs {style.PROCESS_STYLE[class_pair[1]]} (AUC = {roc_auc:.2f})',
        color='blue',
        linewidth=5,
    )
    ax.grid(True)
    ax.set_xlabel('Mistag Rate (kHz)')
    ax.set_ylabel('Signal Efficiency')
    leg = ax.legend(loc='lower right', fontsize=style.SMALL_SIZE + 3, title=signal_proc)
    leg._legend_box.align = "left"

    # Save the plot
    save_path = os.path.join(save_dir, f"ROC_{class_pair[0]}_vs_{class_pair[1]}_with_rate")
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.savefig(f"{save_path}.png", bbox_inches='tight')
    plt.close()


def ROC(y_pred, y_test, class_labels, plot_dir, ROC_dict):
    # Create a colormap for unique colors
    # Use 'tab10' with enough colors
    colormap = cm.get_cmap('Set1', len(class_labels))

    # Create a plot for ROC curves
    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, ax=ax, fontsize=style.CMSHEADER_SIZE)
    for i in range(len(class_labels)):
        # Get true labels and predicted probabilities for the current class
        # Extract the one-hot column for the current class
        y_true = y_test[:, i]
        y_score = y_pred[:, i]  # Predicted probabilities for the current class

        # Compute FPR, TPR, and AUC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ROC_dict[class_labels[i]] = roc_auc
        # Plot the ROC curve for the current class
        ax.plot(
            fpr,
            tpr,
            label=f'{style.PROCESS_STYLE[class_labels[i]]} (AUC = {roc_auc:.2f})',
            color=colormap(i),
            linewidth=style.LINEWIDTH,
        )

    # Plot formatting
    ax.grid(True)
    ax.set_xlabel('Mistag Rate')
    ax.set_ylabel('Signal Efficiency')

    auc_list = [value for key, value in ROC_dict.items()]
    handles, labels = plt.gca().get_legend_handles_labels()
    order = np.argsort(auc_list)
    ax.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        loc='upper left',
        ncol=2,
        fontsize=style.SMALL_SIZE - 3,
    )

    # Save the plot
    save_path = os.path.join(plot_dir, "basic_ROC")
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.savefig(f"{save_path}.png", bbox_inches='tight')
    plt.close()

    return


def confusion(y_pred, y_test, class_labels, plot_dir):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    cm = confusion_matrix(
        np.argmax(y_test, axis=1),
        np.argmax(y_pred, axis=1),
        normalize="true",
    )
    cm = np.round(cm, 3)
    labels = [style.PROCESS_STYLE[class_labels[i]] for i in range(y_test.shape[1])]

    # Create a plot of the confusion matrix
    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, fontsize=style.CMSHEADER_SIZE)
    matrix_display = ConfusionMatrixDisplay(cm, display_labels=labels)

    matrix_display.plot(ax=ax)
    matrix_display.im_.set_clim(0, 1)

    # Remove default the colorbar
    matrix_display.im_.colorbar.remove()

    # Adjust colorbar height to match the plot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    plt.colorbar(matrix_display.im_, cax=cax)

    # Save the plot
    plt.savefig(os.path.join(plot_dir, f"confusion_matrix.png"), bbox_inches='tight')
    plt.savefig(os.path.join(plot_dir, f"confusion_matrix.pdf"), bbox_inches='tight')


def _input_feature_arrays(inputs, model, labels, predictions=None):
    configured_features = []

    for object_name, (features, _, collection_name) in model.inputs_config["objects"].items():
        input_name = next(
            (name for name in (collection_name, object_name) if name in inputs),
            None,
        )
        if input_name is None:
            raise KeyError(
                f"Input collection {collection_name!r} (or model input {object_name!r}) "
                "is missing from the test data."
            )
        values = np.asarray(inputs[input_name])
        if values.ndim != 3 or values.shape[2] != len(features):
            raise ValueError(
                f"Input collection {input_name!r} has shape {values.shape}; "
                f"expected (events, objects, {len(features)})."
            )
        valid = np.any(np.abs(values) > 0, axis=2)
        if "pt" in features:
            valid = values[:, :, features.index("pt")] > 0
        for object_index in range(values.shape[1]):
            object_valid = valid[:, object_index]
            for feature_index, feature_name in enumerate(features):
                feature_predictions = (
                    None if predictions is None else predictions[object_valid]
                )
                configured_features.append(
                    (
                        f"{collection_name}_{object_index + 1}_{feature_name}",
                        values[object_valid, object_index, feature_index],
                        labels[object_valid],
                        feature_predictions,
                    )
                )

    for feature_name in model.inputs_config["event_features"]:
        if feature_name not in inputs:
            raise KeyError(f"Event feature {feature_name!r} is missing from the test data.")
        values = np.asarray(inputs[feature_name]).reshape(len(labels), -1)
        for component in range(values.shape[1]):
            name = feature_name if values.shape[1] == 1 else f"{feature_name}_{component}"
            configured_features.append((name, values[:, component], labels, predictions))
    return configured_features


def _classification_labels(labels, predictions, class_labels):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    if labels.ndim > 1:
        labels = labels.reshape(-1) if labels.shape[1] == 1 else np.argmax(labels, axis=1)
    if predictions.ndim > 1:
        predictions = predictions.reshape(-1) if predictions.shape[1] == 1 else np.argmax(predictions, axis=1)
    if predictions.dtype.kind == "f":
        binary_indices = sorted(class_labels.values())
        if len(binary_indices) != 2:
            raise ValueError("Floating-point predictions require exactly two classes.")
        predictions = (predictions >= 0.5).astype(int)
    return labels.astype(int), predictions.astype(int)


def plot_input_features(inputs, model, class_labels, plot_dir, labels=None):
    """Plot configured input-feature distributions, separated by class."""
    plot_dir = os.path.join(plot_dir, "inputs")
    os.makedirs(plot_dir, exist_ok=True)

    if labels is None:
        labels = np.zeros(len(next(iter(inputs.values()))), dtype=int)
    labels = np.asarray(labels)
    if labels.ndim > 1:
        labels = labels.reshape(-1) if labels.shape[1] == 1 else np.argmax(labels, axis=1)
    if labels.shape[0] != len(next(iter(inputs.values()))):
        raise ValueError("Input arrays and labels contain different numbers of events.")

    class_names = {int(index): name for name, index in class_labels.items()}
    configured_features = _input_feature_arrays(inputs, model, labels)

    for feature_name, values, feature_labels, _ in configured_features:
        values = np.asarray(values, dtype=float)
        feature_labels = np.asarray(feature_labels)
        finite = np.isfinite(values)
        values, feature_labels = values[finite], feature_labels[finite]
        if values.size == 0:
            continue
        low, high = np.min(values), np.max(values)
        if low == high:
            delta = max(abs(low) * 0.05, 0.5)
            low, high = low - delta, high + delta
        bins = np.linspace(low, high, 51)

        fig, ax = plt.subplots(figsize=style.FIGURE_SIZE)
        for class_index in sorted(np.unique(feature_labels)):
            selected = values[feature_labels == class_index]
            if selected.size == 0:
                continue
            name = class_names.get(int(class_index), str(class_index))
            label = style.PROCESS_STYLE.get(name, name)
            ax.hist(
                selected,
                bins=bins,
                weights=np.ones(selected.size) / selected.size,
                histtype="step",
                linewidth=style.LINEWIDTH,
                label=label,
            )
        hep.cms.label(
            llabel=style.CMSHEADER_LEFT,
            rlabel=style.CMSHEADER_RIGHT,
            ax=ax,
            fontsize=style.CMSHEADER_SIZE,
        )
        ax.set_xlabel(feature_name)
        ax.set_ylabel("Normalised events")
        ax.legend(loc="best")
        fig.tight_layout()
        output = os.path.join(plot_dir, f"input_{feature_name}")
        fig.savefig(output + ".png", bbox_inches="tight")
        fig.savefig(output + ".pdf", bbox_inches="tight")
        plt.close(fig)


def shapPlot(shap_values, feature_names, class_names):
    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    feature_order = np.argsort(np.sum(np.mean(np.abs(shap_values), axis=1), axis=0))
    num_features = shap_values[0].shape[1]
    feature_inds = feature_order
    y_pos = np.arange(len(feature_inds))
    left_pos = np.zeros(len(feature_inds))

    axis_color = "#333333"
    class_inds = np.argsort([-np.abs(shap_values[i]).mean() for i in range(len(shap_values))])
    # Use 'tab10' with enough colors
    colormap = cm.get_cmap('Set1', len(class_names))

    for i, ind in enumerate(class_inds):
        global_shap_values = np.abs(shap_values[ind]).mean(0)
        label = style.PROCESS_STYLE[class_names[ind]]
        ax.barh(
            y_pos,
            global_shap_values[feature_inds],
            0.7,
            left=left_pos,
            align='center',
            label=label,
            color=colormap(class_inds[i]),
        )
        left_pos += global_shap_values[feature_inds]

    # ax.set_yticklabels([style.INPUT_FEATURE_STYLE[feature_names[i]] for i in feature_inds])
    ax.legend(loc='lower right', fontsize=30)

    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(color=axis_color, labelcolor=axis_color)
    ax.set_yticks(
        range(len(feature_order)), [style.INPUT_FEATURE_STYLE[feature_names[i]] for i in feature_order], fontsize=30
    )
    ax.set_xlabel("mean (Shapley value) - (average impact on model output magnitude)", fontsize=30)
    plt.tight_layout()


def plot_shaply(model, X_test, class_labels, input_vars, plot_dir):

    labels = list(class_labels.keys())
    model2 = tf.keras.Model(model.event_model.input, model.event_model.output[0])
    model3 = tf.keras.Model(model.event_model.input, model.event_model.output[1])

    for explainer, name in [
        (shap.GradientExplainer(model2, X_test[:1000]), "GradientExplainer"),
    ]:
        print("... {0}: explainer.shap_values(X)".format(name))
        shap_values = explainer.shap_values(X_test[:1000])
        new = np.sum(shap_values, axis=1)
        print("... shap summary_plot classification")
        plt.clf()
        new = np.transpose(new, (2, 0, 1))
        shapPlot(new, input_vars, labels)
        plt.savefig(plot_dir + "/shap_summary_class.pdf", bbox_inches='tight')
        plt.savefig(plot_dir + "/shap_summary_class.png", bbox_inches='tight')

    for explainer, name in [
        (shap.GradientExplainer(model3, X_test[:1000]), "GradientExplainer"),
    ]:
        print("... {0}: explainer.shap_values(X)".format(name))
        shap_values = explainer.shap_values(X_test[:1000])
        new = np.sum(shap_values, axis=1)
        print("... shap summary_plot regression")
        plt.clf()
        labels = ["Regression"]
        new = np.transpose(new, (2, 0, 1))
        shapPlot(new, input_vars, labels)
        plt.savefig(plot_dir + "/shap_summary_reg.pdf", bbox_inches='tight')
        plt.savefig(plot_dir + "/shap_summary_reg.png", bbox_inches='tight')


def basic(model):
    """
    Plot the basic ROCs for different classes. Does not reflect L1 rate
    Returns a dictionary of ROCs for each class
    """

    plot_dir = os.path.join(model.output_directory, "plots/training")

    ROC_dict = {class_label: 0 for class_label in model.class_labels}

    # Load the testing data
    testing_data = np.load(f"{model.output_directory}/testing_data/test_data.npz")
    X_test = {k: testing_data[k] for k in testing_data.files}
    y_test = np.load(f"{model.output_directory}/testing_data/test_labels.npz")['labels']
    y_pred = model.event_model.predict(X_test)

    # Plot ROC curves
    inverted_labels = {v: k for k, v in model.class_labels.items()}
    is_binary = model.training_config['binary']
    if not is_binary:
        # Confusion matrix
        confusion(y_pred, y_test, inverted_labels, plot_dir)
        ROC(y_pred, y_test, inverted_labels, plot_dir, ROC_dict)
        class_pairs = []
        # Generate all possible pairs of classes
        for i in model.class_labels.keys():
            for j in model.class_labels.keys():
                if j != "MinBias":
                    continue
                if i != j:
                    class_pair = [i, j]
                    class_pairs.append(class_pair)

        # Plot the binary ROCs for each class pair
        for class_pair in class_pairs:
            ROC_ovo(y_pred, y_test, model.class_labels, plot_dir, class_pair)
    else:
        ROC_binary(y_pred, y_test, inverted_labels, plot_dir)


    # The saved test_data.npz contains model-prepared inputs (for combined
    # models this is only ``model_input``), so reload raw collections for plots.
    processes = [
        name for name, _ in sorted(model.class_labels.items(), key=lambda item: item[1])
    ]
    test_split = load_test_split(
        processes, os.path.join(model.output_directory, "testing_data")
    )
    raw_test_data, raw_labels, _ = load_data(processes, model, test_split, "test")

    # Plot input distributions
    plot_input_features(raw_test_data, model, model.class_labels, plot_dir, raw_labels)

    # Plot the shaply feature importance
    # plot_shaply(model, X_test, model.class_labels, model.input_vars, plot_dir)

    return
