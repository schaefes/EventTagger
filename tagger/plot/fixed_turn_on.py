import os
import argparse
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
from scipy.special import expit

from tagger.data.tools import load_data, totalMinBiasRate, load_test_split
from tagger.model.common import fromFolder
from tagger.plot import style

style.set_style()

X_VARS = {
    'HT': [np.arange(0, 1000, 20), "H$_T$ [GeV]"],
    'mjj': [np.arange(0, 1000, 40), "m$_{jj}$ [GeV]"],
}


_OUTPUT_KIND = {}  # cache: id(model) -> "logits" | "probs"


def detect_output_kind(model, raw_output, override="auto"):
    """
    Decide whether the model output is a logit or already a probability.

    1. Inspect the last layer's activation (Dense/QDense or an Activation/QActivation layer).
       'sigmoid' -> probs, 'linear'/None -> logits, 'softmax' -> error (not binary).
    2. If the layer is inconclusive, fall back to the output range:
       everything within [0, 1] -> probs, otherwise logits.
    3. If both are available and disagree, the layer wins and a warning is printed.
    """
    if override in ("logits", "probs"):
        return override

    key = id(model)
    if key in _OUTPUT_KIND:
        return _OUTPUT_KIND[key]

    last = model.event_model.layers[-1]
    act = str(last.get_config().get("activation", "")).lower()
    if "softmax" in act:
        raise ValueError("Last layer is softmax: this script is for a single-output binary model.")
    if "sigmoid" in act:
        from_layer = "probs"
    elif act in ("linear", "none", ""):
        from_layer = "logits"
    else:
        from_layer = None  # e.g. an unrecognised quantised activation

    lo, hi = float(np.min(raw_output)), float(np.max(raw_output))
    from_range = "probs" if (lo >= 0.0 and hi <= 1.0) else "logits"

    if from_layer is None:
        kind = from_range
        print(f"[output check] last layer activation '{act}' not recognised; "
              f"range [{lo:.3f}, {hi:.3f}] -> treating as {kind}")
    else:
        kind = from_layer
        msg = (f"[output check] last layer '{last.name}' activation '{act or 'none'}' -> {kind}; "
               f"output range [{lo:.3f}, {hi:.3f}]")
        if from_range != from_layer:
            msg += f"  WARNING: range suggests {from_range}. Override with --score-type if wrong."
        print(msg)

    _OUTPUT_KIND[key] = kind
    return kind


SCORE_TYPE = "auto"  # set from --score-type in __main__


def get_binary_scores(model, data, override=None):
    """Binary model output -> score in [0, 1] (sigmoid applied only if the output is a logit)."""
    override = SCORE_TYPE if override is None else override
    raw = np.asarray(model.event_model.predict(model.prepare_inputs(data)[0],
                                               batch_size=65536)).reshape(-1)
    kind = detect_output_kind(model, raw, override)
    return expit(raw) if kind == "logits" else raw


def wp_tag(nn_wp):
    return f"nn{int(round(nn_wp * 1000)):04d}"  # 0.8 -> nn0800


def rate_tag(rate):
    return f"rate{rate:g}"


def load_minbias(model):
    minbias_path = os.path.join(model.output_directory, "testing_data")
    minbias_split = load_test_split(["MinBias"], minbias_path)
    minbias, *_, extras = load_data(
        ["MinBias"], model, minbias_split, "test", extras=["central_jet_ht"]
    )
    scores = get_binary_scores(model, minbias)
    ht = np.asarray(extras["central_jet_ht"])
    if len(scores) == 0:
        raise ValueError("The MinBias test sample is empty.")
    if len(scores) != len(ht):
        raise ValueError("MinBias scores and HT arrays have different lengths.")
    return scores, ht


def find_threshold_at_rate(values, target_rate, total_rate, label):
    if target_rate < 0 or target_rate > total_rate:
        raise ValueError(
            f"{label} target rate must be between 0 and {total_rate:g} kHz."
        )
    values = np.asarray(values)
    order = np.sort(values)[::-1]
    n = len(order)
    target_pass = int(round(target_rate / total_rate * n))
    if target_pass == 0:
        threshold = np.nextafter(order[0], np.inf)
    elif target_pass >= n:
        threshold = np.nextafter(order[-1], -np.inf)
    else:
        threshold = order[target_pass]
    achieved_rate = np.count_nonzero(values > threshold) / n * total_rate
    return float(threshold), float(achieved_rate)


def set_working_point(model, nn_wp, save_path):
    """
    Fix the NN working point, measure the minbias rate it gives, and find the
    HT threshold with the same rate. Saves everything to one npz.
    """
    os.makedirs(save_path, exist_ok=True)
    scores, ht = load_minbias(model)
    n_bkg = len(scores)
    total = totalMinBiasRate()

    # rate at the chosen NN working point
    n_pass = int(np.count_nonzero(scores > nn_wp))
    p = n_pass / n_bkg
    nn_rate = p * total
    nn_rate_err = total * np.sqrt(p * (1 - p) / n_bkg)
    print(f"NN score > {nn_wp:.3f}: {n_pass} / {n_bkg} minbias events "
          f"-> {nn_rate:.2f} +- {nn_rate_err:.2f} kHz")

    # HT threshold letting the same number of minbias events through
    ht_wp, ht_rate = find_threshold_at_rate(ht, nn_rate, total, "HT")
    print(f"HT > {ht_wp:.1f} GeV: {ht_rate:.2f} kHz")

    np.savez(os.path.join(save_path, f"fixed_wp_{wp_tag(nn_wp)}.npz"),
             nn_wp=nn_wp, nn_rate_khz=nn_rate, nn_rate_err_khz=nn_rate_err,
             ht_wp=ht_wp, ht_rate_khz=ht_rate)
    return nn_rate, ht_wp


def set_working_points_at_rate(model, target_rate, save_path):
    """Derive independent NN and HT thresholds for one target MinBias rate."""
    scores, ht = load_minbias(model)
    total = totalMinBiasRate()
    nn_wp, nn_rate = find_threshold_at_rate(scores, target_rate, total, "NN")
    ht_wp, ht_rate = find_threshold_at_rate(ht, target_rate, total, "HT")
    output = os.path.join(save_path, f"{rate_tag(target_rate)}.npz")
    np.savez(
        output,
        target_rate_khz=target_rate,
        nn_wp=nn_wp,
        nn_rate_khz=nn_rate,
        ht_wp=ht_wp,
        ht_rate_khz=ht_rate,
    )
    print(
        f"Target {target_rate:g} kHz: NN > {nn_wp:.6g} ({nn_rate:.2f} kHz), "
        f"HT > {ht_wp:.1f} GeV ({ht_rate:.2f} kHz)"
    )
    return nn_wp, ht_wp, target_rate


def turn_on_curve_fixed_wp(
    signal_process, model, nn_wp, x_variable, save_path, ht_wp=None, nn_rate=None
):
    """Turn-on of NN (fixed score cut) vs HT cut at the same minbias rate."""
    proc_path = os.path.join(save_path, signal_process)
    os.makedirs(proc_path, exist_ok=True)

    if ht_wp is None or nn_rate is None:
        wp = np.load(os.path.join(save_path, f"fixed_wp_{wp_tag(nn_wp)}.npz"))
        ht_wp, nn_rate = float(wp["ht_wp"]), float(wp["nn_rate_khz"])

    signal_path = os.path.join(model.output_directory, "testing_data")
    signal_split = load_test_split([signal_process], signal_path)
    signal, *_, signal_extras = load_data([signal_process], model, signal_split, "test",
                                          extras=["genjets_pt", "central_jet_ht"])
    score = get_binary_scores(model, signal)
    signal_ht = np.asarray(signal_extras["central_jet_ht"], dtype=float)
    signal_gen_ht = np.asarray(
        ak.to_numpy(ak.sum(signal_extras["genjets_pt"], axis=1)), dtype=float
    )
    if x_variable != "HT":
        raise ValueError("Only x-variable 'HT' is available in signal extras.")
    if not (len(signal_gen_ht) == len(score) == len(signal_ht)):
        raise ValueError(
            "Signal scores, generator HT, and reconstructed HT arrays have "
            f"different lengths: scores={len(score)}, gen_ht={len(signal_gen_ht)}, "
            f"reco_ht={len(signal_ht)}."
        )
    if np.all(signal_gen_ht == 0):
        raise ValueError(
            "Generator HT is zero for every event. Check the generator-jet "
            "extra field used by the input files."
        )
    nn_passed = score > nn_wp
    ht_passed = signal_ht > ht_wp
    only_nn = nn_passed & ~ht_passed

    bins, x_label = X_VARS[x_variable]
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    h, _ = np.histogram(signal_gen_ht, bins=bins)
    if np.sum(h) > 0:
        h = h / np.sum(h)

    def eff_vs_x(passed):
        eff, err = np.zeros(len(bins) - 1), np.zeros(len(bins) - 1)
        for i in range(len(bins) - 1):
            m = (signal_gen_ht >= bins[i]) & (signal_gen_ht < bins[i + 1])
            n = np.sum(m)
            if n > 0:
                eff[i] = np.mean(passed[m])
                err[i] = np.sqrt(eff[i] * (1 - eff[i]) / n)
        return eff, err

    ht_eff, ht_err = eff_vs_x(ht_passed)
    nn_eff, nn_err = eff_vs_x(nn_passed)
    only_eff, only_err = eff_vs_x(only_nn)

    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, ax=ax,
                  fontsize=style.MEDIUM_SIZE - 2)
    ax.errorbar(bin_centers, ht_eff, yerr=ht_err, fmt="s-", capsize=3,
                label=rf"$H_T$ > {ht_wp:.1f} GeV @ {nn_rate:.1f}kHz ($\int \epsilon$={np.mean(ht_passed):.2f})")
    ax.errorbar(bin_centers, nn_eff, yerr=nn_err, fmt="o-", capsize=3,
                label=rf"NN score > {nn_wp:.3f} @ {nn_rate:.1f}kHz ($\int \epsilon$={np.mean(nn_passed):.2f})")
    ax.errorbar(bin_centers, only_eff, yerr=only_err, fmt="D-", capsize=3,
                label=rf"NN only (not $H_T$) ($\int \epsilon$={np.mean(only_nn):.2f})")
    ax.bar(bin_centers, h, width=np.diff(bins), alpha=0.2, color="gray")
    ax.axhline(1.0, color="gray", lw=0.8, ls="--")
    ax.set_xlabel(f"Gen {x_label}")
    ax.set_ylabel(f"Signal efficiency ({style.PROCESS_STYLE[signal_process]})")
    ax.set_ylim(-0.05, 1.2)
    ax.legend(loc="upper left")
    fig.tight_layout()
    base = os.path.join(proc_path, f"turn_on_fixedwp_{wp_tag(nn_wp)}_{x_variable}")
    fig.savefig(base + ".png", transparent=True)
    fig.savefig(base + ".pdf")
    plt.close(fig)


def _input_feature_values(data, model):
    features = []
    n_events = len(next(iter(data.values())))
    for object_name, (fields, _, collection_name) in model.inputs_config["objects"].items():
        input_name = next(
            (name for name in (object_name, collection_name) if name in data),
            None,
        )
        if input_name is None:
            raise KeyError(
                f"Input collection {collection_name!r} (or model input "
                f"{object_name!r}) is missing from loaded data."
            )
        values = np.asarray(data[input_name])
        if values.ndim != 3 or values.shape[2] != len(fields):
            raise ValueError(
                f"Input collection {input_name!r} has shape {values.shape}; "
                f"expected (events, objects, {len(fields)})."
            )
        valid = np.any(np.abs(values) > 0, axis=2)
        if "pt" in fields:
            valid = values[:, :, fields.index("pt")] > 0
        event_indices = np.repeat(np.arange(n_events), valid.sum(axis=1))
        for index, field in enumerate(fields):
            features.append(
                (f"{collection_name}_{field}", values[:, :, index][valid], event_indices)
            )
    for field in model.inputs_config["event_features"]:
        values = np.asarray(data[field]).reshape(n_events, -1)
        for index in range(values.shape[1]):
            name = field if values.shape[1] == 1 else f"{field}_{index}"
            features.append((name, values[:, index], np.arange(n_events)))
    return features


def plot_misclassified_inputs(
    model, signal_process, nn_wp, ht_wp, save_path, tag
):
    """Plot correctly and incorrectly selected signal and MinBias inputs."""
    data_path = os.path.join(model.output_directory, "testing_data")
    signal_split = load_test_split([signal_process], data_path)
    minbias_split = load_test_split(["MinBias"], data_path)
    signal, *_, signal_extras = load_data(
        [signal_process], model, signal_split, "test", extras=["central_jet_ht"]
    )
    minbias, *_, minbias_extras = load_data(
        ["MinBias"], model, minbias_split, "test", extras=["central_jet_ht"]
    )
    signal_scores = get_binary_scores(model, signal)
    minbias_scores = get_binary_scores(model, minbias)

    groups = [
        ("signal", signal, signal_scores > nn_wp),
        ("minbias", minbias, minbias_scores <= nn_wp),
    ]
    output_dir = os.path.join(save_path, signal_process, f"input_classification_{tag}")
    os.makedirs(output_dir, exist_ok=True)

    for classifier, threshold in (("nn", nn_wp), ("ht", ht_wp)):
        if classifier == "nn":
            decisions = [
                (name, data, selected)
                for name, data, selected in groups
            ]
        else:
            signal_ht = np.asarray(signal_extras["central_jet_ht"])
            minbias_ht = np.asarray(minbias_extras["central_jet_ht"])
            decisions = [
                ("signal", signal, signal_ht > threshold),
                ("minbias", minbias, minbias_ht <= threshold),
            ]

        for category, data, correct in decisions:
            for feature_name, values, event_indices in _input_feature_values(data, model):
                values = np.asarray(values, dtype=float)
                if values.size == 0:
                    continue
                low, high = np.nanmin(values), np.nanmax(values)
                if low == high:
                    delta = max(abs(low) * 0.05, 0.5)
                    low, high = low - delta, high + delta
                value_correct = correct[event_indices]
                correct_values = values[value_correct]
                incorrect_values = values[~value_correct]
                if correct_values.size == 0 and incorrect_values.size == 0:
                    continue
                fig, ax = plt.subplots(figsize=style.FIGURE_SIZE)
                bins = np.linspace(low, high, 51)
                if correct_values.size:
                    ax.hist(correct_values, bins=bins, density=True, histtype="step",
                            linewidth=style.LINEWIDTH, label="Correct")
                if incorrect_values.size:
                    ax.hist(incorrect_values, bins=bins, density=True, histtype="step",
                            linewidth=style.LINEWIDTH, label="Misclassified")
                hep.cms.label(
                    llabel=style.CMSHEADER_LEFT,
                    rlabel=style.CMSHEADER_RIGHT,
                    ax=ax,
                    fontsize=style.CMSHEADER_SIZE,
                )
                ax.set_xlabel(feature_name)
                ax.set_ylabel("Normalised events")
                ax.set_title(f"{style.PROCESS_STYLE.get(signal_process if category == 'signal' else 'MinBias', category)} ({classifier})")
                ax.legend(loc="best")
                fig.tight_layout()
                output = os.path.join(
                    output_dir, f"{category}_{classifier}_{feature_name}"
                )
                fig.savefig(output + ".png", bbox_inches="tight")
                fig.savefig(output + ".pdf", bbox_inches="tight")
                plt.close(fig)


def plot_working_point(signal_process, model, nn_wp, ht_wp, rate, x_variable, save_path, tag):
    """Plot signal turn-ons for explicitly supplied NN/HT working points."""
    turn_on_curve_fixed_wp(
        signal_process, model, nn_wp, x_variable, save_path, ht_wp, rate
    )
    proc_path = os.path.join(save_path, signal_process)
    for extension in ("png", "pdf"):
        old = os.path.join(
            proc_path, f"turn_on_fixedwp_{wp_tag(nn_wp)}_{x_variable}.{extension}"
        )
        new = os.path.join(proc_path, f"turn_on_{tag}_{x_variable}.{extension}")
        if os.path.exists(old):
            os.replace(old, new)
    print(
        f"Saved {os.path.join(proc_path, f'turn_on_{tag}_{x_variable}.png')} / .pdf"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal", type=str, default="GGFHHTo4B")
    parser.add_argument("--model", default="tagger/output/ffbinary", help="Input model folder")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--nn-wp", type=float, help="Fixed NN score working point")
    group.add_argument("--rate", type=float, help="Target MinBias rate in kHz")
    parser.add_argument("--x-var", type=str, default="HT", choices=list(X_VARS.keys()))
    parser.add_argument("--score-type", default="auto", choices=["auto", "logits", "probs"],
                        help="auto = detect from the last layer / output range (default)")
    args = parser.parse_args()
    SCORE_TYPE = args.score_type

    model = fromFolder(args.model)
    plot_dir = os.path.join(args.model, "plots/turn_on")
    os.makedirs(plot_dir, exist_ok=True)

    if args.nn_wp is not None:
        rate, ht_wp = set_working_point(model, args.nn_wp, plot_dir)
        tag = wp_tag(args.nn_wp)
        plot_working_point(
            args.signal, model, args.nn_wp, ht_wp, rate, args.x_var, plot_dir, tag
        )
    else:
        nn_wp, ht_wp, rate = set_working_points_at_rate(model, args.rate, plot_dir)
        tag = rate_tag(args.rate)
        plot_working_point(
            args.signal, model, nn_wp, ht_wp, rate, args.x_var, plot_dir, tag
        )
        plot_misclassified_inputs(
            model, args.signal, nn_wp, ht_wp, plot_dir, tag
        )
