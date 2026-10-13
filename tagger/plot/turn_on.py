import os
import argparse
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep
from tagger.data.processes import DATASETS
from tagger.data.tools import load_data, totalMinBiasRate
from tagger.model.common import fromFolder
from tagger.plot import style
style.set_style()

X_VARS = {
    'HT': [np.arange(0, 1000, 20), "H$_T$ [GeV]"],
    'mjj': [np.arange(0, 1000, 40), "m$_{jj}$ [GeV]"]
    }

def find_working_points(model, signal, target_rates, save_path, tolerance=0.5, n_grid=900):
    """
    Find the score threshold such that the fraction of background (minbias)
    events passing `score > threshold`, scaled by `input_rate_khz`, equals
    `target_rate`.

    Parameters
    ----------
    bkg_scores : array-like
        Model scores for a representative minbias (background) sample.
    target_rate : float
        Desired output rate after the cut.
    n_grid : int
        Number of threshold points to scan (finer grid -> more precise wp).

    Returns
    -------
    threshold : float
        Score threshold achieving (approximately) the target rate.
    achieved_rate_khz : float
        The actual rate obtained at that threshold (closest achievable,
        since the background sample is finite).
    """
    proc_path = os.path.join(save_path, signal)
    os.makedirs(proc_path, exist_ok=True)
    minbias, *_, minbias_extras = load_data(['MinBias'], model, percentage=100, test_ratio=0., extras=['central_jet_ht'])
    model_data = model.prepare_inputs(minbias)[0]
    preds = model.event_model.predict(model_data)
    scores, score_names = model.get_scores(preds, signal)

    # Scan thresholds from min to max score
    thresholds = np.linspace(0.88, 1, n_grid)
    n_bkg = len(preds)

    for target_rate in target_rates:
        # get thresholds for all types of scores
        for score, score_name in zip(scores, score_names):

            # For each threshold, fraction passing -> rate
            # (searchsorted on sorted scores is O(log n) per threshold, fast)
            sorted_scores = np.sort(score)
            n_passing = n_bkg - np.searchsorted(sorted_scores, thresholds, side="left")
            pass_fraction = n_passing / n_bkg
            rate_khz = pass_fraction * totalMinBiasRate()

            # Pick the threshold in a tolerance window around the target rate that is closest to the target
            idx = np.argmin(np.abs(rate_khz - target_rate))
            if np.abs(rate_khz[idx] - target_rate) > tolerance:
                raise ValueError(f"Could not find a threshold achieving {target_rate} kHz within {tolerance} kHz. "
                                f"Closest achievable rate is {rate_khz[idx]:.2f} kHz at threshold {thresholds[idx]:.4f}.")
            np.savez(os.path.join(proc_path, f"rate_scan_wp_{score_name}_{int(target_rate)}.npz"), threshold=thresholds[idx], rate_khz=rate_khz[idx])

        # ht wp
        get_ht_wp(minbias_extras['central_jet_ht'], target_rate, tolerance, save_path)

    return

def get_ht_wp(jet_ht, target_rate_khz, tolerance, save_path):
    """
    Find the HT threshold such that the fraction of background (minbias)
    events passing `HT > threshold`, scaled by `input_rate_khz`, equals
    `target_rate_khz`.

    Parameters
    ----------
    ht : array-like
        HT values for a representative minbias (background) sample.
    target_rate_khz : float
        Desired output rate after the cut.

    Returns
    -------
    threshold : float
        HT threshold achieving (approximately) the target rate.
    achieved_rate_khz : float
        The actual rate obtained at that threshold (closest achievable,
        since the background sample is finite).
    """
    n_bkg = len(jet_ht)

    # Scan thresholds from min to max HT
    thresholds = np.linspace(200, 800, 400)

    # For each threshold, fraction passing -> rate
    sorted_ht = np.sort(jet_ht)
    n_passing = n_bkg - np.searchsorted(sorted_ht, thresholds, side="left")
    pass_fraction = n_passing / n_bkg
    rate_khz = pass_fraction * totalMinBiasRate()

    # Pick the threshold in a tolerance window around the target rate that is closest to the target
    idx = np.argmin(np.abs(rate_khz - target_rate_khz))
    if np.abs(rate_khz[idx] - target_rate_khz) > tolerance:
        raise ValueError(f"Could not find an HT threshold achieving {target_rate_khz} kHz within 0.5 kHz."
                         f"Closest achievable rate is {rate_khz[idx]:.2f} kHz at threshold {thresholds[idx]:.4f}.")

    np.savez(os.path.join(save_path, f"ht_scan_wp_{int(target_rate_khz)}.npz"), threshold=thresholds[idx], rate_khz=rate_khz[idx])

    # rate for menu HT seed
    menu_pass = jet_ht > 450
    menu_rate = np.mean(menu_pass) * totalMinBiasRate()
    np.savez(os.path.join(save_path, f"ht_menu_rate.npz"), threshold=450, rate_khz=menu_rate)

    return

def get_or_rate(score, minbias_ht, nn_wp, ht_wp):
    """
    Compute the rate of events passing either the NN score threshold or the HT threshold.

    Parameters
    ----------
    minbias_ht : array-like
        HT values for a representative minbias (background) sample.
    minbias_scores : array-like
        Model scores for a representative minbias (background) sample.
    nn_wp : float
        Neural network score threshold.
    ht_wp : float
        HT threshold.

    Returns
    -------
    or_rate_khz : float
        The actual rate obtained at that threshold (closest achievable,
        since the background sample is finite).
    """

    nn_passed = score > nn_wp
    ht_passed = minbias_ht > ht_wp
    or_passed = nn_passed | ht_passed

    n_bkg = len(score)
    n_passing = np.sum(or_passed)
    pass_fraction = n_passing / n_bkg
    or_rate = pass_fraction * totalMinBiasRate()
    print(f"OR rate at NN WP {nn_wp:.4f} and HT WP {ht_wp:.1f} GeV: {or_rate:.2f} kHz")

    return or_rate

def turn_on_curve(signal_process, model, target_rates, x_variable, save_path):
    """
    Compute signal efficiency (turn-on curve) of `score > threshold` binned
    in `sig_variable`, with binomial (Clopper-Pearson-ish normal approx) errors.

    Returns bin_centers, efficiency, efficiency_err (both arrays of length len(bins)-1)
    """
    proc_path = os.path.join(save_path, signal_process)
    os.makedirs(proc_path, exist_ok=True)
    menu_seed = np.load(os.path.join(save_path, f"ht_menu_rate.npz"))
    signal, *_, signal_extras = load_data([signal_process], model, percentage=100, test_ratio=0., extras=['genjets_pt', 'central_jet_ht'])
    minbias, *_, minbias_extras = load_data(['MinBias'], model, percentage=100, test_ratio=0., extras=['central_jet_ht'])
    signal_gen_pt, signal_ht = signal_extras['genjets_pt'], signal_extras['central_jet_ht']
    minbias_ht = minbias_extras['central_jet_ht']
    signal_gen_ht = ak.sum(signal_gen_pt, axis=1)
    preds = model.event_model.predict(model.prepare_inputs(signal)[0])
    minbias_preds = model.event_model.predict(model.prepare_inputs(minbias)[0])
    scores, score_names = model.get_scores(preds, signal_process)
    minbias_scores, _ = model.get_scores(minbias_preds, signal_process)
    for rate in target_rates:
        for score, minbias_score, score_name in zip(scores, minbias_scores, score_names):
            nn_wp = np.load(os.path.join(proc_path, f"rate_scan_wp_{score_name}_{int(rate)}.npz"))['threshold']
            ht_wp = np.load(os.path.join(save_path,f"ht_scan_wp_{int(rate)}.npz"))['threshold']
            menu_ht = menu_seed['threshold']
            nn_passed = score > nn_wp
            ht_passed = signal_ht > ht_wp
            or_passed = nn_passed.flatten() | ht_passed
            menu_passed = signal_ht > menu_ht
            or_rate = get_or_rate(minbias_score, minbias_ht, nn_wp, ht_wp)
            menu_rate = menu_seed['rate_khz']

            nn_total_eff = np.mean(nn_passed)
            ht_total_eff = np.mean(ht_passed)
            or_total_eff = np.mean(or_passed)
            menu_total_eff = np.mean(menu_passed)

            bins, x_label = X_VARS[x_variable]
            bin_centers = 0.5 * (bins[1:] + bins[:-1])
            nn_efficiency = np.zeros(len(bins) - 1)
            nn_efficiency_err = np.zeros(len(bins) - 1)
            ht_efficiency = np.zeros(len(bins) - 1)
            ht_efficiency_err = np.zeros(len(bins) - 1)
            or_efficiency = np.zeros(len(bins) - 1)
            or_efficiency_err = np.zeros(len(bins) - 1)
            menu_efficiency = np.zeros(len(bins) - 1)
            menu_efficiency_err = np.zeros(len(bins) - 1)
            h, edges = np.histogram(signal_gen_ht, bins=bins)  # just to check binning
            h = h / np.sum(h)  # normalize to 1

            # Turn on vs gen variable
            for i in range(len(bins) - 1):
                bin_mask = (signal_gen_ht >= bins[i]) & (signal_gen_ht < bins[i + 1])
                nn_passed_in_bin = nn_passed[bin_mask]
                ht_passed_in_bin = ht_passed[bin_mask]
                or_passed_in_bin = or_passed[bin_mask]
                menu_passed_in_bin = menu_passed[bin_mask]
                n_total = np.sum(bin_mask)
                if n_total > 0:
                    nn_efficiency[i] = np.mean(nn_passed_in_bin)
                    nn_efficiency_err[i] = np.sqrt(nn_efficiency[i] * (1 - nn_efficiency[i]) / n_total)
                    ht_efficiency[i] = np.mean(ht_passed_in_bin)
                    ht_efficiency_err[i] = np.sqrt(ht_efficiency[i] * (1 - ht_efficiency[i]) / n_total)
                    or_efficiency[i] = np.mean(or_passed_in_bin)
                    or_efficiency_err[i] = np.sqrt(or_efficiency[i] * (1 - or_efficiency[i]) / n_total)
                    menu_efficiency[i] = np.mean(menu_passed_in_bin)
                    menu_efficiency_err[i] = np.sqrt(menu_efficiency[i] * (1 - menu_efficiency[i]) / n_total)
                else:
                    continue

            # plotting
            fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
            hep.cms.label(llabel=style.CMSHEADER_LEFT, rlabel=style.CMSHEADER_RIGHT, ax=ax, fontsize=style.MEDIUM_SIZE-2)
            ax.errorbar(bin_centers, ht_efficiency, yerr=ht_efficiency_err, fmt="s-", capsize=3,
                label=f"$H_T$ @ {rate}kHz, $H_T$ = {ht_wp:.1f} GeV ($\int \epsilon$={ht_total_eff:.2f})")
            ax.errorbar(bin_centers, nn_efficiency, yerr=nn_efficiency_err, fmt="o-", capsize=3,
                label=f"NN ({style.SCORE_LABEL_STYLE[score_name]}) @ {rate}kHz, nn score = {nn_wp:.3f} ($\int \epsilon$={nn_total_eff:.2f})")
            ax.errorbar(bin_centers, or_efficiency, yerr=or_efficiency_err, fmt="^-", capsize=3,
                label=f"NN | $H_T$ @ {or_rate:.1f}kHz ($\int \epsilon$={or_total_eff:.2f})")
            ax.errorbar(bin_centers, menu_efficiency, yerr=menu_efficiency_err, fmt="x-", capsize=3,
                label=f"Menu $H_T$ @ {menu_rate:.1f}kHz ($\int \epsilon$={menu_total_eff:.2f})")
            ax.bar(bin_centers, h, width=np.diff(bins), alpha=0.2, color="gray")
            ax.axhline(1.0, color="gray", lw=0.8, ls="--")
            ax.set_xlabel(f"Gen {x_label}")
            ax.set_ylabel(f"Signal efficiency ({style.PROCESS_STYLE[signal_process]})")
            ax.set_ylim(-0.05, 1.2)
            ax.legend(loc="upper left")
            fig.tight_layout()
            fig.savefig(os.path.join(proc_path, f"turn_on_{score_name}_{x_variable}_{int(rate)}kHz.png"), transparent=True)
            fig.savefig(os.path.join(proc_path, f"turn_on_{score_name}_{x_variable}_{int(rate)}kHz.pdf"))
            plt.close(fig)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal", type=str, default="GGFHHTo4B")
    parser.add_argument("--model", default='tagger/output/ffbinary', help = 'Input model')
    parser.add_argument("--rates", nargs="+", default=[14, 30, 50], help="Rates to scan for working points (kHz)")

    #Different modes
    parser.add_argument('--deriveWPs', action='store_true', help='derive the working points for di-taus')
    parser.add_argument('--eff', action='store_true', help='plot efficiency for VBF-> tautau')

    args = parser.parse_args()

    model = fromFolder(args.model)

    # find wps
    plot_dir = os.path.join(args.model, "plots/turn_on")
    os.makedirs(plot_dir, exist_ok=True)
    if args.deriveWPs:
        find_working_points(model, args.signal, args.rates, plot_dir)

    # plot turn-on curves
    if args.eff:
        turn_on_curve(args.signal, model, args.rates, 'HT', plot_dir)
