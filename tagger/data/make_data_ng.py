import os
import glob
import argparse
import numpy as np
import awkward as ak
import uproot
from coffea.nanoevents.methods import vector

TREE = "outnano/Jets"
NJ, NL = 16, 5  # padding sizes, same as your make_data stage

# ---- column mapping: jet-level file -> names used in your make_data stage ----
# Change these if you want different choices (e.g. jet_pt_corr instead of jet_pt_phys)
KIN = {"pt": "jet_pt_phys", "eta": "jet_eta_phys", "phi": "jet_phi_phys", "mass": "jet_mass_phys"}
SCORES = {  # order matches the ngjet score order in make_data
    "udsTagScore":   "jet_SC4NGJet_score_light",
    "bTagScore":     "jet_SC4NGJet_score_b",
    "cTagScore":     "jet_SC4NGJet_score_charm",
    "gTagScore":     "jet_SC4NGJet_score_gluon",
    "tau_nTagScore": "jet_SC4NGJet_score_taum",
    "tau_pTagScore": "jet_SC4NGJet_score_taup",
    "eTagScore":     "jet_SC4NGJet_score_electron",
    "muTagScore":    "jet_SC4NGJet_score_muon",
}
SC4_BTAG = "jet_bjetscore"  # used for sc4jets_btagScore (assumption, see notes)
EXTRA = ["event", SC4_BTAG, *KIN.values(), *SCORES.values()]


# ------------------------------------------------------------------ loading
def load_events(path):
    """Read jet-level tree(s) and return an event-shaped awkward record array of jets."""
    files = sorted(glob.glob(os.path.join(path, "*.root"))) if os.path.isdir(path) else sorted(glob.glob(path))
    per_file = []
    for f in files:
        with uproot.open(f) as fh:
            flat = fh[TREE].arrays(EXTRA, library="np")
        per_file.append(to_event_shape(flat))
    return ak.concatenate(per_file)


def to_event_shape(flat):
    """Group flat jets by 'event' -> jagged (n_events, var) array of jets."""
    ev = flat["event"].astype(np.int64)
    order = np.argsort(ev, kind="stable")            # make rows of one event contiguous
    ev = ev[order]
    _, counts = np.unique(ev, return_counts=True)    # unique returns sorted ids -> matches order

    def col(name):
        return ak.unflatten(flat[name][order], counts)

    fields = {k: col(v) for k, v in KIN.items()}
    fields.update({k: col(v) for k, v in SCORES.items()})
    fields["btagScore"] = col(SC4_BTAG)
    jets = ak.zip(fields, behavior=vector.behavior, with_name="Momentum4D")
    return jets[ak.argsort(jets.pt, axis=1, ascending=False)]


# ------------------------------------------------------------------ helpers (as in make_data)
def inv_mass(j1, j2):
    return np.sqrt(j1.pt * j2.pt * (np.cosh(abs(j1.eta - j2.eta)) - np.cos(abs(j1.phi - j2.phi))))

def dEta(j1, j2):
    return abs(j1.eta - j2.eta)

def return_jet_field(collection, leading_phi, f):
    if f == "phi":
        phi = ak.fill_none(getattr(collection, f), 0)
        projected = ak.to_numpy(phi) - ak.to_numpy(leading_phi)[:, np.newaxis]
        return np.mod(projected + np.pi, 2 * np.pi) - np.pi
    return ak.to_numpy(ak.fill_none(getattr(collection, f), 0))

def collection_field_arrays(collection, name, leading_phi, fields):
    return {f"{name}_{k}": return_jet_field(collection, leading_phi, k) for k in fields}

def make_pairs(jets):
    jets = jets[:, :4]
    pairs = ak.combinations(jets, 2, axis=1, fields=["j1", "j2"])
    parent = pairs.j1 + pairs.j2
    return ak.zip({
        "pt": parent.pt, "eta": parent.eta, "phi": parent.phi, "mass": parent.mass,
        "dR": pairs.j1.deltaR(pairs.j2),
        **{f"j1_{f}": pairs.j1[f] for f in pairs.j1.fields},
        **{f"j2_{f}": pairs.j2[f] for f in pairs.j2.fields},
    }, behavior=vector.behavior, with_name="Momentum4D")


def pairwise_max(jets, metric):
    """Return the maximum metric over every unique jet pair in each event."""
    pairs = ak.combinations(jets, 2, axis=1, fields=["j1", "j2"])
    values = metric(pairs.j1, pairs.j2)
    return ak.fill_none(ak.max(values, axis=1), 0)


# ------------------------------------------------------------------ main
def main(unified_unpadded, save_path):
    n = len(unified_unpadded)
    unified = ak.pad_none(unified_unpadded, NJ, clip=True)
    pairs = ak.pad_none(make_pairs(unified_unpadded), NJ, clip=True)

    # In this file there is a single jet collection, so sc4 / ng / unified all point to it
    jets = unified
    ngjets = unified

    d = {}
    # invariant masses
    d["mjj"] = ak.fill_none(inv_mass(jets[:, 0], jets[:, 1]), 0)
    d["max_mjj"] = pairwise_max(unified_unpadded, inv_mass)
    d["max_dEta"] = pairwise_max(unified_unpadded, dEta)
    d["delta_eta"] = d["max_dEta"]
    for k in ["mee", "mmm", "mpp", "mtt", "met"]:       # not available in this file
        d[k] = np.zeros(n, dtype=np.float32)

    # event features
    j = unified_unpadded
    d["jet_ht"] = ak.sum(j.pt, axis=1)
    d["central_jet_ht"] = ak.sum(j[(abs(j.eta) < 2.4) & (j.pt > 30)].pt, axis=1)
    d["n_jets"] = ak.num(j, axis=1)
    d["n_ngjets"] = ak.num(j, axis=1)
    for k in ["n_genjets", "n_sc8jets", "n_electrons", "n_muons", "n_photons", "n_taus"]:
        d[k] = np.zeros(n, dtype=np.int64)               # not available in this file

    # per-object arrays
    leading_phi = ak.fill_none(jets[:, 0].phi, 0)
    score_names = list(SCORES)
    d.update(collection_field_arrays(jets, "sc4jets", leading_phi, ["pt", "eta", "phi", "btagScore"]))
    d.update(collection_field_arrays(ngjets, "ngjets", leading_phi, ["pt", "eta", "phi", *score_names]))
    d.update(collection_field_arrays(unified, "unifiedjets", leading_phi, ["pt", "eta", "phi", *score_names]))
    uf = [f for f in unified_unpadded.fields if f != "btagScore"]   # pt, eta, phi, mass, scores
    d.update(collection_field_arrays(pairs, "jetpairs", leading_phi,
             ["pt", "eta", "phi", "mass", "dR"] + [f"j1_{f}" for f in uf] + [f"j2_{f}" for f in uf]))

    # collections not present in this file: zero-filled with the original shapes
    zeros = lambda cols, size, p: {f"{p}_{c}": np.zeros((n, size), dtype=np.float32) for c in cols}
    d.update(zeros(["pt", "eta", "phi", "partonFlavour"], NJ, "genjets"))
    d.update(zeros(["pt", "eta", "phi", "mass"], NJ, "sc8jets"))
    d.update(zeros(["pt", "eta", "hwIso", "hwQual"], NL, "electron"))
    d.update(zeros(["pt", "eta", "hwQual"], NL, "muon"))
    d.update(zeros(["pt", "eta", "hwIso", "hwQual"], NL, "photon"))
    d.update(zeros(["pt", "eta"], NL, "tau"))

    d = {k: np.asarray(v) for k, v in d.items()}
    os.makedirs(save_path, exist_ok=True)
    np.savez(os.path.join(save_path, "data.npz"), **d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="root file, glob, or directory of root files")
    ap.add_argument("--out", required=True, help="output directory for data.npz")
    args = ap.parse_args()
    main(load_events(args.input), args.out)
