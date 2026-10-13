import numpy as np
import awkward as ak
import uproot
import os
import coffea
from coffea.nanoevents.methods import vector
import argparse
from processes import PROC_PATHS, DATA_PATH, DATASETS

# load signal data
def load_data(path, branches):
    events = None
    for f in os.listdir(path):
        data = uproot.open(os.path.join(path, f))
        if events is not None:
            events_2 = data["Events"].arrays(filter_name = branches, how = "zip")
            events = ak.concatenate((events, events_2))
        else:
            events = data["Events"].arrays(filter_name = branches, how = "zip")
    return events

def inv_mass(jet1, jet2):
    m = np.sqrt(jet1.pt * jet2.pt * (np.cosh(abs(jet1.eta - jet2.eta)) - np.cos(abs(jet1.phi - jet2.phi))))
    return m

def dEta(jet1, jet2):
    return abs(jet1.eta - jet2.eta)

def return_jet_field(collection, leading_phi, f):
    # project relative to first phi value in input
    if f == "phi":
        phi = ak.fill_none(getattr(collection, f), 0)          # fill None -> 0 while still an ak.Array
        projected_phi = ak.to_numpy(phi) - ak.to_numpy(leading_phi)[:, np.newaxis]
        projected_phi = np.mod(projected_phi + np.pi, 2 * np.pi) - np.pi
        return projected_phi
    else:
        return ak.to_numpy(ak.fill_none(getattr(collection, f), 0))

# Helper to build all field arrays for one collection in one go
def collection_field_arrays(collection, collection_name, leading_phi, fields):
    return {f"{collection_name}_{k}": return_jet_field(collection, leading_phi, k) for k in fields}

def unified_jets(ng, sc):
    # unify ngjets and scjets into one collection
    # central jets and raw pt > 15 GeV: use ng rest is scjets with jecs
    # All (scjet, ngjet) pairs within each event: shape (events, n_sc, n_ng)
    pairs = ak.cartesian([sc, ng], axis=1, nested=True)
    sc_p, ng_p = ak.unzip(pairs)

    # Exact match is fine if eta/phi are bit-identical; otherwise use a small dR tolerance
    match = (sc_p.eta == ng_p.eta) & (sc_p.phi == ng_p.phi)
    # dphi = (sc_p.phi - ng_p.phi + np.pi) % (2*np.pi) - np.pi
    # match = (sc_p.eta - ng_p.eta)**2 + dphi**2 < 1e-6

    # Sanity checks: every ngjet matches exactly one scjet, and no scjet matches twice
    assert ak.all(ak.sum(match, axis=1) == 1)
    assert ak.all(ak.sum(match, axis=2) <= 1)

    # ngjet pt for each scjet (None where there is no match)
    ng_pt_matched = ak.firsts(ng_p.pt[match], axis=2)

    # Use ngjets pt where a match exists and the jet is central, else scjets pt
    central = abs(sc.eta) < 2.4
    use_ng = central & ~ak.is_none(ng_pt_matched, axis=1)
    unified_pt = ak.where(use_ng, ak.fill_none(ng_pt_matched, 0.0), sc.pt)

    # handle tagging scores
    score_fields = [f for f in ng.fields if "Score" in f]

    # Same as for pt, but for every score: None where the scjet has no ngjet match
    scores = {
        f: ak.fill_none(ak.firsts(ng_p[f][match], axis=2), 0.0)
        for f in score_fields
    }

    unified = ak.zip({
        "pt": unified_pt,
        "eta": sc.eta,
        "phi": sc.phi,
        "mass": sc.mass,
        **scores,
    },
    behavior=vector.behavior,
    with_name="Momentum4D")

    # Sort last, after all fields are attached, so everything is reordered together
    unified = unified[ak.argsort(unified.pt, axis=1, ascending=False)]

    return unified

def make_pairs(jets):
    # make all pairs of jets in each event
    # use only the first pt leading jets
    # All unique pairs of jets within each event (n choose 2, no duplicates, no self-pairs)
    jets = jets[:, :4]  # take only the first 4 leading pt jets
    pairs = ak.combinations(jets, 2, axis=1, fields=["j1", "j2"])

    # Reconstructed parent
    parent = pairs.j1 + pairs.j2
    score_fields = [f for f in jets.fields if "TagScore" in f]
    parent = ak.zip({
        "pt": parent.pt,
        "eta": parent.eta,
        "phi": parent.phi,
        "mass": parent.mass,
        "dR": pairs.j1.deltaR(pairs.j2),
        **{f"j1_{f}": pairs.j1[f] for f in score_fields},
        **{f"j2_{f}": pairs.j2[f] for f in score_fields},
    },
    behavior=vector.behavior,
    with_name="Momentum4D")

    return parent


def main(data, save_path):
    # jet collections
    jets_unpadded = ak.Array(data["L1puppiExtJetSC4"], behavior=vector.behavior, with_name="PtEtaPhiMLorentzVector")
    genjets_unpadded = ak.Array(data["GenJet"], behavior=vector.behavior, with_name="PtEtaPhiMLorentzVector")
    ngjets_unpadded = ak.Array(data["L1puppiJetSC4NG"], behavior=vector.behavior, with_name="PtEtaPhiMLorentzVector")
    unified_unpadded = unified_jets(ngjets_unpadded, jets_unpadded)
    sc8_jets_unpadded = ak.Array(data["L1puppiJetSC8"], behavior=vector.behavior, with_name="PtEtaPhiMLorentzVector")
    taus_unpadded = ak.Array(data["L1nnPuppiTau"], behavior=vector.behavior, with_name="PtEtaPhiMLorentzVector")
    electrons_unpadded = ak.Array(data["L1GTtkElectron"], behavior=vector.behavior, with_name="PtEtaPhiMLorentzVector")
    muons_unpadded = ak.Array(data["L1GTgmtTkMuon"], behavior=vector.behavior, with_name="PtEtaPhiMLorentzVector")
    photons_unpadded = ak.Array(data["L1GTtkPhoton"], behavior=vector.behavior, with_name="PtEtaPhiMLorentzVector")

    # pad objects
    jets = ak.pad_none(jets_unpadded, 16, clip=True)
    ngjets = ak.pad_none(ngjets_unpadded, 16, clip=True)
    unified = ak.pad_none(unified_unpadded, 16, clip=True)
    pairs = ak.pad_none(make_pairs(unified), 16, clip=True)
    genjets = ak.pad_none(genjets_unpadded, 16, clip=True)
    sc8_jets = ak.pad_none(sc8_jets_unpadded, 16, clip=True)
    taus = ak.pad_none(taus_unpadded, 5, clip=True)
    electrons = ak.pad_none(electrons_unpadded, 5, clip=True)
    muons = ak.pad_none(muons_unpadded, 5, clip=True)
    photons = ak.pad_none(photons_unpadded, 5, clip=True)

    # Invariant masses
    save_dict = {}
    save_dict['mjj'] = ak.fill_none(inv_mass(jets[:,0], jets[:,1]), 0)
    save_dict['max_mjj'] = ak.fill_none(ak.max(ak.max(jets.metric_table(jets, axis=1, metric=inv_mass), axis=1), axis=1), 0)
    save_dict['max_dEta'] = ak.fill_none(ak.max(ak.max(jets.metric_table(jets, axis=1, metric=dEta), axis=1), axis=1), 0)
    save_dict['mee'] = ak.fill_none(inv_mass(electrons[:, 0], electrons[:,1]), 0)
    save_dict['mmm'] = ak.fill_none(inv_mass(muons[:,0], muons[:,1]), 0)
    save_dict['mpp'] = ak.fill_none(inv_mass(photons[:, 0], photons[:, 1]), 0)
    save_dict['mtt'] = ak.fill_none(inv_mass(taus[:, 0], taus[:, 1]), 0)
    save_dict['met'] = data["L1puppiMET_et"]

    # Other event features
    save_dict['delta_eta'] = ak.fill_none(ak.max(ak.max(jets.metric_table(jets, axis=1, metric=dEta), axis=1), axis=1), 0)
    save_dict['jet_ht'] = ak.sum(jets_unpadded.pt, axis=1)
    save_dict['central_jet_ht'] = ak.sum(jets_unpadded[(abs(jets_unpadded.eta) < 2.4) & (jets_unpadded.pt > 30)].pt, axis=1)
    save_dict['n_jets'] = ak.num(jets_unpadded, axis=1)
    save_dict['n_ngjets'] = ak.num(ngjets_unpadded, axis=1)
    save_dict['n_genjets'] = ak.num(genjets_unpadded, axis=1)
    save_dict['n_sc8jets'] = ak.num(sc8_jets_unpadded, axis=1)
    save_dict['n_electrons'] = ak.num(electrons_unpadded, axis=1)
    save_dict['n_muons'] = ak.num(muons_unpadded, axis=1)
    save_dict['n_photons'] = ak.num(photons_unpadded, axis=1)
    save_dict['n_taus'] = ak.num(taus_unpadded, axis=1)

    leading_phi = ak.fill_none(jets[:, 0].phi, 0)
    save_dict.update(collection_field_arrays(jets, "sc4jets", leading_phi, ["pt", "eta", "phi", "btagScore"]))
    save_dict.update(collection_field_arrays(ngjets, "ngjets", leading_phi, ["pt", "eta", "phi", "udsTagScore", "bTagScore",
        "cTagScore", "gTagScore", "tau_nTagScore", "tau_pTagScore", "eTagScore", "muTagScore"]))
    save_dict.update(collection_field_arrays(unified, "unifiedjets", leading_phi, ["pt", "eta", "phi", "udsTagScore", "bTagScore",
        "cTagScore", "gTagScore", "tau_nTagScore", "tau_pTagScore", "eTagScore", "muTagScore"]))
    save_dict.update(collection_field_arrays(pairs, "jetpairs", leading_phi,
        ["pt", "eta", "phi", "mass", "dR"] + [f"j1_{f}" for f in unified.fields if "TagScore" in f] + [f"j2_{f}" for f in unified.fields if "TagScore" in f]))
    save_dict.update(collection_field_arrays(genjets, "genjets", leading_phi, ["pt", "eta", "phi", "partonFlavour"]))
    save_dict.update(collection_field_arrays(sc8_jets, "sc8jets", leading_phi, ["pt", "eta", "phi", "mass"]))
    save_dict.update(collection_field_arrays(electrons, "electron", leading_phi, ["pt", "eta", "hwIso", "hwQual"]))
    save_dict.update(collection_field_arrays(muons, "muon", leading_phi, ["pt", "eta", "hwQual"]))
    save_dict.update(collection_field_arrays(photons, "photon", leading_phi, ["pt", "eta", "hwIso", "hwQual"]))
    save_dict.update(collection_field_arrays(taus, "tau", leading_phi, ["pt", "eta"]))

    # Save data
    np.savez(os.path.join(save_path, f"data.npz"), **save_dict)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal", type=str)
    args = parser.parse_args()
    save_path = DATASETS[args.signal][0]
    os.makedirs(save_path, exist_ok=True)

    branches = [
                "/(L1puppiExtJetSC4)_(pt|eta|phi|mass|btagScore)/",
                "/(L1puppiJetSC8)_(pt|eta|phi|mass)/",
                "/(L1puppiJetSC4NG)_(pt|eta|phi|udsTagScore|bTagScore|cTagScore|gTagScore|tau_nTagScore|tau_pTagScore|eTagScore|muTagScore)/",
                "/(GenJet)_(pt|eta|phi|partonFlavour)/",
                "/(L1GTtkElectron)_(pt|eta|phi|hwIso|hwQual)/",
                "/(L1GTgmtTkMuon)_(pt|eta|phi|hwQual)/",
                "/(L1GTtkPhoton)_(pt|eta|phi|hwIso|hwQual)/",
                "/(L1nnPuppiTau)_(pt|eta|phi)/",
                "L1puppiMET_pt", "L1puppiMET_et"
                ]
    data = load_data(PROC_PATHS[args.signal], branches)
    main(data, save_path)

