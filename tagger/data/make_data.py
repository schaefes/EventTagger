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

def main(data, n_jets_cut, n_taus_cut, save_path):
    # jet collections
    jets_unpadded = ak.Array(data["L1puppiExtJetSC4"], behavior=vector.behavior, with_name="PtEtaPhiMLorentzVector")
    genjets_unpadded = ak.Array(data["GenJet"], behavior=vector.behavior, with_name="PtEtaPhiMLorentzVector")
    ngjets_unpadded = ak.Array(data["L1puppiJetSC4NG"], behavior=vector.behavior, with_name="PtEtaPhiMLorentzVector")
    sc8_jets_unpadded = ak.Array(data["L1puppiJetSC8"], behavior=vector.behavior, with_name="PtEtaPhiMLorentzVector")
    taus_unpadded = ak.Array(data["L1nnPuppiTau"], behavior=vector.behavior, with_name="PtEtaPhiMLorentzVector")
    electrons_unpadded = ak.Array(data["L1GTtkElectron"], behavior=vector.behavior, with_name="PtEtaPhiMLorentzVector")
    muons_unpadded = ak.Array(data["L1GTgmtTkMuon"], behavior=vector.behavior, with_name="PtEtaPhiMLorentzVector")
    photons_unpadded = ak.Array(data["L1GTtkPhoton"], behavior=vector.behavior, with_name="PtEtaPhiMLorentzVector")

    # pad objects
    jets = ak.pad_none(jets_unpadded, 16, clip=True)
    ngjets = ak.pad_none(ngjets_unpadded, 16, clip=True)
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
    main(data, 1, 0, save_path)

