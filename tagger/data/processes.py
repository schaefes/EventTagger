import os

# raw ntuples data
DATA_PATH = "/eos/cms/store/group/dpg_trigger/comm_trigger/L1Trigger/stella/phase2/menu/ntuples/v45wTT_151pre5_basepuppi"
PROC_PATHS = {
    'MinBias': os.path.join(DATA_PATH, 'MinBias_TuneCP5_14TeV-pythia8/MinBias_Spring24_200PUALCA_V45_reL1wTT_151pre5/260710_101659/0000'),
    'VBFHToCC': os.path.join(DATA_PATH, 'VBFHToCC_M-125_TuneCP5_14TeV-powheg-pythia8/VBFHtoCC_Spring24_200PU_V45_reL1wTT_151pre5/260708_073723/0000'),
    'VBFHToBB': os.path.join(DATA_PATH, 'VBFHToBB_M-125_TuneCP5_14TeV-powheg-pythia8/VBFHtoBB_Spring24_200PU_V45_reL1wTT_151pre5/260710_101720/0000'),
    'VBFHToTauTau': os.path.join(DATA_PATH, 'VBF_HToTauTau_M-125_TuneCP5_14TeV-powheg-pythia8/VBFHtoTaus_Spring24_200PU_V45_reL1wTT_151pre5/260805_125914/0000'),
    'QCD': os.path.join(DATA_PATH, 'QCD_Pt-20ToInf_TuneCP5_14TeV-pythia8'),
    'GGFHHTo4B': os.path.join(DATA_PATH, 'GluGluToHHTo4B_node_SM_TuneCP5_14TeV-amcatnlo-pythia8/GluGluToHHTo4B_Spring24_PU200_V45_reL1wTT_151pre5/260805_125938/0000'),
    'GGFHHTo2B2Tau': os.path.join(DATA_PATH, 'GluGluToHHTo2B2Tau_node_SM_TuneCP5_14TeV-madgraph-pythia8/GluGluToHHTo2B2Tau_Spring24_PU200_V45_reL1wTT_151pre5/260805_125925/0000'),
}

# processed datasets
DATASETS = {
    "QCD": [f"/eos/user/s/stella/EventTagger/data/QCD"],
    "VBFHToBB": [f"/eos/user/s/stella/EventTagger/data/VBFHToBB"],
    "VBFHToCC": [f"/eos/user/s/stella/EventTagger/data/VBFHToCC"],
    "VBFHToTauTau": [f"/eos/user/s/stella/EventTagger/data/VBFHToTauTau"],
    "VBF": [f"/eos/user/s/stella/EventTagger/data/VBFHToBB",
            f"/eos/user/s/stella/EventTagger/data/VBFHToCC",
            f"/eos/user/s/stella/EventTagger/data/VBFHToTauTau"],
    "GGF": [f"/eos/user/s/stella/EventTagger/data/GGFHHTo4B",
            f"/eos/user/s/stella/EventTagger/data/GGFHHTo2B2Tau"],
    "MinBias": [f"/eos/user/s/stella/EventTagger/data/MinBias"],
    "GGFHHTo4B": [f"/eos/user/s/stella/EventTagger/data/GGFHHTo4B"],
    "GGFHHTo2B2Tau": [f"/eos/user/s/stella/EventTagger/data/GGFHHTo2B2Tau"],
    }

BACKGROUND_PROCESSES = ["QCD", "MinBias"]
