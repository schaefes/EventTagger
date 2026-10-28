import matplotlib.pyplot as plt
import mplhep as hep

colours = ["black", "red", "orange", "green", "blue"]
LINESTYLES = [
    "-",
    "--",
    "dotted",
    (0, (3, 5, 1, 5)),
    (
        0,
        (
            3,
            5,
            1,
            1,
            1,
            5,
        ),
    ),
    (0, (3, 10, 1, 10)),
    (0, (3, 10, 1, 10, 1, 10)),
]

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

SMALL_SIZE = 25
MEDIUM_SIZE = 28
BIGGER_SIZE = 35

LEGEND_WIDTH = 20
LINEWIDTH = 5
MARKERSIZE = 20

FIGURE_SIZE = (17, 17)

CMSHEADER_LEFT = "Phase 2 Simulation Preliminary"
CMSHEADER_RIGHT = "PU 200 (14 TeV)"
CMSHEADER_SIZE = BIGGER_SIZE

CLASS_LABEL_STYLE = {
    'b': 'b',
    'charm': 'c',
    'light': 'light',
    'gluon': 'gluon',
    'taum': '$\\tau_{h}^{-}$',
    'taup': '$\\tau_{h}^{+}$',
    'electron': 'Electron',
    'muon': 'Muon',
    'inclusive': 'Inclusive',
    'Regression': 'Regression',
    "taus": "Taus",
    "jets": "Jets (b, c, light, gluon)",
    "leptons": "Leptons (muon, electron)",
}

PROCESS_STYLE = {
        'GGFHHTo4B': r'GGF $\rightarrow$ HH $\rightarrow$ 4b',
        'GGFHHTo2B2Tau': r'GGF $\rightarrow$ HH $\rightarrow$ b$\bar{b}$t$\bar{t}$',
        'VBFHtt': r'VBF $\rightarrow$ H $\rightarrow$ t$\bar{t}$',
        'GGFHHTo2B2Tau': r'GGF $\rightarrow$ HH $\rightarrow$ b$\bar{b}$t$\bar{t}$',
        'MinBias': 'MinBias',
        'VBFHToBB': r"$VBF H \to b\bar{b}$",
        'VBFHToCC': r"$VBF H \to c\bar{c}$",
        'VBFHToInvisible': r"$VBF \to invisible$",
        'VBF': r"$VBF \to H (b\bar{b}, c\bar{c}, \tau^+\tau^-)$",
        'GGF': r"$GGF \to HH (4b, 2b2\tau)$",
}

SCORE_LABEL_STYLE = {
    'ovo_score': 'OvO',
    'total_signal': r'$\Sigma$ sig',
    'raw': 'Raw',
}

def set_style():
    # Setup plotting to CMS style
    hep.cms.label()
    hep.cms.text("Simulation")
    plt.style.use(hep.style.CMS)

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE + 5)  # fontsize of the x and y labels
    plt.rc('axes', linewidth=LINEWIDTH + 2)  # thickness of axes
    plt.rc('xtick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE - 2)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # line thickness
    import matplotlib as mpl

    mpl.rcParams['lines.linewidth'] = 5

    import matplotlib

    matplotlib.rcParams['xtick.major.size'] = 20
    matplotlib.rcParams['xtick.major.width'] = 5
    matplotlib.rcParams['xtick.minor.size'] = 10
    matplotlib.rcParams['xtick.minor.width'] = 4

    matplotlib.rcParams['ytick.major.size'] = 20
    matplotlib.rcParams['ytick.major.width'] = 5
    matplotlib.rcParams['ytick.minor.size'] = 10
    matplotlib.rcParams['ytick.minor.width'] = 4
