from data_processor import build_model_data
import numpy as np

'''
Culumn 0 (Id): unique id (int) of the sample
    NOT A FEATURE
Column 1 (Prediction): unique values ["b", "s"] -> converted to [0, 1] respectively
    converters={1: lambda x: 0 if chr(ord(x)) == "b" else 1}
    NOT A FEATURE
------------------------------------------------------------------------------------------------------------------
Column 2 (DER_mass_MMC): estimated mass m_H (float) of the Higgs boson candidate
    MAY BE UNDEFINED (-999)
Column 3 (DER_mass_transverse_met_lep): ransverse mass (float) between missing transverse energy and lepton
Column 4 (DER_mass_vis): invariant mass (float) of hadronic tau and lepton
Column 5 (DER_pt_h): modulus (float) of vector sum of transverse momentum of hadronic tau, lepton, and missing
    transverse energy vector
Column 6 (DER_deltaeta_jet_jet): absolute value (float) of pseudorapidity separation between two jets
    (UNDEFINED IF PRI_jet_num <= 1)
    MAY BE UNDEFINED (-999)
Column 7 (DER_mass_jet_jet): invariant mass (float) of two jets
    (UNDEFINED IF PRI_jet_num <= 1)
    MAY BE UNDEFINED (-999)
Column 8 (DER_prodeta_jet_jet): product (float) of pseudorapidities of two jets
    (UNDEFINED IF PRI_jet_num <= 1)
    MAY BE UNDEFINED (-999)
Column 9 (DER_deltar_tau_lep): R separation (float) between hadronic tau and lepton
Column 10 (DER_pt_tot): modulus of vector sum (float) of missing transverse momenta and transverse momenta of
    hadronic tau, lepton, leading jet (IF PRI_jet_num >= 1) and subleading jet (IF PRI_jet_num == 2)
    (BUT NOT OF ANY ADDITIONAL JETS)
Column 11 (DER_sum_pt): sum of the moduli (float) of transverse momenta of hadronic tau, lepton, leading jet
    (IF PRI_jet_num >= 1) and subleading jet (IF PRI_jet_num == 2) and other jets (IF PRI_jet_num == 3)
Column 12 (DER_pt_ratio_lep_tau): ratio of transverse momenta (float) of lepton and hadronic tau
Column 13 (DER_met_phi_centrality): centrality of azimuthal angle (float) of missing transverse energy vector w.r.t.
    hadronic tau and lepton
Column 14 (DER_lep_eta_centrality): centrality of pseudorapidity (float) of lepton w.r.t. two jets
    (UNDEFINED IF PRI_jet_num <= 1)
    MAY BE UNDEFINED (-999)
------------------------------------------------------------------------------------------------------------------
Column 15 (PRI_tau_pt): transverse momentum (float) of the hadronic tau
Column 16 (PRI_tau_eta): pseudorapidity (float) of hadronic tau
Column 17 (PRI_tau_phi): azimuth angle (float) of hadronic tau
Column 18 (PRI_lep_pt): transverse momentum (float) of lepton (electron or muon)
Column 19 (PRI_lep_eta): pseudorapidity (float) of lepton
Column 20 (PRI_lep_phi): azimuth angle (float) of lepton
Column 21 (PRI_met): missing transverse energy (float)
Column 22 (PRI_met_phi): azimuth angle (float) of missing transverse energy
Column 23 (PRI_met_sumet): total transverse energy (float) in detector
Column 24 (PRI_jet_num): number of jets (integer with value of 0, 1, 2 or 3; possible larger values have been
    capped at 3)
Column 25 (PRI_jet_leading_pt): transverse momentum (float) of leading jet (jet with largest transverse momentum)
    (UNDEFINED IF PRI_jet_num == 0)
    MAY BE UNDEFINED (-999)
Column 26 (PRI_jet_leading_eta): pseudorapidity (float) of leading jet
    (UNDEFINED IF PRI_jet_num == 0)
    MAY BE UNDEFINED (-999)
Column 27 (PRI_jet_leading_phi): azimuth angle (float) of leading jet
    (UNDEFINED IF PRI_jet_num == 0)
    MAY BE UNDEFINED (-999)
Column 28 (PRI_jet_subleading_pt): transverse momentum (float) of subleading jet (jet with second largest transverse momentum)
    (UNDEFINED IF PRI_jet_num <= 1)
    MAY BE UNDEFINED (-999)
Column 29 (PRI_jet_subleading_eta): pseudorapidity (float) of subleading jet
    (UNDEFINED IF PRI_jet_num <= 1)
    MAY BE UNDEFINED (-999)
Column 30 (PRI_jet_subleading_phi): azimuth angle (float) of subleading jet
    (UNDEFINED IF PRI_jet_num <= 1)
    MAY BE UNDEFINED (-999)
Column 31 (PRI_jet_all_pt): scalar sum (float) of the transverse momentum of all the jets of the events
'''

DATA_PATH_SAMPLE_SUBMISSION = "../data/sample-submission.csv"
DATA_PATH_TEST = "../data/test.csv"
DATA_PATH_TRAIN = "../data/train.csv"

# Function greatly inspired from the given file "proj1_helpers.py".
def load_data(data_path, sub_sample=False):
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)

    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    yb = np.ones(len(y))
    yb[np.where(y == "b")] = 0

    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    y, tx = build_model_data(yb, input_data)
    return y, tx, ids
