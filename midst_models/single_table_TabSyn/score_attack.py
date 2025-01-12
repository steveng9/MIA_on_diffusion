from sklearn.metrics import roc_curve, roc_auc_score
import pickle
import sys
import numpy as np
import pandas as pd
from scipy import stats

def activate_1(p_rel, confidence=1, center=True) -> np.ndarray:
    logs = np.log(p_rel)
    median = np.median(logs) if center else 0
    probabilities = 1 / (1 + np.exp(-1 * confidence * (logs - median)))
    return probabilities

def activate_2(p_rel, confidence=1, center=True) -> np.ndarray:
    zscores = stats.zscore(p_rel)
    median = np.median(zscores) if center else 0
    probabilities = 1 / (1 + np.exp(-1 * confidence * (zscores - median)))
    return probabilities

def activate_3(p_rel, confidence=1, center=True) -> np.ndarray:
    logs = np.log(p_rel)
    zscores = stats.zscore(logs)
    median = np.median(zscores) if center else 0
    probabilities = 1 / (1 + np.exp(-1 * confidence * (zscores - median)))
    return probabilities

def activate_4(p_rel, confidence=1, center=True) -> np.ndarray:
    median = np.median(p_rel) if center else 0
    probabilities = 1 / (1 + np.exp(-1 * confidence * (p_rel - median)))
    return probabilities

threat_model = "black_box"
model_num = sys.argv[1]

with open(f"attack_artifacts/loss_results/model{model_num}/synth_losses_{model_num}.pkl", "rb") as f:
    synth_losses = pickle.load(f)
    challenge_mse_lossS, challenge_ce_lossS, challenge_kl_lossS, challenge_accS = synth_losses
with open(f"attack_artifacts/loss_results/model{model_num}/aux_losses_{model_num}.pkl", "rb") as f:
    aux_losses = pickle.load(f)
    challenge_mse_lossA, challenge_ce_lossA, challenge_kl_lossA, challenge_accA = aux_losses


# challenge = pd.read_csv(f"../../data/tabsyn_{threat_model}/train/tabsyn_{model_num}/challenge_with_id.csv", header="infer")
membership = pd.read_csv(f"../../data/tabsyn_{threat_model}/train/tabsyn_{model_num}/challenge_label.csv", header="infer")
membership = membership['is_train'].tolist()

ratio_mse = np.array(challenge_mse_lossS) / np.where(np.array(challenge_mse_lossA) < 0.000000001, 0.000000001, np.array(challenge_mse_lossA))
ratio_kl = np.array(challenge_kl_lossS) / np.where(np.array(challenge_kl_lossA) < 0.000000001, 0.000000001, np.array(challenge_kl_lossA))
ratio_ce = np.array(challenge_ce_lossS) / np.where(np.array(challenge_ce_lossA) < 0.000000001, 0.000000001, np.array(challenge_ce_lossA))
ratio_acc = np.array(challenge_accS) / np.where(np.array(challenge_accA) < 0.000000001, 0.000000001, np.array(challenge_accA))


prediction_mse = 1 - np.array(activate_3(ratio_mse))
prediction_ce = 1 - np.array(activate_3(ratio_ce))
ens_prediction = (prediction_mse + prediction_ce) / 2
prediction_kl = 1 - np.array(activate_3(ratio_kl))
prediction_acc = 1 - np.array(activate_3(ratio_acc))


desired_fpr = 0.1
for pred in [prediction_mse, prediction_ce, ens_prediction, prediction_kl]: #, prediction_acc]:
    fpr, tpr, thresholds = roc_curve(membership, pred)
    tpr_at_desired_fpr = np.interp(desired_fpr, fpr, tpr)
    print(tpr_at_desired_fpr, roc_auc_score(membership, pred))

print()
