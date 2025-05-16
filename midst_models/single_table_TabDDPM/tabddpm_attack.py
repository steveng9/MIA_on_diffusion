
import sys
import os
import json
import pandas as pd
import numpy as np
import pickle

from numpy import mean
from scipy import stats
from argparse import Namespace
from os import listdir
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.profiler

from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA


import category_encoders
from complex_pipeline import (
    clava_clustering,
    clava_training,
    clava_load_pretrained,
    clava_synthesizing,
    load_configs,
)
from pipeline_modules import load_multi_table



import warnings
warnings.filterwarnings("ignore")




verbose = False



# python vae_diffusion_attack.py action=synth_diff model=1 SV=4000 SD=10000 save_score=false

acceptable_args = {
    "action": str,
    "model": int,
    "all_models": str,
    "save_score": bool,
    "AD": int,  # Number of Denoiser epochs on auxiliary data
    "SD": int,  # Number of Denoiser epochs on synthetic data
}

def parse_args():
    parsed_args = {}
    for arg in sys.argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)  # Split only on the first '='
            try:
                parsed_args[key] = acceptable_args[key](value)
            except KeyError:
                raise KeyError(f"Invalid argument argument: {key}. \nAvailable arguments: {acceptable_args}")
        else:
            raise ValueError(f"Invalid argument format: {arg}. Expected 'keyword=value'.")
    return Namespace(**parsed_args)



def main():
    if torch.cuda.is_available():print("Using CUDA device!")
    else: print("NOT Using CUDA!")

    args = parse_args()

    if args.action == 'list_models': list_all_pretrained_models(args)
    # elif args.action == 'sanity_diff': attack_diffusion_sanity_check(args)
    # elif args.action == 'train_several': train_several(args)
    # elif args.action == 'attack_several': attack_several(args)
    # elif args.action == 'sanity_several': sanitycheck_several(args)
    # elif args.action == 'pca': PCA_on_encoded_challenge_points(args)
    # elif args.action == 'pca_several': pca_several(args)
    #
    # # the three main stages of the attack
    elif args.action == 'aux_diff': train_aux_diffusion(args)
    elif args.action == 'synth_diff': train_synth_diffusion(args)
    elif args.action == 'attack_diff': attack_diffusion(args)

    else: raise ValueError(f"Invalid action: {args.action}")



def list_all_pretrained_models(args):
    aux_models_path = "attack_artifacts/models/modelAUX/tabddpmA/"

    print("\nAuxiliary Diffusion models:")
    for f in [f for f in listdir(aux_models_path) if f.find("diffus") != -1]:
        print("\t* " + f)

    print("\nSynthetic Diffusion models:")
    for i in range(30):
        synth_models_path = f"attack_artifacts/models/model{i+1}/tabddpmS/"
        try: synth_denoisers = [f for f in listdir(synth_models_path) if f.find("diffus") != -1]
        except FileNotFoundError: synth_denoisers = []
        if len(synth_denoisers) != 0:
            print(f"\tmodel {i+1}")
            for synth_denoiser in synth_denoisers:
                print(f"\t\t* {synth_denoiser}")






def train_aux_diffusion(args):
    num_epochs_AD = args.AD
    print(f"\nTraining auxiliary Denoiser with {num_epochs_AD} epochs!!!\n\n")
    ATTACK_ARTIFACTS = "attack_artifacts/"
    MODEL_PATH_A = ATTACK_ARTIFACTS + f"models/modelAUX/tabddpmA"
    DATA_DIR = f"../../data/auxiliary_inferred/"

    tabddpm_aux_models = train_diffusion_for_attack(num_epochs_AD, MODEL_PATH_A, DATA_DIR, "trans_aux")
    dump_artifact(tabddpm_aux_models, MODEL_PATH_A + f"/tabddpm_diffus_aux_AD{num_epochs_AD}")


def train_synth_diffusion(args):
    model_num = args.model
    num_epochs_SD = args.SD
    print(f"\nTraining Synthetic Denoiser, model: {model_num}, with {num_epochs_SD} epochs!!!\n\n")
    ATTACK_ARTIFACTS = "attack_artifacts/"
    MODEL_PATH_S = ATTACK_ARTIFACTS + f"models/model{model_num}/tabddpmS"
    DATA_DIR = f"../../data/tabddpm_black_box/train/tabddpm_{model_num}/"

    tabddpm_aux_models = train_diffusion_for_attack(num_epochs_SD, MODEL_PATH_S, DATA_DIR, "train_with_id")
    dump_artifact(tabddpm_aux_models, MODEL_PATH_S + f"/tabddpm_diffus_synth_SD{num_epochs_SD}_m{model_num}")


def attack_diffusion(args):
    model_num = args.model
    num_epochs_AD = args.AD
    num_epochs_SD = args.SD

    threat_model = "black_box"
    ATTACK_ARTIFACTS = "attack_artifacts/"
    DATA_DIR_CHALLENGE = ATTACK_ARTIFACTS + f"data/model{model_num}/data_challenge/"
    MODEL_PATH_S = ATTACK_ARTIFACTS + f"models/model{model_num}/tabddpmS"
    MODEL_PATH_A = ATTACK_ARTIFACTS + f"models/modelAUX/tabddpmA"
    MODEL_PATH_C = ATTACK_ARTIFACTS + f"models/model{model_num}/tabddpmC"
    LOSS_RESULTS = ATTACK_ARTIFACTS + f"loss_results/model{model_num}/"
    os.makedirs(LOSS_RESULTS, exist_ok=True)
    DATA_DIR = f"../../data/tabddpm_black_box/train/tabddpm_{model_num}/"

    if not os.path.exists(DATA_DIR + "challenge_no_id.csv"):
        challenge = pd.read_csv(f"../../data/tabddpm_{threat_model}/train/tabddpm_{model_num}/challenge_with_id.csv", header="infer")
        challenge.drop(columns=["trans_id", "account_id"], inplace=True)
        challenge.to_csv(DATA_DIR + "challenge_no_id.csv", index=False)

    config_path = "configs/trans.json"
    configs, _ = load_configs(config_path, MODEL_PATH_S)

    tables_c, relation_order, dataset_meta = load_multi_table(DATA_DIR, metadata_dir="configs/", dataset_name="challenge_no_id")
    tables_c, all_group_lengths_prob_dicts = clava_clustering(tables_c, relation_order, MODEL_PATH_C, configs)

    tabddpm_aux = load_artifact(MODEL_PATH_A + f"/tabddpm_diffus_aux_AD{num_epochs_AD}")
    tabddpm_synth = load_artifact(MODEL_PATH_S + f"/tabddpm_diffus_synth_SD{num_epochs_SD}_m{model_num}")
    losses_s, predictions_s = attack_challenge_points(tabddpm_synth, tables_c, MODEL_PATH_S)
    losses_a, predictions_a = attack_challenge_points(tabddpm_aux, tables_c, MODEL_PATH_A)




    # 20,
    # 121,
    # 199,
    # 205-forward,
    # 260-noise pred netowrk ,
    # 351-myu tita,
    # 338- mse

    score_diffusion_attack(model_num, losses_s, losses_a, predictions_s, predictions_a)




################################################################
##########         HELPER FUNCTIONS          ###################
################################################################


def train_diffusion_for_attack(num_epochs, model_path, data_dir, dataset_name):
    if not os.path.exists(f"{data_dir}/train_no_id.csv"):
        df = pd.read_csv(f"{data_dir}/{dataset_name}.csv", header="infer")
        df.drop(columns=["trans_id", "account_id"], inplace=True)
        df.to_csv(f"{data_dir}/train_no_id.csv", index=False)

    config_path = "configs/trans.json"
    configs, _ = load_configs(config_path, model_path)

    configs["diffusion"]["iterations"] = num_epochs
    configs["classifier"]["iterations"] = num_epochs

    tables, relation_order, dataset_meta = load_multi_table(data_dir, metadata_dir="configs/", dataset_name="train_no_id")
    tables, all_group_lengths_prob_dicts = clava_clustering(tables, relation_order, model_path, configs)

    models = clava_training(tables, relation_order, model_path, configs)
    return models


def attack_challenge_points(tabddpm_models, tables_c, model_path):
    tabddpm_models[(None, 'trans')]['diffusion'].attack(tables_c)
    return 1, 1


def show_pca_reduction(X, y):
    pca = PCA(n_components=4)
    principal_components = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    for target, color in [(0, 'r'), (1, 'b')]:
        indices = np.array(y) == target
        plt.scatter(principal_components[indices, 0], principal_components[indices, 1], label=target, c=color)

    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('2D PCA of Iris Dataset')
    plt.legend()
    plt.show()



def score_diffusion_attack(model_num, losses_s, losses_a, predictions_s, predictions_a):
    threat_model = "black_box"

    membership = pd.read_csv(f"../../data/tabddpm_{threat_model}/train/tabddpm_{model_num}/challenge_label.csv", header="infer")
    membership = membership['is_train'].tolist()

    all_zeta_loss = []
    all_zeta_predictions = []
    tpr_ls = []
    auc_ls = []
    tpr_ps = []
    auc_ps = []
    for t in range(len(losses_s)):
        zeta_loss = losses_s[t].mean(1) / losses_a[t].mean(1)
        zeta_predictions = predictions_s[t] / predictions_a[t]

        # activated_zeta_loss = (1 - activate_3(np.array(zeta_loss.mean(1))))
        # activated_zeta_predictions = (1 - activate_3(np.array(zeta_predictions.mean(1))))
        activated_zeta_loss = (1 - activate_3(np.array(zeta_loss)))
        activated_zeta_predictions = (1 - activate_3(np.array(zeta_predictions)))

        tpr_l, auc_l = get_tpr_at_fpr(membership, activated_zeta_loss)
        tpr_p, auc_p = get_tpr_at_fpr(membership, activated_zeta_predictions)
        # print(f"{t+1}: \t{tpr_l:.4f}\t{auc_l:.4f}\t\t{tpr_p:.4f}\t{auc_p:.4f}")
        all_zeta_loss.append(activated_zeta_loss)
        all_zeta_predictions.append(activated_zeta_predictions)
        tpr_ls.append(tpr_l)
        auc_ls.append(auc_l)
        tpr_ps.append(tpr_p)
        auc_ps.append(auc_p)

    tpr_allT_l, auc_allT_l = get_tpr_at_fpr(membership, np.array(all_zeta_loss).mean(0))
    tpr_allT_p, auc_allT_p = get_tpr_at_fpr(membership, np.array(all_zeta_predictions).mean(0))

    # print()
    print(f"model: {model_num},", '\t{0:.2f}'.format(tpr_allT_l), '\t{0:.2f}'.format(auc_allT_l), '\t{0:.2f}'.format(tpr_allT_p), '\t{0:.2f}'.format(auc_allT_p))
    # print(f"{mean(tpr_ls)}\t{mean(auc_ls)}\t{mean(tpr_ps)}\t{mean(auc_ps)}")


def get_tpr_at_fpr(y_true, y_pred):
    desired_fpr = 0.1
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    tpr_at_desired_fpr = np.interp(desired_fpr, fpr, tpr)
    auc = roc_auc_score(y_true, y_pred)
    return tpr_at_desired_fpr, auc


def dump_artifact(artifact, name):
    pickle_file = open(name, 'wb')
    pickle.dump(artifact, pickle_file)
    pickle_file.close()

def load_artifact(name):
    pickle_file = open(name, 'rb')
    artifact = pickle.load(pickle_file)
    pickle_file.close()
    return artifact

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



if __name__ == '__main__':
    main()
