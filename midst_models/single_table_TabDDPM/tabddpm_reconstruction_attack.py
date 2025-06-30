ON_UW_SERVER = False

import sys
import os
import json
import pandas as pd
import numpy as np
import pickle
import math
from argparse import Namespace
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.profiler

sys.path.append("../../")
import category_encoders
from complex_pipeline import (
    clava_clustering,
    clava_training,
    clava_load_pretrained,
    clava_synthesizing,
    load_configs, clava_reconstructing,
)
from pipeline_modules import load_multi_table
import warnings
warnings.filterwarnings("ignore")


num_epochs = 200_000
resamples = 10
jump = lambda t: max(0, t-10)
# jump = lambda t: math.floor(t*.75)
num_epochs_classifier = 20_000

reconstruction = True
reconstruct_method_RePaint = True
verbose = False
# data_path = "/home/golobs/data/" if ON_UW_SERVER else "/Users/golobs/Documents/GradSchool/NIST-CRC-25/NIST_Red-Team_Problems1-24_v2/"
data_path = "/home/golobs/data/" if ON_UW_SERVER else "/Users/golobs/Documents/GradSchool/NIST-CRC-25/25_PracticeProblem/"
DATA_NAME = "25_Demo_MST_e10_25f"
# DATA_NAME = "25_Demo_CellSupression_25f"
# DATA_NAME = "7_MST_e10_25f_QID1"
# DATA_NAME = "19_CELL_SUPPRESSION_25f_QID1"


# QI 1
QI = ['F37', 'F41', 'F2', 'F17', 'F22', 'F32', 'F47']
HIDDEN = ['F23', 'F13', 'F11', 'F43', 'F36', 'F15', 'F33', 'F25', 'F18', 'F5', 'F30', 'F10', 'F12', 'F50', 'F3', 'F1', 'F9', 'F21']

# QI 2
# QI = ['F37', 'F41', 'F3', 'F13', 'F18', 'F23', 'F30']
# HIDDEN = ['F11', 'F43', 'F5', 'F36', 'F25', 'F47', 'F32', 'F15', 'F33', 'F17', 'F10', 'F12', 'F2', 'F1', 'F50', 'F22', 'F9', 'F21']

features_25 = ['F1', 'F2', 'F3', 'F5', 'F9', 'F10', 'F11', 'F12', 'F13', 'F15', 'F17', 'F18', 'F21', 'F22', 'F23', 'F25', 'F30', 'F32', 'F33', 'F36', 'F37', 'F41', 'F43', 'F47', 'F50']



def main_attack():
    if torch.cuda.is_available(): print("Using CUDA device :)")
    else: print("NOT Using CUDA!")
    data_names = [
        # "25_Demo_AIM_e1_25f",
        # "25_Demo_ARF_25f",
        # "25_Demo_CellSupression_25f",
        "25_Demo_MST_e10_25f",
        # "25_Demo_RANKSWAP_25f",
        # "25_Demo_Synthpop_25f",
        # "25_Demo_TVAE_25f",
    ]

    for data_name in data_names:
        print(f"\n\n\n\n\n")

        train_diffusion(data_name)
        if reconstruction:
            reconstruct_data(data_name)
        else:
            gen_synth_data()



def one_feature_at_a_time_attack():
    if torch.cuda.is_available(): print("Using CUDA device :)")
    else: print("NOT Using CUDA!")
    data_names = [
        # "25_Demo_AIM_e1_25f",
        # "25_Demo_ARF_25f",
        # "25_Demo_CellSupression_25f",
        # "25_Demo_MST_e10_25f",
        # "25_Demo_RANKSWAP_25f",
        "25_Demo_Synthpop_25f",
        # "25_Demo_TVAE_25f",
    ]

    for data_name in data_names:
        print(f"\n\n\n\n\n")
        print(data_name)
        scores = []
        # for hidden_feature in sorted(HIDDEN):
        for hidden_feature in sorted(['F11', 'F13', 'F23']):
            data_name_reduced = data_name + "_reduced"

            synth = pd.read_csv(data_path + data_name + "_Deid.csv")
            synth_with_only_one_hidden_features = synth[QI + [hidden_feature]]
            synth_with_only_one_hidden_features.to_csv(data_path + data_name_reduced + "_Deid.csv", index=False)
            train_diffusion(data_name_reduced, qi=QI, hidden_features=[hidden_feature])

            score = reconstruct_data(data_name_reduced, qi=QI, hidden_features=[hidden_feature])
            print("SCORE for ", hidden_feature, score)
            scores.append(score)
        print()
        print()
        print(scores)






def train_diffusion(data_name, qi=QI, hidden_features=HIDDEN):
    print(f"\nTraining TabDDPM with {num_epochs} epochs\n\n")
    ATTACK_ARTIFACTS = "attack_artifacts_nist_crc/"
    MODEL_PATH = ATTACK_ARTIFACTS + f"models/e{num_epochs}"
    targets_name = "25_Demo_25f_OriginalData" if "Demo" in data_name else data_name + "_AttackTargets"
    print(f"targets_name : {targets_name}")
    targets = pd.read_csv(data_path + targets_name + ".csv")
    partial_data = targets[qi]
    DATA_DIR = data_path
    if reconstruction:
        dataset_name = data_name + "_Deid"
    else:
        dataset_name = "25_Demo_25f_OriginalData_noID"
    # dataset_name = "refined_training_data"

    config_path = "configs_nist_crc/crc_data.json"
    configs, _ = load_configs(config_path, MODEL_PATH)

    configs["diffusion"]["iterations"] = num_epochs
    configs["classifier"]["iterations"] = num_epochs_classifier

    tables, relation_order, dataset_meta = load_multi_table(DATA_DIR, metadata_dir="configs_nist_crc/", dataset_name=dataset_name)
    tables, all_group_lengths_prob_dicts = clava_clustering(tables, relation_order, MODEL_PATH, configs)

    partial_data[hidden_features] = tables['crc_data']['df'][hidden_features] # NOTE: temporary measure to make dimensionality match training data
    column_order = tables['crc_data']['df'].drop(['placeholder'], axis=1).columns
    if 'target' in column_order:
        column_order = tables['crc_data']['df'].drop(['placeholder', 'target'], axis=1).columns

    partial_data = partial_data[column_order]
    known_features_mask = np.zeros((len(partial_data), 25))
    known_features_mask[:, [partial_data.columns.get_loc(col) for col in qi]] = 1

    # models = clava_training_for_reconstruction(tables, relation_order, MODEL_PATH, configs)
    if reconstruction:
        models = clava_training(tables, relation_order, MODEL_PATH, configs,
            for_reconstruction=not reconstruct_method_RePaint,
            partial_data=partial_data,
            known_features_mask=known_features_mask)
    else:
        models = clava_training(tables, relation_order, MODEL_PATH, configs,
             for_reconstruction=False)
    # models, tables, all_group_lengths_prob_dicts, dataset_meta, relation_order, configs = train_diffusion_for_attack(num_epochs, MODEL_PATH, DATA_DIR, dataset_name)

    os.makedirs(MODEL_PATH + f"/e{num_epochs}", exist_ok=True)
    dump_artifact(models, MODEL_PATH + f"/model")
    dump_artifact(tables, MODEL_PATH + f"/tables")
    dump_artifact(all_group_lengths_prob_dicts, MODEL_PATH + f"/all_group_lengths_prob_dicts")
    dump_artifact(dataset_meta, MODEL_PATH + f"/dataset_meta")
    dump_artifact(relation_order, MODEL_PATH + f"/relation_order")
    dump_artifact(configs, MODEL_PATH + f"/configs")


def reconstruct_data(data_name, qi=QI, hidden_features=HIDDEN):
    print(f"\nReconstructing TabDDPM \n\n")
    ATTACK_ARTIFACTS = "attack_artifacts_nist_crc/"
    MODEL_PATH = ATTACK_ARTIFACTS + f"models"
    targets_name = "25_Demo_25f_OriginalData" if "Demo" in data_name else data_name + "_AttackTargets"
    print(f"targets_name : {targets_name}")
    targets = pd.read_csv(data_path + targets_name + ".csv")
    partial_data = targets[qi]

    models = load_artifact(MODEL_PATH + f"/e{num_epochs}/model")
    tables = load_artifact(MODEL_PATH + f"/e{num_epochs}/tables")
    all_group_lengths_prob_dicts = load_artifact(MODEL_PATH + f"/e{num_epochs}/all_group_lengths_prob_dicts")
    dataset_meta = load_artifact(MODEL_PATH + f"/e{num_epochs}/dataset_meta")
    relation_order = load_artifact(MODEL_PATH + f"/e{num_epochs}/relation_order")
    configs = load_artifact(MODEL_PATH + f"/e{num_epochs}/configs")
    partial_data[hidden_features] = tables['crc_data']['df'][hidden_features] # NOTE: temporary measure to make dimensionality match training data

    column_order = tables['crc_data']['df'].drop(['placeholder'], axis=1).columns
    if 'target' in column_order:
        column_order = tables['crc_data']['df'].drop(['placeholder', 'target'], axis=1).columns

    partial_data = partial_data[column_order]
    known_features_mask = np.zeros((len(partial_data), 25))
    known_features_mask[:, [partial_data.columns.get_loc(col) for col in qi]] = 1

    cleaned_tables = clava_reconstructing(
        tables,
        relation_order,
        ATTACK_ARTIFACTS + "models",
        all_group_lengths_prob_dicts,
        models,
        configs,
        partial_data,
        known_features_mask,
        reconstruct_method_RePaint,
        sample_scale=1 if "debug" not in configs else configs["debug"]["sample_scale"],
        resamples=resamples,
        jump=jump,
    )

    # Cast int values that saved as string to int for further evaluation
    for key in cleaned_tables.keys():
        for col in cleaned_tables[key].columns:
            if cleaned_tables[key][col].dtype == "object":
                try:
                    cleaned_tables[key][col] = cleaned_tables[key][col].astype(int)
                except ValueError:
                    print(f"Column {col} cannot be converted to int.")

    reconstructed = cleaned_tables['crc_data']
    reconstructed.to_csv(data_path + f"reconstructed_{data_name}.csv")


    reconstruction_scores = pd.DataFrame(index=features_25)
    scores = calculate_reconstruction_score(targets, reconstructed, hidden_features)
    reconstruction_scores.loc[hidden_features, "tabddpm_recon"] = scores

    print(f"\n\nDeid = {data_name}:\n")
    for x in reconstruction_scores.loc[sorted(hidden_features), "tabddpm_recon"].T.to_numpy():
        print(x, end=",")
    print(np.array(scores).mean())
    return np.array(scores).mean()



def gen_synth_data():
    print(f"\nSynthesizing TabDDPM \n\n")
    ATTACK_ARTIFACTS = "attack_artifacts_nist_crc/"
    MODEL_PATH = ATTACK_ARTIFACTS + f"models"


    models = load_artifact(MODEL_PATH + f"/e{num_epochs}/model")
    tables = load_artifact(MODEL_PATH + f"/e{num_epochs}/tables")
    all_group_lengths_prob_dicts = load_artifact(MODEL_PATH + f"/e{num_epochs}/all_group_lengths_prob_dicts")
    dataset_meta = load_artifact(MODEL_PATH + f"/e{num_epochs}/dataset_meta")
    relation_order = load_artifact(MODEL_PATH + f"/e{num_epochs}/relation_order")
    configs = load_artifact(MODEL_PATH + f"/e{num_epochs}/configs")

    cleaned_tables, synthesizing_time_spent, matching_time_spent = clava_synthesizing(
        tables,
        relation_order,
        ATTACK_ARTIFACTS + "models",
        all_group_lengths_prob_dicts,
        models,
        configs,
        sample_scale=1 if "debug" not in configs else configs["debug"]["sample_scale"],
    )

    print("\nSynthesizing Complete!\n")
    for key in cleaned_tables.keys():
        print(f"{key}:")

    # Cast int values that saved as string to int for further evaluation
    synth_final = None
    for key in cleaned_tables.keys():
        for col in cleaned_tables[key].columns:
            if cleaned_tables[key][col].dtype == "object":
                try:
                    cleaned_tables[key][col] = cleaned_tables[key][col].astype(int)
                except ValueError:
                    print(f"Column {col} cannot be converted to int.")

        synth_final = cleaned_tables[key]

        synth_final.to_csv(data_path + f"TabDDPM_generated_1.csv")


################################################################
##########         HELPER FUNCTIONS          ###################
################################################################





def dump_artifact(artifact, name):
    pickle_file = open(name, 'wb')
    pickle.dump(artifact, pickle_file)
    pickle_file.close()

def load_artifact(name):
    pickle_file = open(name, 'rb')
    artifact = pickle.load(pickle_file)
    pickle_file.close()
    return artifact


def calculate_reconstruction_score(df_original, df_reconstructed, hidden_features):
    total_records = len(df_original)

    scores = []
    for col in hidden_features:
        value_counts = df_original[col].value_counts()
        rarity_scores = df_original[col].map(total_records / value_counts)
        max_score = rarity_scores.sum()

        score = ( (df_original[col].values == df_reconstructed[col].values) * rarity_scores ).sum()
        scores.append(round(score / max_score * 100, 1))
    return scores


if __name__ == '__main__':
    # main_attack()
    one_feature_at_a_time_attack()

