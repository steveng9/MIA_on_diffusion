ON_UW_SERVER = True

import sys
import os
import json
import pandas as pd
import numpy as np
import pickle
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


verbose = False
data_path = "/home/golobs/data/" if ON_UW_SERVER \
    else "/Users/golobs/Documents/GradSchool/NIST-CRC-25/NIST_Red-Team_Problems1-24_v2/"
# DATA_NAME = "25_Demo_MST_e10_25f"
# DATA_NAME = "25_Demo_CellSupression_25f"
DATA_NAME = "7_MST_e10_25f_QID1"
# DATA_NAME = "19_CELL_SUPPRESSION_25f_QID1"
QI = ['F37', 'F41', 'F2', 'F17', 'F22', 'F32', 'F47']
HIDDEN = ['F23', 'F13', 'F11', 'F43', 'F36', 'F15', 'F33', 'F25', 'F18', 'F5', 'F30', 'F10', 'F12', 'F50', 'F3', 'F1', 'F9', 'F21']
features_25 = ['F1', 'F2', 'F3', 'F5', 'F9', 'F10', 'F11', 'F12', 'F13', 'F15', 'F17', 'F18', 'F21', 'F22', 'F23', 'F25', 'F30', 'F32', 'F33', 'F36', 'F37', 'F41', 'F43', 'F47', 'F50']
num_epochs = 300_000
num_epochs_classifier = 20_000



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


def main_attack():
    if torch.cuda.is_available(): print("Using CUDA device :)")
    else: print("NOT Using CUDA!")
    data_names = [
        "25_Demo_AIM_e1_25f",
        "25_Demo_ARF_25f",
        "25_Demo_CellSupression_25f",
        "25_Demo_MST_e10_25f",
        "25_Demo_RANKSWAP_25f",
        "25_Demo_Synthpop_25f",
        "25_Demo_TVAE_25f",
    ]

    for data_name in data_names:
        train_diffusion(data_name)
        reconstruct_data(data_name)






def train_diffusion(data_name):
    print(f"\nTraining TabDDPM with {num_epochs} epochs\n\n")
    ATTACK_ARTIFACTS = "attack_artifacts_nist_crc/"
    MODEL_PATH = ATTACK_ARTIFACTS + f"models/e{num_epochs}"
    DATA_DIR = data_path
    # dataset_name = data_name + "_Deid"
    dataset_name = "refined_training_data"

    config_path = "configs_nist_crc/crc_data.json"
    configs, _ = load_configs(config_path, MODEL_PATH)

    configs["diffusion"]["iterations"] = num_epochs
    configs["classifier"]["iterations"] = num_epochs_classifier

    tables, relation_order, dataset_meta = load_multi_table(DATA_DIR, metadata_dir="configs_nist_crc/", dataset_name=dataset_name)
    tables, all_group_lengths_prob_dicts = clava_clustering(tables, relation_order, MODEL_PATH, configs)

    models = clava_training(tables, relation_order, MODEL_PATH, configs)
    # models, tables, all_group_lengths_prob_dicts, dataset_meta, relation_order, configs = train_diffusion_for_attack(num_epochs, MODEL_PATH, DATA_DIR, dataset_name)

    os.makedirs(MODEL_PATH + f"/e{num_epochs}", exist_ok=True)
    dump_artifact(models, MODEL_PATH + f"/model")
    dump_artifact(tables, MODEL_PATH + f"/tables")
    dump_artifact(all_group_lengths_prob_dicts, MODEL_PATH + f"/all_group_lengths_prob_dicts")
    dump_artifact(dataset_meta, MODEL_PATH + f"/dataset_meta")
    dump_artifact(relation_order, MODEL_PATH + f"/relation_order")
    dump_artifact(configs, MODEL_PATH + f"/configs")


def reconstruct_data(data_name):
    ATTACK_ARTIFACTS = "attack_artifacts_nist_crc/"
    MODEL_PATH = ATTACK_ARTIFACTS + f"models"
    targets_name = "25_Demo_25f_OriginalData" if "Demo" in data_name else data_name + "_AttackTargets"
    print(f"targets_name : {targets_name}")
    targets = pd.read_csv(data_path + targets_name + ".csv")
    partial_data = targets[QI]

    models = load_artifact(MODEL_PATH + f"/e{num_epochs}/model")
    tables = load_artifact(MODEL_PATH + f"/e{num_epochs}/tables")
    all_group_lengths_prob_dicts = load_artifact(MODEL_PATH + f"/e{num_epochs}/all_group_lengths_prob_dicts")
    dataset_meta = load_artifact(MODEL_PATH + f"/e{num_epochs}/dataset_meta")
    relation_order = load_artifact(MODEL_PATH + f"/e{num_epochs}/relation_order")
    configs = load_artifact(MODEL_PATH + f"/e{num_epochs}/configs")
    hidden_columns = ['F23', 'F13', 'F11', 'F43', 'F36', 'F15', 'F33', 'F25', 'F18', 'F5', 'F30', 'F10', 'F12', 'F50', 'F3', 'F1', 'F9', 'F21']
    partial_data[hidden_columns] = tables['crc_data']['df'][hidden_columns] # NOTE: temporary measure to make dimensionality match training data

    column_order = tables['crc_data']['df'].drop(['placeholder'], axis=1).columns
    if 'target' in column_order:
        column_order = tables['crc_data']['df'].drop(['placeholder', 'target'], axis=1).columns

    partial_data = partial_data[column_order]
    known_features_mask = np.zeros((len(partial_data), 25))
    known_features_mask[:, [partial_data.columns.get_loc(col) for col in QI]] = 1

    cleaned_tables = clava_reconstructing(
        tables,
        relation_order,
        ATTACK_ARTIFACTS + "models",
        all_group_lengths_prob_dicts,
        models,
        configs,
        partial_data,
        known_features_mask,
        sample_scale=1 if "debug" not in configs else configs["debug"]["sample_scale"],
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
    scores = calculate_reconstruction_score(targets, reconstructed, HIDDEN)
    reconstruction_scores.loc[HIDDEN, "tabddpm_recon"] = scores

    print(f"\n\nDeid = {data_name}:\n")
    for x in reconstruction_scores.loc[sorted(HIDDEN), "tabddpm_recon"].T.to_numpy():
        print(x, end=",")
    print(np.array(scores).mean())
    print(f"\n\n\n\n\n")



def gen_synth_data():
    num_epochs = 3
    print(f"\nSynthesizing TabDDPM \n\n")
    ATTACK_ARTIFACTS = "attack_artifacts_nist_crc/"
    MODEL_PATH = ATTACK_ARTIFACTS + f"models"
    # dataset_name = "19_CELL_SUPPRESSION_25f_QID1_Deid"
    dataset_name = "7_MST_e10_25f_QID1_Deid"


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
        print(cleaned_tables[key].head(10))

    # Cast int values that saved as string to int for further evaluation
    for key in cleaned_tables.keys():
        for col in cleaned_tables[key].columns:
            if cleaned_tables[key][col].dtype == "object":
                try:
                    cleaned_tables[key][col] = cleaned_tables[key][col].astype(int)
                except ValueError:
                    print(f"Column {col} cannot be converted to int.")



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
    create_large_training_set()


