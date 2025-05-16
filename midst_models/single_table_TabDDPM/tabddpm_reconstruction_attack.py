ON_UW_SERVER = False

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
    load_configs, clava_reconstructing,
)
from midst_models.single_table_TabDDPM.configs_nist_crc.generate_config_files import data_path
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

    train_diffusion()
    # gen_synth_data()
    # reconstruct_data()






def train_diffusion():
    num_epochs = 200_000
    print(f"\nTraining TabDDPM with {num_epochs} epochs\n\n")
    ATTACK_ARTIFACTS = "attack_artifacts_nist_crc/"
    MODEL_PATH = ATTACK_ARTIFACTS + f"models/e{num_epochs}"
    DATA_DIR = "/Users/golobs/Documents/GradSchool/NIST-CRC-25/NIST_Red-Team_Problems1-24_v2/"
    # dataset_name = "19_CELL_SUPPRESSION_25f_QID1_Deid"
    dataset_name = "7_MST_e10_25f_QID1_Deid"

    config_path = "configs_nist_crc/crc_data.json"
    configs, _ = load_configs(config_path, MODEL_PATH)

    # configs["diffusion"]["iterations"] = num_epochs
    # configs["classifier"]["iterations"] = 20_000

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


#
# def train_diffusion_for_attack(num_epochs, model_path, data_dir, dataset_name):
#     config_path = "configs_nist_crc/crc_data.json"
#     configs, _ = load_configs(config_path, model_path)
#
#     configs["diffusion"]["iterations"] = num_epochs
#     configs["classifier"]["iterations"] = num_epochs
#
#     tables, relation_order, dataset_meta = load_multi_table(data_dir, metadata_dir="configs_nist_crc/", dataset_name=dataset_name)
#     tables, all_group_lengths_prob_dicts = clava_clustering(tables, relation_order, model_path, configs)
#
#     models = clava_training(tables, relation_order, model_path, configs)
#     return models, tables, all_group_lengths_prob_dicts, dataset_meta, relation_order, configs

def reconstruct_data():
    num_epochs = 3
    qi = ['F37', 'F41', 'F2', 'F17', 'F22', 'F32', 'F47']
    ATTACK_ARTIFACTS = "attack_artifacts_nist_crc/"
    MODEL_PATH = ATTACK_ARTIFACTS + f"models"
    # dataset_name = "19_CELL_SUPPRESSION_25f_QID1_AttackTargets"
    targets_name = "7_MST_e10_25f_QID1_AttackTargets"
    targets = pd.read_csv(data_path + targets_name + ".csv")
    partial_data = targets[qi]

    models = load_artifact(MODEL_PATH + f"/e{num_epochs}/model")
    tables = load_artifact(MODEL_PATH + f"/e{num_epochs}/tables")
    all_group_lengths_prob_dicts = load_artifact(MODEL_PATH + f"/e{num_epochs}/all_group_lengths_prob_dicts")
    dataset_meta = load_artifact(MODEL_PATH + f"/e{num_epochs}/dataset_meta")
    relation_order = load_artifact(MODEL_PATH + f"/e{num_epochs}/relation_order")
    configs = load_artifact(MODEL_PATH + f"/e{num_epochs}/configs")
    hidden_columns = ['F23', 'F13', 'F11', 'F43', 'F36', 'F15', 'F33', 'F25', 'F18', 'F5', 'F30', 'F10', 'F12', 'F50', 'F3', 'F1', 'F9', 'F21']
    partial_data[hidden_columns] = tables['crc_data']['df'][hidden_columns] # NOTE: temporary measure to make dimensionality match training data

    column_order = tables['crc_data']['df'].drop(['placeholder'], axis=1).columns
    partial_data = partial_data[column_order]
    known_features_mask = torch.zeros((len(partial_data), 25))
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



if __name__ == '__main__':
    main()
