from sys import meta_path

from torch.nn.functional import dropout

from midst_models.single_table_TabDDPM.pipeline_utils import load_multi_table_CUSTOM

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
    clava_training_CUSTOM,
    clava_training,
    clava_load_pretrained,
    clava_synthesizing,
    load_configs, clava_reconstructing,
)
from pipeline_modules import load_multi_table
import warnings
warnings.filterwarnings("ignore")



def jump_max10(t):
    return max(0, t-10)

def jump_threeQuarter(t):
    return math.floor(t*.75)

dropout_default = 0.1
batch_size_default = 4096
lr_default =  0.0006
weight_decay_default = 1e-05
num_epochs_default = 100_000
num_timesteps_default = 1000
resamples_default = 10
jump_default = jump_max10

reconstruction = True
# reconstruct_method_RePaint = True
verbose = False
# data_path = "/home/golobs/data/" if ON_UW_SERVER else "/Users/stevengolob/Documents/school/PhD/NIST_CRC/NIST_Red-Team_Problems1-24_v2/"
# data_path = "/home/golobs/data/" if ON_UW_SERVER else "/Users/stevengolob/Documents/school/PhD/NIST_CRC/25_PracticeProblem/"
# DATA_NAME = "25_Demo_MST_e10_25f"
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






def train_diffusion(data_name):
    print(f"\nTraining TabDDPM with {num_epochs} epochs\n\n")
    ATTACK_ARTIFACTS = "/Users/stevengolob/PycharmProjects/MIA_on_diffusion/midst_models/single_table_TabDDPM/attack_artifacts_nist_crc/"
    MODEL_PATH = ATTACK_ARTIFACTS + f"models/e{num_epochs}"
    targets_name = "25_Demo_25f_OriginalData" if "Demo" in data_name else data_name + "_AttackTargets"
    print(f"targets_name : {targets_name}")
    targets = pd.read_csv(data_path + targets_name + ".csv")
    DATA_DIR = data_path
    dataset_name = "25_Demo_25f_OriginalData_noID"
    # dataset_name = "refined_training_data"

    meta_path = "/Users/stevengolob/PycharmProjects/MIA_on_diffusion/midst_models/single_table_TabDDPM/configs_nist_crc/"
    config_path = meta_path + "crc_data.json"
    configs, _ = load_configs(config_path, MODEL_PATH)

    configs["diffusion"]["iterations"] = num_epochs
    # configs["classifier"]["iterations"] = num_epochs_classifier

    tables, relation_order, dataset_meta = load_multi_table(DATA_DIR, metadata_dir=meta_path, dataset_name=dataset_name)
    tables, all_group_lengths_prob_dicts = clava_clustering(tables, relation_order, MODEL_PATH, configs)

    # partial_data[hidden_features] = tables['crc_data']['df'][hidden_features] # NOTE: temporary measure to make dimensionality match training data
    column_order = tables['crc_data']['df'].drop(['placeholder'], axis=1).columns
    if 'target' in column_order:
        column_order = tables['crc_data']['df'].drop(['placeholder', 'target'], axis=1).columns

    models = clava_training(tables, relation_order, MODEL_PATH, configs,
         for_reconstruction=False)

    os.makedirs(MODEL_PATH + f"/e{num_epochs}", exist_ok=True)
    dump_artifact(models, MODEL_PATH + f"/model")
    dump_artifact(tables, MODEL_PATH + f"/tables")
    dump_artifact(all_group_lengths_prob_dicts, MODEL_PATH + f"/all_group_lengths_prob_dicts")
    dump_artifact(dataset_meta, MODEL_PATH + f"/dataset_meta")
    dump_artifact(relation_order, MODEL_PATH + f"/relation_order")
    dump_artifact(configs, MODEL_PATH + f"/configs")

def generate_synth_data():
    print(f"\nSynthesizing TabDDPM \n\n")
    ATTACK_ARTIFACTS = "attack_artifacts_nist_crc/"
    MODEL_PATH = ATTACK_ARTIFACTS + f"models/e{num_epochs}"


    models = load_artifact(MODEL_PATH + f"/model")
    tables = load_artifact(MODEL_PATH + f"/tables")
    all_group_lengths_prob_dicts = load_artifact(MODEL_PATH + f"/all_group_lengths_prob_dicts")
    dataset_meta = load_artifact(MODEL_PATH + f"/dataset_meta")
    relation_order = load_artifact(MODEL_PATH + f"/relation_order")
    configs = load_artifact(MODEL_PATH + f"/configs")

    cleaned_tables, synthesizing_time_spent, matching_time_spent = clava_synthesizing(
        tables,
        relation_order,
        MODEL_PATH,
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



def train_diffusion_for_reconstruction(cfg, meta, domain, synth, qi, hidden_features, reconstruct_method_RePaint=False):

    # ATTACK_ARTIFACTS = cfg["dataset"].get("")
    # MODEL_PATH = ATTACK_ARTIFACTS + f"models/e{num_epochs}"
    # targets_name = "25_Demo_25f_OriginalData" if "Demo" in data_name else data_name + "_AttackTargets"
    # print(f"targets_name : {targets_name}")
    # targets = pd.read_csv(data_path + targets_name + ".csv")
    # DATA_DIR = data_path
    # dataset_name = data_name + "_Deid"
    # dataset_name = "refined_training_data"

    # partial_data = targets[qi]

    # meta_path = "/Users/stevengolob/PycharmProjects/MIA_on_diffusion/midst_models/single_table_TabDDPM/configs_nist_crc/"
    # config_path = meta_path + "crc_data.json"

    diffusion_config = make_config_for_diffusion_model(cfg)
    # diffusion_config["diffusion"]["iterations"] = cfg["attack_params"].get("num_epochs", num_epochs_default)

    tables, relation_order, dataset_meta = load_multi_table_CUSTOM(meta, domain, synth)
    tables, all_group_lengths_prob_dicts = clava_clustering(tables, relation_order, cfg["dataset"]["artifacts"], diffusion_config)

    # TODO: fix this band-aid
    # partial_data[hidden_features] = synth[hidden_features] # NOTE: temporary measure to make dimensionality match training data

    # all_columns = qi + hidden_features
    # column_order = tables['crc_data']['df'][all_columns].columns
    column_order = qi + hidden_features # all_columns # todo: can I just do this instead?

    # column_order = tables['crc_data']['df'].drop(['placeholder'], axis=1).columns
    # if 'target' in column_order:
    #     column_order = tables['crc_data']['df'].drop(['placeholder', 'target'], axis=1).columns

    # partial_data = partial_data[column_order]
    # known_features_mask = np.zeros((len(partial_data), len(column_order)))
    known_features_mask = np.zeros((len(synth), len(column_order)))
    known_features_mask[:, [synth.columns.get_loc(col) for col in qi]] = 1

    model = clava_training_CUSTOM(tables, diffusion_config, not reconstruct_method_RePaint, known_features_mask)

    dump_artifact(model, cfg["dataset"]["artifacts"] + f"/model_ckpt.pkl")
    dump_artifact(tables, cfg["dataset"]["artifacts"] + f"/tables.pkl")
    dump_artifact(all_group_lengths_prob_dicts, cfg["dataset"]["artifacts"] + f"/all_group_lengths_prob_dicts.pkl")
    # dump_artifact(dataset_meta, cfg["dataset"]["artifacts"] + f"/dataset_meta.pkl")
    dump_artifact(relation_order, cfg["dataset"]["artifacts"] + f"/relation_order.pkl")
    dump_artifact(diffusion_config, cfg["dataset"]["artifacts"] + f"/configs.pkl")
    dump_artifact(known_features_mask, cfg["dataset"]["artifacts"] + f"/known_features_mask.pkl")


def reconstruct_data(cfg, targets, qi, hidden_features, reconstruct_method_RePaint=False):
    # print(f"\nReconstructing TabDDPM \n\n")
    # ATTACK_ARTIFACTS = "/Users/stevengolob/PycharmProjects/MIA_on_diffusion/midst_models/single_table_TabDDPM/attack_artifacts_nist_crc/"
    # MODEL_PATH = ATTACK_ARTIFACTS + f"models/e{num_epochs}"
    # targets_name = "25_Demo_25f_OriginalData" if "Demo" in data_name else data_name + "_AttackTargets"
    # print(f"targets_name : {targets_name}")
    # targets = pd.read_csv(data_path + targets_name + ".csv")
    partial_data = targets[qi]

    model = load_artifact(cfg["dataset"]["artifacts"] + f"/model_ckpt.pkl")
    tables = load_artifact(cfg["dataset"]["artifacts"] + f"/tables.pkl")
    all_group_lengths_prob_dicts = load_artifact(cfg["dataset"]["artifacts"] + f"/all_group_lengths_prob_dicts.pkl")
    # dataset_meta = load_artifact(cfg["dataset"]["artifacts"] + f"/dataset_meta.pkl")
    relation_order = load_artifact(cfg["dataset"]["artifacts"] + f"/relation_order.pkl")
    configs = load_artifact(cfg["dataset"]["artifacts"] + f"/configs.pkl")
    known_features_mask = load_artifact(cfg["dataset"]["artifacts"] + f"/known_features_mask.pkl")

    # TODO: fix this band-aid
    partial_data[hidden_features] = tables['crc_data']['df'][hidden_features] # NOTE: temporary measure to make dimensionality match training data

    # column_order = tables['crc_data']['df'].drop(['placeholder'], axis=1).columns
    column_order = qi + hidden_features

    # if 'target' in column_order:
    #     column_order = tables['crc_data']['df'].drop(['placeholder', 'target'], axis=1).columns

    partial_data = partial_data[column_order]
    # known_features_mask = np.zeros((len(partial_data), len(column_order)))
    # known_features_mask[:, [partial_data.columns.get_loc(col) for col in qi]] = 1

    reconstructed = clava_reconstructing(
        tables,
        relation_order,
        all_group_lengths_prob_dicts,
        model,
        configs,
        partial_data,
        known_features_mask,
        reconstruct_method_RePaint,
        cfg["attack_params"].get("resamples", resamples_default),
        globals()[cfg["attack_params"].get("jump_fn", jump_default.__name__)],
        sample_scale=1 if "debug" not in configs else configs["debug"]["sample_scale"],
    )

    # TODO: make option for continuous vs. categorical vs. int values
    # Cast int values that saved as string to int for further evaluation
    for col in reconstructed.columns:
        if reconstructed[col].dtype == "object":
            try:
                reconstructed[col] = reconstructed[col].astype(int)
            except ValueError:
                print(f"Column {col} cannot be converted to int.")

    reconstructed.to_csv(cfg["dataset"]["artifacts"] + f"/reconstructed.csv")
    return reconstructed



################################################################
##########         HELPER FUNCTIONS          ###################
################################################################

def make_config_for_diffusion_model(cfg):
    return {
        # "general": {
        #     "sample_prefix": "",
        # },
        "diffusion": {
            "d_layers": cfg["attack_params"]["hidden_dims"],
            "dropout": cfg["attack_params"].get("dropout", dropout_default),
            "num_timesteps": cfg["attack_params"].get("num_timesteps", num_timesteps_default),
            "model_type": "mlp",
            "iterations": cfg["attack_params"].get("num_epochs", num_epochs_default),
            "batch_size": cfg["attack_params"].get("batch_size", batch_size_default),
            "lr": cfg["attack_params"].get("lr", lr_default),
            "gaussian_loss_type": "mse",
            "weight_decay": cfg["attack_params"].get("weight_decay", weight_decay_default),
            "scheduler": "cosine"
        },
        "sampling": { # TODO: do I need this?
            "batch_size": 20000,
        },
    }



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
    main_attack()
    # one_feature_at_a_time_attack()

