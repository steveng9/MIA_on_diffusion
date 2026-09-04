from midst_models.single_table_TabDDPM.pipeline_utils import load_multi_table_CUSTOM

ON_UW_SERVER = False

import sys
import pandas as pd
import numpy as np
import pickle
import math

import torch
import torch.profiler

sys.path.append("../../")
from complex_pipeline import (
    clava_clustering,
    clava_training_CUSTOM,
    clava_reconstructing,
)
import warnings
warnings.filterwarnings("ignore")



def jump_max10(t):
    return max(0, t-10)

def jump_max20(t):
    return max(0, t-20)

def jump_max50(t):
    return max(0, t-50)

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

verbose = False


# QI 1
QI = ['F37', 'F41', 'F2', 'F17', 'F22', 'F32', 'F47']
HIDDEN = ['F23', 'F13', 'F11', 'F43', 'F36', 'F15', 'F33', 'F25', 'F18', 'F5', 'F30', 'F10', 'F12', 'F50', 'F3', 'F1', 'F9', 'F21']

# QI 2
# QI = ['F37', 'F41', 'F3', 'F13', 'F18', 'F23', 'F30']
# HIDDEN = ['F11', 'F43', 'F5', 'F36', 'F25', 'F47', 'F32', 'F15', 'F33', 'F17', 'F10', 'F12', 'F2', 'F1', 'F50', 'F22', 'F9', 'F21']

features_25 = ['F1', 'F2', 'F3', 'F5', 'F9', 'F10', 'F11', 'F12', 'F13', 'F15', 'F17', 'F18', 'F21', 'F22', 'F23', 'F25', 'F30', 'F32', 'F33', 'F36', 'F37', 'F41', 'F43', 'F47', 'F50']


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

            score = reconstruct_data_categorical(data_name_reduced, qi=QI, hidden_features=[hidden_feature])
            print("SCORE for ", hidden_feature, score)
            scores.append(score)
        print()
        print()
        print(scores)


def train_diffusion_for_reconstruction(cfg, meta, domain, synth, qi, hidden_features, reconstruct_method_RePaint=False):
    diffusion_config = make_config_for_diffusion_model(cfg)
    column_order = qi + hidden_features
    synth = synth[column_order]
    tables, relation_order, dataset_meta = load_multi_table_CUSTOM(meta, domain, synth)
    tables, all_group_lengths_prob_dicts = clava_clustering(tables, relation_order, cfg["dataset"]["artifacts"], diffusion_config)
    known_features_mask = np.zeros((len(synth), len(column_order)))
    known_features_mask[:, :len(qi)] = 1

    model = clava_training_CUSTOM(tables, diffusion_config, not reconstruct_method_RePaint, known_features_mask)

    dump_artifact(model, cfg["dataset"]["artifacts"] + f"/model_ckpt.pkl")
    dump_artifact(tables, cfg["dataset"]["artifacts"] + f"/tables.pkl")
    dump_artifact(all_group_lengths_prob_dicts, cfg["dataset"]["artifacts"] + f"/all_group_lengths_prob_dicts.pkl")
    dump_artifact(relation_order, cfg["dataset"]["artifacts"] + f"/relation_order.pkl")
    dump_artifact(diffusion_config, cfg["dataset"]["artifacts"] + f"/configs.pkl")
    dump_artifact(known_features_mask, cfg["dataset"]["artifacts"] + f"/known_features_mask.pkl")


def reconstruct_data_categorical(cfg, targets, qi, hidden_features, reconstruct_method_RePaint=False):
    model = load_artifact(cfg["dataset"]["artifacts"] + f"/model_ckpt.pkl")
    tables = load_artifact(cfg["dataset"]["artifacts"] + f"/tables.pkl")
    all_group_lengths_prob_dicts = load_artifact(cfg["dataset"]["artifacts"] + f"/all_group_lengths_prob_dicts.pkl")
    relation_order = load_artifact(cfg["dataset"]["artifacts"] + f"/relation_order.pkl")
    configs = load_artifact(cfg["dataset"]["artifacts"] + f"/configs.pkl")
    known_features_mask = load_artifact(cfg["dataset"]["artifacts"] + f"/known_features_mask.pkl")

    partial_data = targets[qi]
    partial_data[hidden_features] = list(tables.values())[0]['df'][hidden_features] # NOTE: temporary measure to make dimensionality match training data
    column_order = qi + hidden_features
    partial_data = partial_data[column_order]

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


def calculate_continuous_vals_reconstruction_score(train, reconstruction, hidden_features):
    results = {}
    for hidden_feature in hidden_features:
        real = train[hidden_feature].values
        recon = reconstruction[hidden_feature].values

        # Normalize by range of real data
        data_range = real.max() - real.min()

        if data_range == 0:
            # Constant column
            normalized_error = 0 if np.allclose(real, recon) else np.inf
        else:
            # Normalized absolute error
            normalized_error = np.abs(real - recon) / data_range

        results[hidden_feature] = {
            'mean_abs_error': np.mean(np.abs(real - recon)),
            'normalized_mae': np.mean(normalized_error),
            'mse': np.mean((real - recon) ** 2),
            'rmse': np.sqrt(np.mean((real - recon) ** 2)),
            'normalized_rmse': np.sqrt(np.mean(normalized_error ** 2)),
            'max_error': np.max(np.abs(real - recon))
        }

    return pd.DataFrame(results).T



if __name__ == '__main__':
    one_feature_at_a_time_attack()

