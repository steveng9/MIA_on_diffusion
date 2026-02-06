# from sys import meta_path

# from torch.nn.functional import dropout

# from midst_models.single_table_TabDDPM.pipeline_utils import load_multi_table_CUSTOM

ON_UW_SERVER = False

import sys
import os
import pickle

import torch
from torch.utils.data import DataLoader
import torch.profiler

sys.path.append("../../")
import category_encoders
from complex_pipeline import (
    clava_clustering,
    clava_training,
    clava_synthesizing,
    load_configs,
)
from pipeline_modules import load_multi_table
import warnings
warnings.filterwarnings("ignore")



dropout_default = 0.1
batch_size_default = 4096
lr_default =  0.0006
weight_decay_default = 1e-05
num_epochs_default = 100_000
num_timesteps_default = 1000
verbose = True

DATA_DIR = "/home/golobs/data/" if ON_UW_SERVER else "/Users/stevengolob/Documents/school/PhD/reconstruction_project/data/nist_arizona_data/NIST_CRC/25_PracticeProblem/"
DATA_NAME = "25_Demo_25f_OriginalData"
MODEL_PATH = DATA_DIR + f"tabddpm_models/"
META_PATH = DATA_DIR + f"tabddpm_meta/"
CONFIG_PATH = META_PATH + "crc_data.json"
features_25 = ['F1', 'F2', 'F3', 'F5', 'F9', 'F10', 'F11', 'F12', 'F13', 'F15', 'F17', 'F18', 'F21', 'F22', 'F23', 'F25', 'F30', 'F32', 'F33', 'F36', 'F37', 'F41', 'F43', 'F47', 'F50']





def main():
    if torch.cuda.is_available(): print("Using CUDA device :)")
    else: print("NOT Using CUDA!")

    train_diffusion(DATA_NAME)
    generate_synth_data()


def train_diffusion(data_name):
    configs, _ = load_configs(CONFIG_PATH, MODEL_PATH)
    configs["general"]["data_dir"] = DATA_DIR
    configs["general"]["exp_name"] = "trial_1"
    configs["general"]["workspace_dir"] = MODEL_PATH
    # configs["general"]["test_data_dir"] = DATA_DIR
    print(f"\nTraining TabDDPM with {configs['diffusion']['iterations']} epochs\n\n")

    tables, relation_order, dataset_meta = load_multi_table(DATA_DIR, metadata_dir=META_PATH, dataset_name=DATA_NAME)
    tables, all_group_lengths_prob_dicts = clava_clustering(tables, relation_order, MODEL_PATH, configs)

    column_order = tables['crc_data']['df'].drop(['placeholder'], axis=1).columns
    if 'target' in column_order:
        column_order = tables['crc_data']['df'].drop(['placeholder', 'target'], axis=1).columns

    models = clava_training(tables, relation_order, MODEL_PATH, configs)

    os.makedirs(MODEL_PATH, exist_ok=True)
    dump_artifact(models, MODEL_PATH + f"/model.pkl")
    dump_artifact(tables, MODEL_PATH + f"/tables.pkl")
    dump_artifact(all_group_lengths_prob_dicts, MODEL_PATH + f"/all_group_lengths_prob_dicts.pkl")
    # dump_artifact(dataset_meta, MODEL_PATH + f"/dataset_meta.pkl")
    dump_artifact(relation_order, MODEL_PATH + f"/relation_order.pkl")
    dump_artifact(configs, MODEL_PATH + f"/configs.pkl")



def generate_synth_data():
    print(f"\nSynthesizing TabDDPM \n\n")
    models = load_artifact(MODEL_PATH + f"/model.pkl")
    tables = load_artifact(MODEL_PATH + f"/tables.pkl")
    all_group_lengths_prob_dicts = load_artifact(MODEL_PATH + f"/all_group_lengths_prob_dicts.pkl")
    relation_order = load_artifact(MODEL_PATH + f"/relation_order.pkl")
    configs = load_artifact(MODEL_PATH + f"/configs.pkl")

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

        synth_final.to_csv(DATA_DIR + f"TabDDPM_generated_1.csv")
    print(f"saved to {DATA_DIR}/TabDDPM_generated_1.csv")



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

