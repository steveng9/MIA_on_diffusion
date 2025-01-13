
import sys
import os
import json
import pandas as pd
import numpy as np
import pickle

import torch
from torch.utils.data import DataLoader
import torch.profiler

from scripts.process_dataset import get_column_name_mapping, train_val_test_split

from src.data import preprocess, TabularDataset
from src.tabsyn.pipeline import TabSyn
from src import load_config
import warnings
warnings.filterwarnings("ignore")



num_epochs = int(sys.argv[1])
save_results = sys.argv[2] == "save"
model_number = int(sys.argv[3])


def main():
    # main_train()
    attack_VAE()


def attack_VAE():
    print()
    print(f"Attacking VAE with {num_epochs} epochs!!!")
    print()
    print()

    threat_model = "black_box"
    model_num = model_number

    INFO_DIR = "data_info"
    ATTACK_ARTIFACTS = "attack_artifacts/"

    DATA_NAME = "trans/"
    DATA_DIR_ALL = ATTACK_ARTIFACTS + f"data/model{model_num}/data_all/"
    DATA_DIR_SYNTH = ATTACK_ARTIFACTS + f"data/model{model_num}/data_synth/"
    DATA_DIR_CHALLENGE = ATTACK_ARTIFACTS + f"data/model{model_num}/data_challenge/"
    MODEL_PATH_S = ATTACK_ARTIFACTS + f"models/model{model_num}/tabsynS"
    MODEL_PATH_A = ATTACK_ARTIFACTS + f"models/model{model_num}/tabsynA"
    LOSS_RESULTS = ATTACK_ARTIFACTS + f"loss_results/model{model_num}/"
    os.makedirs(LOSS_RESULTS, exist_ok=True)

    train = pd.read_csv(f"../../data/tabsyn_{threat_model}/train/tabsyn_{model_num}/train_with_id.csv", header="infer")
    synth = pd.read_csv(f"../../data/tabsyn_{threat_model}/train/tabsyn_{model_num}/trans_synthetic.csv", header="infer")
    challenge = pd.read_csv(f"../../data/tabsyn_{threat_model}/train/tabsyn_{model_num}/challenge_with_id.csv", header="infer")
    membership = pd.read_csv(f"../../data/tabsyn_{threat_model}/train/tabsyn_{model_num}/challenge_label.csv", header="infer")
    aux = pd.read_csv(f"../../data/auxiliary_inferred/trans_aux.csv", header="infer")

    with open(f"data/processed_data/trans/info.json", "r") as file:
        data_info = json.load(file)

    train.drop(columns=["trans_id", "account_id"], inplace=True)
    challenge.drop(columns=["trans_id", "account_id"], inplace=True)
    aux.drop(columns=["trans_id", "account_id"], inplace=True)
    modified_process_data("trans", INFO_DIR, DATA_DIR_SYNTH, synth)
    modified_process_data("trans_all", INFO_DIR, DATA_DIR_ALL, aux)
    modified_process_data("trans", INFO_DIR, DATA_DIR_CHALLENGE, challenge)

    config_path = "src/configs/trans.toml"
    raw_config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using CUDA device!")
    # if torch.backends.mps.is_available():
    #     print("MPS Available!")
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    _, X_train_num_c, X_train_cat_c, X_test_num_c, X_test_cat_c, categories_c, d_numerical_c = preprocess_for_attack(raw_config, device, DATA_DIR_CHALLENGE + "processed_data/trans/", DATA_DIR_ALL)

    tabsyn_synth = train_vae_for_attack(raw_config, device, num_epochs, MODEL_PATH_S, DATA_NAME, preprocess_for_attack(raw_config, device, DATA_DIR_SYNTH + "processed_data/trans/", DATA_DIR_ALL))
    challenge_mse_loss_s, challenge_ce_loss_s, challenge_kl_loss_s, challenge_acc_s = tabsyn_synth.attack_vae(X_test_num_c, X_test_cat_c)
    if save_results:
        with open(LOSS_RESULTS + f'synth_losses_{model_num}.pkl', 'wb') as file:
            pickle.dump((challenge_mse_loss_s, challenge_ce_loss_s, challenge_kl_loss_s, challenge_acc_s), file)

    tabsyn_aux = train_vae_for_attack(raw_config, device, num_epochs//3, MODEL_PATH_A, DATA_NAME, preprocess_for_attack(raw_config, device, DATA_DIR_ALL + "processed_data/trans_all/", DATA_DIR_ALL))
    challenge_mse_loss_a, challenge_ce_loss_a, challenge_kl_loss_a, challenge_acc_a = tabsyn_aux.attack_vae(X_test_num_c, X_test_cat_c)
    if save_results:
        with open(LOSS_RESULTS + f'aux_losses_{model_num}.pkl', 'wb') as file:
            pickle.dump((challenge_mse_loss_a, challenge_ce_loss_a, challenge_kl_loss_a, challenge_acc_a), file)


def preprocess_for_attack(raw_config, device, data_dir, data_dir_all):
    X_num, X_cat, categories, d_numerical = preprocess(
        data_dir,
        ref_dataset_path=data_dir_all + "processed_data/trans_all/",
        transforms=raw_config["transforms"],
        task_type=raw_config["task_type"],
    )
    X_train_num, X_test_num = X_num
    X_train_cat, X_test_cat = X_cat
    X_train_num, X_test_num = torch.tensor(X_train_num).float(), torch.tensor(X_test_num).float()
    X_train_cat, X_test_cat = torch.tensor(X_train_cat, dtype=torch.long), torch.tensor(X_test_cat, dtype=torch.long)
    X_test_num = X_test_num.float().to(device)
    X_test_cat = X_test_cat.to(device)

    train_data = TabularDataset(X_train_num.float(), X_train_cat)
    return train_data, X_train_num, X_train_cat, X_test_num, X_test_cat, d_numerical, categories



def train_vae_for_attack(raw_config, device, epochs, model_path, data_name, processed_data_artifacts):
    train_data, X_train_num, X_train_cat, X_test_num, X_test_cat, d_numerical, categories = processed_data_artifacts
    train_loader = DataLoader(
        train_data,
        batch_size=raw_config["train"]["vae"]["batch_size"],
        shuffle=True,
        num_workers=raw_config["train"]["vae"]["num_dataset_workers"],
    )
    tabsyn = TabSyn(
        train_loader,
        X_test_num,
        X_test_cat,
        num_numerical_features=d_numerical,
        num_classes=categories,
        device=device,
    )
    tabsyn.instantiate_vae(
        **raw_config["model_params"], optim_params=raw_config["train"]["optim"]["vae"]
    )
    os.makedirs(f"{model_path}/{data_name}/vae", exist_ok=True)
    tabsyn.train_vae(
        **raw_config["loss_params"],
        # num_epochs=raw_config["train"]["vae"]["num_epochs"],
        num_epochs=epochs,
        save_path=os.path.join(model_path, data_name, "vae"),
    )
    tabsyn.save_vae_embeddings(
        X_train_num, X_train_cat, vae_ckpt_dir=os.path.join(model_path, data_name, "vae")
    )
    return tabsyn


def attack_diffusion():
    threat_model = "black_box"
    model_num = 1

    INFO_DIR = "data_info"
    ATTACK_ARTIFACTS = "attack_artifacts/"

    DATA_NAME = "trans/"
    DATA_DIR_ALL = ATTACK_ARTIFACTS + f"data/model{model_num}/data_all/"
    DATA_DIR_SYNTH = ATTACK_ARTIFACTS + f"data/model{model_num}/data_synth/"
    DATA_DIR_CHALLENGE = ATTACK_ARTIFACTS + f"data/model{model_num}/data_challenge/"

    MODEL_PATH_S = ATTACK_ARTIFACTS + f"models/model{model_num}/tabsynS"
    MODEL_PATH_A = ATTACK_ARTIFACTS + f"models/model{model_num}/tabsynA"
    MODEL_PATH_C = ATTACK_ARTIFACTS + f"models/model{model_num}/tabsynC"

    LOSS_RESULTS = ATTACK_ARTIFACTS + f"loss_results/model{model_num}/"
    os.makedirs(LOSS_RESULTS, exist_ok=True)

    train = pd.read_csv(f"../../data/tabsyn_{threat_model}/train/tabsyn_{model_num}/train_with_id.csv", header="infer")
    synth = pd.read_csv(f"../../data/tabsyn_{threat_model}/train/tabsyn_{model_num}/trans_synthetic.csv", header="infer")
    challenge = pd.read_csv(f"../../data/tabsyn_{threat_model}/train/tabsyn_{model_num}/challenge_with_id.csv", header="infer")
    membership = pd.read_csv(f"../../data/tabsyn_{threat_model}/train/tabsyn_{model_num}/challenge_label.csv", header="infer")
    aux = pd.read_csv(f"../../data/auxiliary_inferred/trans_aux.csv", header="infer")

    with open(f"data/processed_data/trans/info.json", "r") as file:
        data_info = json.load(file)

    train.drop(columns=["trans_id", "account_id"], inplace=True)
    challenge.drop(columns=["trans_id", "account_id"], inplace=True)
    aux.drop(columns=["trans_id", "account_id"], inplace=True)
    # modified_process_data("trans_train", INFO_DIR, DATA_DIR, train)
    modified_process_data("trans", INFO_DIR, DATA_DIR_SYNTH, synth)
    modified_process_data("trans_all", INFO_DIR, DATA_DIR_ALL, aux)
    modified_process_data("trans", INFO_DIR, DATA_DIR_CHALLENGE, challenge)

    ## TabSyn Algorithm

    config_path = "src/configs/trans.toml"
    raw_config = load_config(config_path)

    ## make dataset
    X_num_synth, X_cat_synth, categories_synth, d_numerical_synth = preprocess(
        DATA_DIR_SYNTH + "processed_data/trans/",
        ref_dataset_path= DATA_DIR_ALL + "processed_data/trans_all/",
        transforms=raw_config["transforms"],
        task_type=raw_config["task_type"],
    )
    X_num_aux, X_cat_aux, categories_aux, d_numerical_aux = preprocess(
        DATA_DIR_ALL + "processed_data/trans_all/",
        ref_dataset_path=DATA_DIR_ALL + "processed_data/trans_all/",
        transforms=raw_config["transforms"],
        task_type=raw_config["task_type"],
    )
    X_num_chal, X_cat_chal, categories_chal, d_numerical_chal = preprocess(
        DATA_DIR_CHALLENGE + "processed_data/trans/",
        ref_dataset_path= DATA_DIR_ALL + "processed_data/trans_all/",
        transforms=raw_config["transforms"],
        task_type=raw_config["task_type"],
    )

    XS_train_num, XS_test_num = X_num_synth
    XS_train_cat, XS_test_cat = X_cat_synth

    XA_train_num, XA_test_num = X_num_aux
    XA_train_cat, XA_test_cat = X_cat_aux

    XC_train_num, XC_test_num = X_num_chal
    XC_train_cat, XC_test_cat = X_cat_chal


    # convert to float tensor
    XS_train_num, XS_test_num = torch.tensor(XS_train_num).float(), torch.tensor(XS_test_num).float()
    XS_train_cat, XS_test_cat = torch.tensor(XS_train_cat, dtype=torch.long), torch.tensor(XS_test_cat, dtype=torch.long)

    XA_train_num, XA_test_num = torch.tensor(XA_train_num).float(), torch.tensor(XA_test_num).float()
    XA_train_cat, XA_test_cat = torch.tensor(XA_train_cat, dtype=torch.long), torch.tensor(XA_test_cat, dtype=torch.long)

    XC_train_num, XC_test_num = torch.tensor(XC_train_num).float(), torch.tensor(XC_test_num).float()
    XC_train_cat, XC_test_cat = torch.tensor(XC_train_cat, dtype=torch.long), torch.tensor(XC_test_cat, dtype=torch.long)


    # create dataset module
    train_dataS = TabularDataset(XS_train_num.float(), XS_train_cat)
    train_dataA = TabularDataset(XA_train_num.float(), XA_train_cat)
    train_dataC = TabularDataset(XC_train_num.float(), XC_train_cat)


    # move test data to gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if torch.backends.mps.is_available():
    #     print("MPS Available!")
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


    XS_test_num = XS_test_num.float().to(device)
    XS_test_cat = XS_test_cat.to(device)

    XA_test_num = XA_test_num.float().to(device)
    XA_test_cat = XA_test_cat.to(device)

    XC_test_num = XC_test_num.float().to(device)
    XC_test_cat = XC_test_cat.to(device)

    # create train dataloader
    train_loaderS = DataLoader(
        train_dataS,
        batch_size=raw_config["train"]["vae"]["batch_size"],
        shuffle=True,
        num_workers=raw_config["train"]["vae"]["num_dataset_workers"],
    )
    train_loaderA = DataLoader(
        train_dataA,
        batch_size=raw_config["train"]["vae"]["batch_size"],
        shuffle=True,
        num_workers=raw_config["train"]["vae"]["num_dataset_workers"],
    )
    train_loaderC = DataLoader(
        train_dataC,
        batch_size=raw_config["train"]["vae"]["batch_size"],
        shuffle=True,
        num_workers=raw_config["train"]["vae"]["num_dataset_workers"],
    )

    ## Instantiate Model
    tabsynS = TabSyn(
        train_loaderS,
        XS_test_num,
        XS_test_cat,
        num_numerical_features=d_numerical_synth,
        num_classes=categories_synth,
        device=device,
    )
    tabsynA = TabSyn(
        train_loaderA,
        XA_test_num,
        XA_test_cat,
        num_numerical_features=d_numerical_aux,
        num_classes=categories_aux,
        device=device,
    )
    tabsynC = TabSyn(
        train_loaderC,
        XC_test_num,
        XC_test_cat,
        num_numerical_features=d_numerical_chal,
        num_classes=categories_chal,
        device=device,
    )


    ## Train Diffusion Model
    train_zS, _ = tabsynS.load_latent_embeddings(
        os.path.join(MODEL_PATH_S, DATA_NAME, "vae")
    )  # train_z dim: B x in_dim
    train_zA, _ = tabsynA.load_latent_embeddings(
        os.path.join(MODEL_PATH_A, DATA_NAME, "vae")
    )  # train_z dim: B x in_dim
    # train_zC, _ = tabsynC.load_latent_embeddings(
    #     os.path.join(MODEL_PATH_C, DATA_NAME, "vae")
    # )  # train_z dim: B x in_dim

    # normalize embeddings
    meanS, stdS = train_zS.mean(0), train_zS.std(0)
    latent_train_dataS = (train_zS - meanS) / 2

    meanA, stdA = train_zA.mean(0), train_zA.std(0)
    latent_train_dataA = (train_zA - meanA) / 2

    # meanC, stdC = train_zC.mean(0), train_zC.std(0)
    # latent_train_dataC = (train_zC - meanC) / 2

    # create data loader
    latent_train_loaderS = DataLoader(
        latent_train_dataS,
        batch_size=raw_config["train"]["diffusion"]["batch_size"],
        shuffle=True,
        num_workers=raw_config["train"]["diffusion"]["num_dataset_workers"],
    )
    latent_train_loaderA = DataLoader(
        latent_train_dataA,
        batch_size=raw_config["train"]["diffusion"]["batch_size"],
        shuffle=True,
        num_workers=raw_config["train"]["diffusion"]["num_dataset_workers"],
    )
    # latent_train_loaderC = DataLoader(
    #     latent_train_dataC,
    #     batch_size=raw_config["train"]["diffusion"]["batch_size"],
    #     shuffle=True,
    #     num_workers=raw_config["train"]["diffusion"]["num_dataset_workers"],
    # )

    # instantiate diffusion model for training
    tabsynS.instantiate_diffusion(
        in_dim=train_zS.shape[1],
        hid_dim=train_zS.shape[1],
        optim_params=raw_config["train"]["optim"]["diffusion"],
    )
    tabsynA.instantiate_diffusion(
        in_dim=train_zA.shape[1],
        hid_dim=train_zA.shape[1],
        optim_params=raw_config["train"]["optim"]["diffusion"],
    )
    # tabsynC.instantiate_diffusion(
    #     in_dim=train_zC.shape[1],
    #     hid_dim=train_zC.shape[1],
    #     optim_params=raw_config["train"]["optim"]["diffusion"],
    # )

    os.makedirs(f"{MODEL_PATH_S}/{DATA_NAME}", exist_ok=True)
    os.makedirs(f"{MODEL_PATH_A}/{DATA_NAME}", exist_ok=True)
    # os.makedirs(f"{MODEL_PATH_C}/{DATA_NAME}", exist_ok=True)

    print()

    # train diffusion model
    tabsynS.train_diffusion(
        latent_train_loaderS,
        # num_epochs=raw_config["train"]["diffusion"]["num_epochs"],
        num_epochs=2,
        ckpt_path=os.path.join(MODEL_PATH_S, DATA_NAME),
    )
    tabsynA.train_diffusion(
        latent_train_loaderA,
        # num_epochs=raw_config["train"]["diffusion"]["num_epochs"],
        num_epochs=2,
        ckpt_path=os.path.join(MODEL_PATH_A, DATA_NAME),
    )
    # tabsynC.train_diffusion(
    #     latent_train_loaderC,
    #     # num_epochs=raw_config["train"]["diffusion"]["num_epochs"],
    #     num_epochs=2,
    #     ckpt_path=os.path.join(MODEL_PATH_C, DATA_NAME),
    # )






    ## Load pretrained model
    latent_embeddings_pathS = os.path.join(MODEL_PATH_S, DATA_NAME, "vae")
    pretrained_model_pathS = os.path.join(MODEL_PATH_S, DATA_NAME)
    latent_embeddings_pathA = os.path.join(MODEL_PATH_A, DATA_NAME, "vae")
    pretrained_model_pathA = os.path.join(MODEL_PATH_A, DATA_NAME)

    tabsynS.instantiate_vae(**raw_config["model_params"], optim_params=None)
    train_zS, token_dimS = tabsynS.load_latent_embeddings(latent_embeddings_pathS)
    tabsynS.instantiate_diffusion(
        in_dim=train_zS.shape[1], hid_dim=train_zS.shape[1], optim_params=None
    )
    tabsynS.load_model_state(ckpt_dir=pretrained_model_pathS, dif_ckpt_name="model.pt")

    tabsynA.instantiate_vae(**raw_config["model_params"], optim_params=None)
    train_zA, token_dimA = tabsynA.load_latent_embeddings(latent_embeddings_pathA)
    tabsynA.instantiate_diffusion(
        in_dim=train_zA.shape[1], hid_dim=train_zA.shape[1], optim_params=None
    )
    tabsynA.load_model_state(ckpt_dir=pretrained_model_pathA, dif_ckpt_name="model.pt")


    ## Sample Data
    # load data info file
    with open(os.path.join(PROCESSED_DATA_DIR, DATA_NAME, "info.json"), "r") as file:
        data_info = json.load(file)
    data_info["token_dim"] = token_dim

    # get inverse tokenizers
    _, _, categories, d_numerical, num_inverse, cat_inverse = preprocess(
        os.path.join(PROCESSED_DATA_DIR, DATA_NAME),
        ref_dataset_path=REF_DATA_PATH,
        transforms=raw_config["transforms"],
        task_type=raw_config["task_type"],
        inverse=True,
    )

    os.makedirs(os.path.join(SYNTH_DATA_DIR, DATA_NAME), exist_ok=True)

    # sample data
    num_samples = train_z.shape[0]
    in_dim = train_z.shape[1]
    mean_input_emb = train_z.mean(0)
    tabsyn.sample(
        num_samples,
        in_dim,
        mean_input_emb,
        info=data_info,
        num_inverse=num_inverse,
        cat_inverse=cat_inverse,
        save_path=os.path.join(SYNTH_DATA_DIR, DATA_NAME, "tabsyn.csv"),
    )

    df = pd.read_csv(os.path.join(SYNTH_DATA_DIR, DATA_NAME, "tabsyn.csv"))
    df.head(10)



def main_train():
    threat_model = "black_box"
    model_num = 1

    INFO_DIR = "data_info"
    DATA_DIR = "data"
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed_data")
    # SYNTH_DATA_DIR = os.path.join(DATA_DIR, "synthetic_data")
    DATA_NAME = "trans"
    DATA_DIR_ALL = "all_data/"

    MODEL_PATH = "models/tabsyn"

    train = pd.read_csv(f"../../data/tabsyn_{threat_model}/train/tabsyn_{model_num}/train_with_id.csv", header="infer")
    aux = pd.read_csv(f"../../data/auxiliary_inferred/trans_aux.csv", header="infer")

    with open(f"data/processed_data/trans/info.json", "r") as file:
        data_info = json.load(file)

    train.drop(columns=["trans_id", "account_id"], inplace=True)
    aux.drop(columns=["trans_id", "account_id"], inplace=True)
    modified_process_data("trans", INFO_DIR, DATA_DIR, train)
    modified_process_data("trans_all", INFO_DIR, DATA_DIR_ALL, aux)

    ## TabSyn Algorithm

    config_path = "src/configs/trans.toml"
    raw_config = load_config(config_path)

    ## make dataset
    X_num, X_cat, categories, d_numerical = preprocess(
        "data/processed_data/trans/",
        ref_dataset_path="all_data/processed_data/trans_all/",
        transforms=raw_config["transforms"],
        task_type=raw_config["task_type"],
    )

    X_train_num, X_test_num = X_num
    X_train_cat, X_test_cat = X_cat

    # convert to float tensor
    X_train_num, X_test_num = (
        torch.tensor(X_train_num).float(),
        torch.tensor(X_test_num).float(),
    )
    X_train_cat, X_test_cat = torch.tensor(X_train_cat, dtype=torch.long), torch.tensor(X_test_cat, dtype=torch.long)

    # create dataset module
    train_data = TabularDataset(X_train_num.float(), X_train_cat)

    # move test data to gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if torch.backends.mps.is_available():
    #     print("MPS Available!")
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    X_test_num = X_test_num.float().to(device)
    X_test_cat = X_test_cat.to(device)

    # create train dataloader
    train_loader = DataLoader(
        train_data,
        batch_size=raw_config["train"]["vae"]["batch_size"],
        shuffle=True,
        num_workers=raw_config["train"]["vae"]["num_dataset_workers"],
    )

    ## Instantiate Model
    tabsyn = TabSyn(
        train_loader,
        X_test_num,
        X_test_cat,
        num_numerical_features=d_numerical,
        num_classes=categories,
        device=device,
    )

    ## Train VAE
    tabsyn.instantiate_vae(
        **raw_config["model_params"], optim_params=raw_config["train"]["optim"]["vae"]
    )

    os.makedirs(f"{MODEL_PATH}/{DATA_NAME}/vae", exist_ok=True)
    tabsyn.train_vae(
        **raw_config["loss_params"],
        # num_epochs=raw_config["train"]["vae"]["num_epochs"],
        num_epochs=1,
        save_path=os.path.join(MODEL_PATH, DATA_NAME, "vae"),
    )

    # embed all inputs in the latent space
    tabsyn.save_vae_embeddings(
        X_train_num, X_train_cat, vae_ckpt_dir=os.path.join(MODEL_PATH, DATA_NAME, "vae")
    )

    ## Train Diffusion Model
    train_z, _ = tabsyn.load_latent_embeddings(
        os.path.join(MODEL_PATH, DATA_NAME, "vae")
    )  # train_z dim: B x in_dim

    # normalize embeddings
    mean, std = train_z.mean(0), train_z.std(0)
    latent_train_data = (train_z - mean) / 2

    # create data loader
    latent_train_loader = DataLoader(
        latent_train_data,
        batch_size=raw_config["train"]["diffusion"]["batch_size"],
        shuffle=True,
        num_workers=raw_config["train"]["diffusion"]["num_dataset_workers"],
    )

    # instantiate diffusion model for training
    tabsyn.instantiate_diffusion(
        in_dim=train_z.shape[1],
        hid_dim=train_z.shape[1],
        optim_params=raw_config["train"]["optim"]["diffusion"],
    )

    os.makedirs(f"{MODEL_PATH}/{DATA_NAME}", exist_ok=True)
    # train diffusion model
    tabsyn.train_diffusion(
        latent_train_loader,
        # num_epochs=raw_config["train"]["diffusion"]["num_epochs"],
        num_epochs=1,
        ckpt_path=os.path.join(MODEL_PATH, DATA_NAME),
    )

    ## Load pretrained model
    latent_embeddings_path = os.path.join(MODEL_PATH, DATA_NAME, "vae")
    pretrained_model_path = os.path.join(MODEL_PATH, DATA_NAME)

    tabsyn.instantiate_vae(**raw_config["model_params"], optim_params=None)
    train_z, token_dim = tabsyn.load_latent_embeddings(latent_embeddings_path)
    tabsyn.instantiate_diffusion(
        in_dim=train_z.shape[1], hid_dim=train_z.shape[1], optim_params=None
    )
    tabsyn.load_model_state(ckpt_dir=pretrained_model_path, dif_ckpt_name="model.pt")


    ## Sample Data
    # load data info file
    with open(os.path.join(PROCESSED_DATA_DIR, DATA_NAME, "info.json"), "r") as file:
        data_info = json.load(file)
    data_info["token_dim"] = token_dim

    # get inverse tokenizers
    _, _, categories, d_numerical, num_inverse, cat_inverse = preprocess(
        os.path.join(PROCESSED_DATA_DIR, DATA_NAME),
        ref_dataset_path=REF_DATA_PATH,
        transforms=raw_config["transforms"],
        task_type=raw_config["task_type"],
        inverse=True,
    )

    os.makedirs(os.path.join(SYNTH_DATA_DIR, DATA_NAME), exist_ok=True)

    # sample data
    num_samples = train_z.shape[0]
    in_dim = train_z.shape[1]
    mean_input_emb = train_z.mean(0)
    tabsyn.sample(
        num_samples,
        in_dim,
        mean_input_emb,
        info=data_info,
        num_inverse=num_inverse,
        cat_inverse=cat_inverse,
        save_path=os.path.join(SYNTH_DATA_DIR, DATA_NAME, "tabsyn.csv"),
    )

    df = pd.read_csv(os.path.join(SYNTH_DATA_DIR, DATA_NAME, "tabsyn.csv"))
    df.head(10)


def modified_process_data(name, info_path, data_dir, data_df):
    processed_data_dir = os.path.join(data_dir, "processed_data")

    with open(f"{info_path}/{name}.json", "r") as f:
        info = json.load(f)

    num_data = data_df.shape[0]

    column_names = (
        info["column_names"] if info["column_names"] else data_df.columns.tolist()
    )

    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]
    target_col_idx = info["target_col_idx"]

    idx_mapping, inverse_idx_mapping, idx_name_mapping = get_column_name_mapping(
        data_df, num_col_idx, cat_col_idx, target_col_idx, column_names
    )

    num_columns = [column_names[i] for i in num_col_idx]
    cat_columns = [column_names[i] for i in cat_col_idx]
    target_columns = [column_names[i] for i in target_col_idx]

    # assume no "test data" specified
    num_train = num_data
    # num_train = int(num_data * 0.9)
    num_test = num_data - num_train

    train_df, test_df, seed = train_val_test_split(
        data_df, cat_columns, num_train, num_test
    )

    train_df.columns = range(len(train_df.columns))
    test_df.columns = range(len(test_df.columns))

    col_info = {}

    for col_idx in num_col_idx:
        col_info[col_idx] = {}
        col_info["type"] = "numerical"
        col_info["max"] = float(train_df[col_idx].max())
        col_info["min"] = float(train_df[col_idx].min())

    for col_idx in cat_col_idx:
        col_info[col_idx] = {}
        col_info["type"] = "categorical"
        col_info["categorizes"] = list(set(train_df[col_idx]))

    for col_idx in target_col_idx:
        if info["task_type"] == "regression":
            col_info[col_idx] = {}
            col_info["type"] = "numerical"
            col_info["max"] = float(train_df[col_idx].max())
            col_info["min"] = float(train_df[col_idx].min())
        else:
            col_info[col_idx] = {}
            col_info["type"] = "categorical"
            col_info["categorizes"] = list(set(train_df[col_idx]))

    info["column_info"] = col_info

    train_df.rename(columns=idx_name_mapping, inplace=True)
    test_df.rename(columns=idx_name_mapping, inplace=True)

    for col in num_columns:
        train_df.loc[train_df[col] == "?", col] = np.nan
    for col in cat_columns:
        train_df.loc[train_df[col] == "?", col] = "nan"
    for col in num_columns:
        test_df.loc[test_df[col] == "?", col] = np.nan
    for col in cat_columns:
        test_df.loc[test_df[col] == "?", col] = "nan"

    X_num_train = train_df[num_columns].to_numpy().astype(np.float32)
    X_cat_train = train_df[cat_columns].to_numpy().astype(np.int64)
    y_train = train_df[target_columns].to_numpy()

    X_num_test = test_df[num_columns].to_numpy().astype(np.float32)
    X_cat_test = test_df[cat_columns].to_numpy().astype(np.int32)
    y_test = test_df[target_columns].to_numpy()

    if not os.path.exists(f"{processed_data_dir}/{name}"):
        os.makedirs(f"{processed_data_dir}/{name}")

    np.save(f"{processed_data_dir}/{name}/X_num_train.npy", X_num_train)
    np.save(f"{processed_data_dir}/{name}/X_cat_train.npy", X_cat_train)
    np.save(f"{processed_data_dir}/{name}/y_train.npy", y_train)

    np.save(f"{processed_data_dir}/{name}/X_num_test.npy", X_num_test)
    np.save(f"{processed_data_dir}/{name}/X_cat_test.npy", X_cat_test)
    np.save(f"{processed_data_dir}/{name}/y_test.npy", y_test)

    train_df[num_columns] = train_df[num_columns].astype(np.float32)
    test_df[num_columns] = test_df[num_columns].astype(np.float32)

    train_df.to_csv(f"{processed_data_dir}/{name}/train.csv", index=False)
    test_df.to_csv(f"{processed_data_dir}/{name}/test.csv", index=False)

    info["column_names"] = column_names
    info["train_num"] = train_df.shape[0]
    info["test_num"] = test_df.shape[0]

    info["idx_mapping"] = idx_mapping
    info["inverse_idx_mapping"] = inverse_idx_mapping
    info["idx_name_mapping"] = idx_name_mapping

    metadata = {"columns": {}}
    task_type = info["task_type"]
    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]
    target_col_idx = info["target_col_idx"]

    for i in num_col_idx:
        metadata["columns"][i] = {}
        metadata["columns"][i]["sdtype"] = "numerical"
        metadata["columns"][i]["computer_representation"] = "Float"

    for i in cat_col_idx:
        metadata["columns"][i] = {}
        metadata["columns"][i]["sdtype"] = "categorical"

    if task_type == "regression":
        for i in target_col_idx:
            metadata["columns"][i] = {}
            metadata["columns"][i]["sdtype"] = "numerical"
            metadata["columns"][i]["computer_representation"] = "Float"

    else:
        for i in target_col_idx:
            metadata["columns"][i] = {}
            metadata["columns"][i]["sdtype"] = "categorical"

    info["metadata"] = metadata

    with open(f"{processed_data_dir}/{name}/info.json", "w") as file:
        json.dump(info, file, indent=4)

    print(f"Processing and Saving {name} Successfully!")

    print("Dataset Name:", name)
    print("Total Size:", info["train_num"] + info["test_num"])
    print("Train Size:", info["train_num"])
    print("Test Size:", info["test_num"])
    if info["task_type"] == "regression":
        num = len(info["num_col_idx"] + info["target_col_idx"])
        cat = len(info["cat_col_idx"])
    else:
        cat = len(info["cat_col_idx"] + info["target_col_idx"])
        num = len(info["num_col_idx"])
    print("Number of Numerical Columns:", num)
    print("Number of Categorical Columns:", cat)



# code for profiling GPU usage:
#
# with torch.profiler.profile(
#     activities=[
#         torch.profiler.ProfilerActivity.CPU,
#         torch.profiler.ProfilerActivity.CUDA,
#     ],
#     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
#     record_shapes=True,
#     with_stack=True
# ) as prof:
#     # Your PyTorch training/inference code here
#
# print(prof.key_averages().table(sort_by="cuda_time_total"))



if __name__ == '__main__':
    main()
