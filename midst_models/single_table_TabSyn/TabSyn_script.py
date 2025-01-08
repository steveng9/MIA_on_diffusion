

import os
import json
import pandas as pd
from pprint import pprint

import torch
from torch.utils.data import DataLoader

from scripts.process_dataset import process_data

from src.data import preprocess, TabularDataset
from src.tabsyn.pipeline import TabSyn
from src import load_config







def main():
    INFO_DIR = "data_info"

    DATA_DIR = "data/"
    RAW_DATA_DIR = os.path.join(DATA_DIR, "raw_data")
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed_data")
    SYNTH_DATA_DIR = os.path.join(DATA_DIR, "synthetic_data")
    DATA_NAME = "trans"

    MODEL_PATH = "models/tabsyn"


    # process data
    process_data(DATA_NAME, INFO_DIR, DATA_DIR)

    # review data
    df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, DATA_NAME, "train.csv"))
    df.head(10)


    # review json file and its contents
    with open(f"{PROCESSED_DATA_DIR}/{DATA_NAME}/info.json", "r") as file:
        data_info = json.load(file)



    DATA_DIR_ALL = "all_data/"
    RAW_DATA_DIR_ALL = os.path.join(DATA_DIR_ALL, "raw_data")
    PROCESSED_DATA_DIR_ALL = os.path.join(DATA_DIR_ALL, "processed_data")
    DATA_NAME_ALL = "trans_all"

    process_data(DATA_NAME_ALL, INFO_DIR, DATA_DIR_ALL)

    REF_DATA_PATH = os.path.join(PROCESSED_DATA_DIR_ALL, DATA_NAME_ALL)


    ## TabSyn Algorithm


    config_path = os.path.join("src/configs", f"{DATA_NAME}.toml")
    raw_config = load_config(config_path)

    pprint(raw_config)



    ## make dataset

    # preprocess data
    X_num, X_cat, categories, d_numerical = preprocess(
        os.path.join(PROCESSED_DATA_DIR, DATA_NAME),
        ref_dataset_path=REF_DATA_PATH,
        transforms=raw_config["transforms"],
        task_type=raw_config["task_type"],
    )

    # separate train and test data
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

    # instantiate VAE model for training
    tabsyn.instantiate_vae(
        **raw_config["model_params"], optim_params=raw_config["train"]["optim"]["vae"]
    )

    os.makedirs(f"{MODEL_PATH}/{DATA_NAME}/vae", exist_ok=True)
    tabsyn.train_vae(
        **raw_config["loss_params"],
        # num_epochs=raw_config["train"]["vae"]["num_epochs"],
        num_epochs=2,
        save_path=os.path.join(MODEL_PATH, DATA_NAME, "vae"),
    )


    # embed all inputs in the latent space
    tabsyn.save_vae_embeddings(
        X_train_num, X_train_cat, vae_ckpt_dir=os.path.join(MODEL_PATH, DATA_NAME, "vae")
    )



    ## Train Diffusion Model

    # load latent space embeddings
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
        num_epochs=2,
        ckpt_path=os.path.join(MODEL_PATH, DATA_NAME),
    )



    ## Load pretrained model

    latent_embeddings_path = os.path.join(MODEL_PATH, DATA_NAME, "vae")
    pretrained_model_path = os.path.join(MODEL_PATH, DATA_NAME)

    # instantiate VAE model
    tabsyn.instantiate_vae(**raw_config["model_params"], optim_params=None)

    # load latent embeddings of input data
    train_z, token_dim = tabsyn.load_latent_embeddings(latent_embeddings_path)

    # instantiate diffusion model
    tabsyn.instantiate_diffusion(
        in_dim=train_z.shape[1], hid_dim=train_z.shape[1], optim_params=None
    )

    # load state from checkpoint
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





if __name__ == '__main__':
    main()
