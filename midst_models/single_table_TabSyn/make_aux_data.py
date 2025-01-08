import argparse
import json
import os
import sys
import numpy as np
import pandas as pd


dir = "/Users/golobs/PycharmProjects/MIDSTModels/data"


def infer_aux_data():
    trans_ids = set()
    trans_aux = pd.DataFrame()

    for sdg in ["tabsyn", "tabddpm"]:
        for threat_model in ["white_box", "black_box"]:
            for i in range(30):
                df = pd.read_csv(f"{dir}/{sdg}_{threat_model}/train/{sdg}_{str(i+1)}/train_with_id.csv")
                ids = df["trans_id"].values.tolist()
                nonintersection = df[~df["trans_id"].isin(trans_ids)]
                print(nonintersection.shape)
                trans_aux = pd.concat([trans_aux, nonintersection], ignore_index=True)
                trans_ids = trans_ids.union(set(ids))

    print(len(ids), trans_aux.shape)
    trans_aux.to_csv(f"{dir}/auxiliary_inferred/trans_aux.csv", index=False)


def verify_aux_data():
    aux_inferred = pd.read_csv(f"{dir}/auxiliary_inferred/trans_aux.csv")
    aux = pd.read_csv(f"{dir}/auxiliary/trans.csv", sep=';')
    ids_inf = set(aux_inferred["trans_id"].values.tolist())
    ids_ = set(aux["trans_id"].values.tolist())

    assert len(ids_inf) == len(ids_inf.intersection(ids_))
    assert len(ids_) == len(ids_inf.union(ids_))
    print("inferred data looks good")



# infer_aux_data()
verify_aux_data()
