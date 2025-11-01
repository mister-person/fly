import os
import pandas as pd
import numpy as np

BANC = 0
FAFB = 1
MBANC = 2

def load(dataset):
    pass
    if dataset == FAFB:
        path_comp = './Drosophila_brain_model/Completeness_783.csv'
        path_con = './Drosophila_brain_model/Connectivity_783.parquet'
        df_comp = pd.read_csv(path_comp, index_col=0)
        df_con = pd.read_parquet(path_con)
    elif dataset == BANC:
        path_comp = './data/banc_completeness.csv'
        path_con = './data/banc_connectivity.parquet'
        df_comp = pd.read_csv(path_comp, index_col=0)
        df_con = pd.read_parquet(path_con)
        # neurons_to_activate = [720575941540215501, 720575941459097503 ] #giant fiber??
    elif dataset == MBANC:
        path_comp = './data/mbanc_completeness.csv'
        path_con = './data/mbanc_connectivity.parquet'
        try:
            df_comp = pd.read_csv(path_comp, index_col=0)
            df_con = pd.read_parquet(path_con)
        except FileNotFoundError:
            df_comp, df_con = process_mbanc_data()
            df_comp.to_csv(path_comp, index=False, header=[""])
            df_con.to_parquet(path_con)
    else:
        return (None, None)

    return df_comp, df_con

banc_walk = [720575941626500746, 720575941491992807] #walk

def process_mbanc_data():
    df_con_initial = pd.read_feather('../flywire/connectome-weights-male-cns-v0.9-minconf-0.5.feather')
    nts = pd.read_feather("../flywire/body-neurotransmitters-male-cns-v0.9.feather")
    df_neurons = pd.read_feather("../flywire/body-annotations-male-cns-v0.9-minconf-0.5.feather")
    #filter out synnapses with neurons that aren't in the neuron table
    l = len(df_neurons)
    pre_syns_indexes = np.searchsorted(df_neurons["bodyId"], df_con_initial["body_pre"])
    post_syns_indexes = np.searchsorted(df_neurons["bodyId"], df_con_initial["body_post"])
    pre_mask = df_neurons["bodyId"].to_numpy()[pre_syns_indexes % l] == df_con_initial["body_pre"]
    post_mask = df_neurons["bodyId"].to_numpy()[post_syns_indexes % l] == df_con_initial["body_post"]
    #filter synapses <3 connections
    syn_mask = df_con_initial["weight"] >= 3
    df_con_filtered = df_con_initial[pre_mask & post_mask & syn_mask]
    print(df_con_filtered)
    print("masks:", pre_mask, post_mask, syn_mask)
    print("mask lengths", sum(pre_mask), sum(post_mask), sum(pre_mask & post_mask))
    print("old con list size", len(df_con_initial))
    print("new con list size", len(df_con_filtered))

    pre_syns_nt = np.searchsorted(nts["body"], df_con_filtered["body_pre"])
    print(sum(nts["body"].to_numpy()[pre_syns_nt] != df_con_filtered["body_pre"]))
    con_nts = nts["consensus_nt"][pre_syns_nt]
    con_nts_exc = np.select([con_nts == "gaba", con_nts == 'unclear'], [-1, 0], default=1)
    con_nts_strength = df_con_filtered["weight"] * con_nts_exc

    #repeating work here but who cares
    pre_syns_indexes = np.searchsorted(df_neurons["bodyId"], df_con_filtered["body_pre"])
    # pre_syns_indexes = pre_syns_nt
    post_syns_indexes = np.searchsorted(df_neurons["bodyId"], df_con_filtered["body_post"])
    # post_syns_indexes = np.searchsorted(nts["body"], df_con_filtered["body_post"])
    # l = len(df_neurons)
    # print(sum(df_neurons["bodyId"].to_numpy()[pre_syns_indexes % l] != df_con_filtered["body_pre"]))
    # print(sum(df_neurons["bodyId"].to_numpy()[post_syns_indexes % l] != df_con_filtered["body_post"]))
    l = len(df_neurons)
    print("check, should be 0:", sum(df_neurons["bodyId"].to_numpy()[pre_syns_indexes % l] != df_con_filtered["body_pre"]))
    print("check, should be 0:", sum(df_neurons["bodyId"].to_numpy()[post_syns_indexes % l] != df_con_filtered["body_post"]))

    columns = {
        "Presynaptic_ID": df_con_filtered["body_pre"],
        "Postsynaptic_ID": df_con_filtered["body_post"],
        "Presynaptic_Index": pre_syns_indexes,
        "Postsynaptic_Index": post_syns_indexes,
        "Connectivity": df_con_filtered["weight"],
        "Excitatory": con_nts_exc,
        "Excitatory x Connectivity": con_nts_strength,
    }
    print(11)
    df_con = pd.DataFrame(columns)
    print(12)

    # df_comp = df_neurons["bodyId"]
    df_comp: pd.Series | pd.DataFrame = df_neurons["bodyId"]

    return df_comp, df_con

if __name__ == "__main__":
    con, neurons = load(MBANC)
    print("con", con)
    print("neurons", neurons)
