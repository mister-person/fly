from collections import defaultdict
import os
import pandas as pd
import numpy as np

BANC = 0
FAFB = 1
MBANC = 2
MBANC_NO_OPTIC = 3

dataset_names = {"banc": BANC, "fafb": FAFB, "mbanc": MBANC, "mbanc-no-optic": MBANC_NO_OPTIC}

def load(dataset: int | str):
    if type(dataset) == str:
        dataset = dataset_names[dataset]
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
    elif dataset == MBANC_NO_OPTIC:
        path_comp = './data/mbanc_no_optic_completeness.csv'
        path_con = './data/mbanc_no_optic_connectivity.parquet'
        try:
            df_comp = pd.read_csv(path_comp, index_col=0)
            df_con = pd.read_parquet(path_con)
        except FileNotFoundError:
            df_comp, df_con = process_mbanc_data(filter_optic = True)
            df_comp.to_csv(path_comp, index=False, header=[""])
            df_con.to_parquet(path_con)
    else:
        raise Exception(f"unknown dataset {dataset}")

    return df_comp, df_con

cache = {}
def get_synapse_map(dataset):
    if dataset in cache:
        return cache[dataset]
    if dataset == "banc":
        SYNAPSES_FILENAME = "./data/banc_connectivity.parquet"
    elif dataset == "fafb":
        SYNAPSES_FILENAME = "./Drosophila_brain_model/Connectivity_783.parquet"
    elif dataset == "mbanc":
        SYNAPSES_FILENAME = "./data/mbanc_connectivity.parquet"
    elif dataset == "mbanc-no-optic":
        SYNAPSES_FILENAME = "./data/mbanc_no_optic_connectivity.parquet"
    else:
        raise Exception(f"unknown dataset {dataset}")

    synapses_df = pd.read_parquet(SYNAPSES_FILENAME)
    synapses = synapses_df.to_numpy()
    synapse_map = defaultdict(list)
    reverse_synapse_map = defaultdict(list)
    for synapse in synapses[1:]:
        synapse_map[synapse[0]].append((synapse[1], synapse[6]))
        reverse_synapse_map[synapse[1]].append((synapse[0], synapse[6]))

    cache[dataset] = synapses, synapse_map, reverse_synapse_map
    return synapses, synapse_map, reverse_synapse_map

banc_walk = [720575941626500746, 720575941491992807] #walk

def process_mbanc_data(filter_optic = False):
    df_con_initial = pd.read_feather('./data/connectome-weights-male-cns-v0.9-minconf-0.5.feather')
    nts = pd.read_feather("./data/body-neurotransmitters-male-cns-v0.9.feather")
    df_neurons = pd.read_feather("./data/body-annotations-male-cns-v0.9-minconf-0.5.feather")
    #filter out synapses with neurons that aren't in the neuron table
    l = len(df_neurons)
    pre_syns_indexes = np.searchsorted(df_neurons["bodyId"], df_con_initial["body_pre"])
    post_syns_indexes = np.searchsorted(df_neurons["bodyId"], df_con_initial["body_post"])
    pre_mask = df_neurons["bodyId"].to_numpy()[pre_syns_indexes % l] == df_con_initial["body_pre"]
    post_mask = df_neurons["bodyId"].to_numpy()[post_syns_indexes % l] == df_con_initial["body_post"]

    #filter synapses <3 connections
    syn_mask = df_con_initial["weight"] >= 3

    df_con_filtered = df_con_initial[pre_mask & post_mask & syn_mask]
    assert(type(df_con_filtered) == pd.DataFrame)
    
    #filter out synapses and neurons from optic lobe
    if filter_optic:
        excluded_neurons = set()
        excluded_neurons.update(df_neurons[df_neurons["superclass"] == "ol_intrinsic"]["bodyId"])
        # excluded_neurons.update(df_neurons[df_neurons["superclass"].isnull()]["bodyId"])
        optic_mask = np.full(len(df_con_filtered), True)
        for i in range(len(df_con_filtered)):
            if df_con_filtered["body_pre"].iloc[i] in excluded_neurons or df_con_filtered["body_post"].iloc[i] in excluded_neurons:
                optic_mask[i] = False
            if i % 1000000 == 0:
                print(i)
        # df_con = df_con[optic_mask]
        print("optic mask len", len(optic_mask))
        df_con_filtered = df_con_filtered[optic_mask]
        assert(type(df_con_filtered) == pd.DataFrame)

        neuron_optic_mask = np.full(len(df_neurons), True)
        for i in range(len(df_neurons)):
            if df_neurons["bodyId"][i] in excluded_neurons:
                neuron_optic_mask[i] = False
        df_neurons = df_neurons[neuron_optic_mask]
        assert(type(df_neurons) == pd.DataFrame)

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
    neurons, con = load(MBANC_NO_OPTIC)
    print("con", con)
    print("neurons", neurons)
