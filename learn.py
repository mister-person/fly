from collections import defaultdict
import math
import multiprocessing
from queue import Queue
import threading
import time
from brian2 import ms
import pandas as pd
import numpy as np

import data
from gymtest import start_mjc_thread
import neuron_groups
import neuron_model
import pygame_loop
import test

def run_sim(df_neu, df_con, neurons_to_activate, learned_params, dataset, brian_control_queue, spike_queue, input_queue):
    mjc_spikes_queue = Queue()
    obs_queue = Queue()
    mjc_thread = threading.Thread(target=start_mjc_thread, args=(dataset, mjc_spikes_queue, None, obs_queue))
    #mjc_thread.start()

    print("sending start event")
    brian_control_queue.put(("start", learned_params))

    spikes_acc = []
    times_acc = []
    obs = None
    while True:
        next_spikes = spike_queue.get()
        if next_spikes is None:
            break
        update_time, spikes = next_spikes
        # obs = obs_queue.get()
        input_queue.put(())

        mjc_spikes_queue.put(spikes, False)
        spikes_acc.extend(spikes)
        times = np.empty(len(spikes))
        times.fill(update_time)
        times_acc.extend(times)
        # spike_and_times = np.stack((times, spikes), 1, dtype=object)
        # pygame_spike_queue.put((now, spike_and_times))

    return np.array([times_acc, spikes_acc]).transpose(), obs

def wrapper_thing():
    dataset = "mbanc"

    excluded_neurons = set()
    if dataset == "fafb":
        neurons_to_activate = test.neu_sugar
    elif dataset == "banc":
        neurons_to_activate = [720575941626500746, 720575941491992807] #walk
        # neurons_to_activate = [720575941540215501, 720575941459097503 ] #giant fiber??
    elif dataset == "mbanc" or dataset == "mbanc-no-optic":
        neurons_to_activate = [10045, 10056] #walk
        # neurons_to_activate = [] #giant fiber??

        # neu_info = pd.read_feather('../flywire/body-annotations-male-cns-v0.9-minconf-0.5.feather')
        # excluded_neurons.update(neu_info[neu_info["superclass"] == "ol_intrinsic"]["bodyId"])
        # excluded_neurons.update(neu_info[neu_info["superclass"].isnull()]["bodyId"])
    else:
        raise Exception(f"unknown dataset {dataset}")

    # df_neu = pd.read_csv(path_comp, index_col=0)
    # df_con = pd.read_parquet(path_con)
    df_neu, df_con = data.load(dataset)

    multiprocessing.freeze_support()
    mp_context = multiprocessing.get_context("spawn")
    pygame_spike_queue = mp_context.Queue()
    control_queue = mp_context.Queue()
    frame_queue = mp_context.Queue()
    frame_queue.put((700, 700, np.empty(shape=(700*700*3,), dtype=np.int8).tobytes(), 0))
    target_func = pygame_loop.start_pygame
    render_process = mp_context.Process(target=target_func, args=[pygame_spike_queue, control_queue, frame_queue, dataset, neurons_to_activate])
    render_process.start()

    spike_queue = Queue()
    input_queue = Queue()
    brian_control_queue = Queue()
    neuron_thread = threading.Thread(
        target=neuron_model.start_neuron_sim, 
        args=(df_neu, df_con, dataset, neurons_to_activate, brian_control_queue, spike_queue, input_queue),
        kwargs={"runtime": 100 * ms}
    )
    neuron_thread.start()

    # best_learned_params = {"syn_weight_mods": np.random.normal(1, 1, len(df_con))}
    # best_learned_params = {"syn_weight_mods": np.full(len(df_con), 1)}
    best_learned_params = {"neu_weight_mods": np.full(len(df_neu), 1)}
    learned_params = best_learned_params.copy()
    best_reward = 0
    random_delta = np.zeros(len(df_neu))
    while True:
        spikes, last_obs = run_sim(df_neu, df_con, neurons_to_activate, learned_params, dataset, brian_control_queue, spike_queue, input_queue)

        reward = get_reward(spikes)
        # reward = sum(learned_params["neu_weight_mods"])
        print("reward was", reward)
        print(best_learned_params["neu_weight_mods"][0])

        pygame_spike_queue.put(None)
        pygame_spike_queue.put((spikes[-1][0], spikes))

        if reward >= best_reward:
            best_learned_params = learned_params.copy()
            best_reward = reward
            momentum = random_delta
            print("momentuming")
            # momentum = 0
        else:
            learned_params = best_learned_params.copy()
            momentum = 0

        random_delta = np.random.normal(0, .005, len(df_neu)) + momentum
        learned_params["neu_weight_mods"] = learned_params["neu_weight_mods"] + random_delta

def get_gradient(df_neu, df_con: pd.DataFrame, neuron_to_push = neuron_groups.mbanc_leg_neuron_groups["rf_trochanter_extensor"][0]):
    print(neuron_to_push)
    grads_n = np.full(len(df_neu), 0)
    np_con = np.concatenate((np.expand_dims(np.arange(0, len(df_con)), 1), df_con.to_numpy()[:, [2, 3, 6]]), 1)
    # np_con: connection index, pre_index, post_index, weight

    flyid2i = {j: i for i, j in enumerate(df_neu.index)}  # flywire id: brian ID
    i2flyid = {j: i for i, j in flyid2i.items()} # brian ID: flywire ID

    '''
    grads = np.full(len(df_con), 0)
    if True == False:
        for i, c in enumerate(df_con.itertuples()):
            if getattr(c, "Postsynaptic_ID") == neuron_to_push:
                grads[i] += getattr(c, "_7") #type: ignore
    print(2)
    # neuron_to_push = 413843
    '''

    next_list = np.array([i2flyid[neuron_to_push]])
    print("next list 1", next_list)
    grads_con = np.full(len(df_con), 0)
    grads_neu = np.full(len(df_neu), 0)

    for x in range(5):
        mask = np.isin(np_con[:, 2], next_list) # 2 = post_iindex. mask is list of indices that connect to next_list
        masked = np_con[mask] # masked is con_index, pre_index, post_index, weight but only of connections from next_list

        neuron_grads = masked

        grads_neu[masked[:, 1]] += masked[:, 3]
        grads_con[masked[:, 0]] += masked[:, 3]

        next_list = masked[:, 1]
        print("next list 2", next_list, len(next_list))
        print("grads", next_list, len(next_list))

    print(3)

    print(grads2)
    print(grads2[np.where(grads2)])

def get_reward(spikes):
    groups = neuron_groups.mbanc_leg_neuron_groups
    all_leg_neurons = set()
    for neurons in groups.values():
        all_leg_neurons.update(neurons)

    _, synapse_map, rev_synapse_map = data.get_synapse_map("mbanc")
    next_synapses = {}
    for spike_id in all_leg_neurons:
        for pre_spike_id, strength in rev_synapse_map[spike_id]:
            next_synapses[pre_spike_id] = (strength, spike_id)

    counter = defaultdict(float)
    weak_counter = defaultdict(float)
    for spike_id in spikes[:, 1]:
        if spike_id in all_leg_neurons:
            counter[spike_id] += 0
        elif spike_id in next_synapses and next_synapses[spike_id][0] > 0:
            # weak_counter[spike_id] += 1
            counter[next_synapses[spike_id][1]] += .01
            
    print(counter)
    return sum([math.sqrt(x) for x in counter.values()])
        
if __name__ == "__main__":
    # wrapper_thing()
    df_neu, df_con = data.load("mbanc")
    get_gradient(df_neu, df_con)
