from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import multiprocessing
from queue import Empty, Queue
import threading
from time import sleep
from brian2 import Hz, Mvolt, Network, NeuronGroup, PoissonInput, Quantity, SpikeGeneratorGroup, SpikeMonitor, Synapses, mV, ms, network_operation, second, volt
import brian2
from brian2.core.variables import VariableView
import data
from drawutils import SpikeDrawer
import sys
import os
import Drosophila_brain_model.model as model
from gymtest import setup_fly
import gymtest
import neuron_groups
import neuron_model
import pygame_loop
import test
import pandas as pd
import numpy as np
import flygym
import flygym.state as flygym_state
import flygym.preprogrammed as flygym_preprogrammed
from flygym.preprogrammed import all_leg_dofs
import time
from profile_dec import profile

import testbanc

def start_sim(df_comp, df_con, neurons_to_activate):
    # neu.p_weight = 65 * mV

    # poi_inp, neu = model.poi(neu, [], [flyid2i[flyid] for flyid in test.neu_sugar], params)
    # silence neurons
    # syn = model.silence([], syn)
    # collect in Network object

    multiprocessing.freeze_support()
    mp_context = multiprocessing.get_context("spawn")
    pygame_spike_queue = mp_context.Queue()
    control_queue = mp_context.Queue()
    frame_queue = mp_context.Queue()
    # frame_queue.put((700, 700, np.empty(shape=(700*700*3,), dtype=np.int8).tobytes(), 0))
    target_func = pygame_loop.start_pygame
    render_process = mp_context.Process(target=target_func, args=[pygame_spike_queue, control_queue, frame_queue, dataset, {n: (255, 255, 255) for n in neurons_to_activate}])
    render_process.start()

    # neurons = set(neuron_groups.rf_leg_motor_neurons)

    # mjc_spikes_queue = Queue()
    # obs_queue = Queue()
    # mjc_thread = threading.Thread(target=start_mjc_thread, args=(dataset, mjc_spikes_queue, frame_queue, obs_queue))
    # mjc_thread.start()

    mjc_thread = gymtest.MjcSim(dataset_name=dataset)

    spike_queue = Queue()
    input_queue = Queue()
    learned_params = {"syn_weight_mods": np.random.normal(1, 1, len(df_con))}
    neuron_thread = threading.Thread(target=neuron_model.start_neuron_sim, args=(df_comp, df_con, dataset, neurons_to_activate, control_queue, spike_queue, input_queue))
    neuron_thread.start()

    control_queue.put(("start", learned_params))

    spikes_acc = []
    times_acc = []
    last_time = time.monotonic()
    while True:
        update_time, spikes = spike_queue.get()
        _ = mjc_thread.obs_queue.get()
        input_queue.put(())

        spikes_acc.extend(spikes)
        times = np.empty(len(spikes))
        times.fill(update_time)
        times_acc.extend(times)
        # spike_and_times = np.stack((times, spikes), 1, dtype=object)
        # pygame_spike_queue.put((now, spike_and_times))
        mjc_thread.put_spikes(spikes)

        now = time.monotonic()
        if now - last_time > 1/20:
            # print(spikes_acc)
            # print(times_acc)
            # print(zip(spikes_acc, times_acc))
            # print(list(zip(spikes_acc, times_acc)))
            # print(list(zip(spikes_acc, times_acc)))
            pygame_spike_queue.put((update_time, list(zip(times_acc, spikes_acc))), False)
            spikes_acc.clear()
            times_acc.clear()
            last_time = now

if __name__ == "__main__":
    dataset = "mbanc"

    if dataset == "fafb":
        config = {
             'path_res'  : './results',                              # directory to store results,
             'path_comp' : './Drosophila_brain_model/Completeness_783.csv',        # csv of the complete list of Flywire neurons,
             'path_con'  : './Drosophila_brain_model/Connectivity_783.parquet',    # connectivity data,
             'n_proc'    : -1,                                               # number of CPU cores (-1: use all),
        }
        neurons_to_activate = test.neu_sugar
    elif dataset == "banc":
        config = {
             'path_res'  : './results',                              # directory to store results,
             'path_comp' : './data/banc_completeness.csv',        # csv of the complete list of Flywire neurons,
             'path_con'  : './data/banc_connectivity.parquet',    # connectivity data,
             'n_proc'    : -1,                                               # number of CPU cores (-1: use all),
        }
        neurons_to_activate = [720575941626500746, 720575941491992807] #walk
        # neurons_to_activate = [720575941540215501, 720575941459097503 ] #giant fiber??
    else:
        config = {
             'path_res'  : './results',                              # directory to store results,
             'path_comp' : './data/mbanc_completeness.csv',        # csv of the complete list of Flywire neurons,
             'path_con'  : './data/mbanc_connectivity.parquet',    # connectivity data,
             'n_proc'    : -1,                                               # number of CPU cores (-1: use all),
        }
        neurons_to_activate = [10045, 10056] #walk
        # neurons_to_activate = [] #giant fiber??

    path_comp = config["path_comp"]
    path_con = config["path_con"]

    # df_comp = pd.read_csv(path_comp, index_col=0)
    # df_con = pd.read_parquet(path_con)
    df_comp, df_con = data.load(dataset)

    """
    filter_neurons = []
    # filter_neurons += neuron_groups.banc_e1
    # filter_neurons += neuron_groups.banc_e2
    # filter_neurons += neuron_groups.banc_i1
    # filter_neurons += neuron_groups.banc_i2
    filter_neurons += [720575941626500746]
    filtered_con = df_con.iloc[:0,:].copy()
    for i, a in enumerate(df_con.to_numpy()):
        if a[0] in filter_neurons and a[1] in filter_neurons:
            filtered_con.loc[i] = a

    print("filtered_connections", filtered_con)
    """

    start_sim(df_comp, df_con, neurons_to_activate)
    
