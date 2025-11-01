import multiprocessing
from queue import Queue
import threading
import time
from brian2 import ms
import pandas as pd
import numpy as np

from gymtest import start_mjc_thread
import neuron_groups
import neuron_model
import pygame_loop
import test

def run_sim(df_comp, df_con, neurons_to_activate, learned_params, dataset):
    mjc_spikes_queue = Queue()
    obs_queue = Queue()
    mjc_thread = threading.Thread(target=start_mjc_thread, args=(dataset, mjc_spikes_queue, None, obs_queue))
    #mjc_thread.start()

    spike_queue = Queue()
    input_queue = Queue()
    neuron_thread = threading.Thread(
        target=neuron_model.start_neuron_sim, 
        args=(df_comp, df_con, dataset, neurons_to_activate, Queue(), spike_queue, input_queue, learned_params),
        kwargs={"runtime": 300 * ms}
    )
    neuron_thread.start()

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

    if dataset == "fafb":
        path_comp = './Drosophila_brain_model/Completeness_783.csv'
        path_con = './Drosophila_brain_model/Connectivity_783.parquet'
        neurons_to_activate = test.neu_sugar
    elif dataset == "banc":
        path_comp = './data/banc_completeness.csv'
        path_con = './data/banc_connectivity.parquet'
        neurons_to_activate = [720575941626500746, 720575941491992807] #walk
        # neurons_to_activate = [720575941540215501, 720575941459097503 ] #giant fiber??
    else:
        path_comp = './data/mbanc_completeness.csv'
        path_con = './data/mbanc_connectivity.parquet'
        neurons_to_activate = [10045, 10056] #walk
        # neurons_to_activate = [] #giant fiber??

    df_comp = pd.read_csv(path_comp, index_col=0)
    df_con = pd.read_parquet(path_con)

    multiprocessing.freeze_support()
    mp_context = multiprocessing.get_context("spawn")
    pygame_spike_queue = mp_context.Queue()
    control_queue = mp_context.Queue()
    frame_queue = mp_context.Queue()
    frame_queue.put((700, 700, np.empty(shape=(700*700*3,), dtype=np.int8).tobytes(), 0))
    target_func = pygame_loop.start_pygame
    render_process = mp_context.Process(target=target_func, args=[pygame_spike_queue, control_queue, frame_queue, dataset, neurons_to_activate])
    render_process.start()

    # max_learned_params = {"weight_mods": np.random.normal(1, 1, len(df_con))}
    max_learned_params = {"weight_mods": np.full(len(df_con), 1)}
    learned_params = max_learned_params.copy()
    max_reward = 0
    while True:

        spikes, last_obs = run_sim(df_comp, df_con, neurons_to_activate, learned_params, dataset)

        reward = get_reward(spikes)
        print("reward was", reward)
        print(max_learned_params["weight_mods"][0])

        pygame_spike_queue.put(None)
        pygame_spike_queue.put((spikes[-1][0], spikes))

        if reward > max_reward:
            max_learned_params = learned_params.copy()
            max_reward = reward
        else:
            learned_params = max_learned_params.copy()

        random_delta = np.random.normal(0, .1, len(df_con))
        learned_params["weight_mods"] = learned_params["weight_mods"] + random_delta

def get_reward(spikes):
    groups = neuron_groups.mbanc_leg_neuron_groups
    all_leg_neurons = set()
    for neurons in groups.values():
        all_leg_neurons.update(neurons)

    counter = 0
    for spike_id in spikes[:, 1]:
        if spike_id in all_leg_neurons:
            counter += 1
    return counter
        
if __name__ == "__main__":
    wrapper_thing()
