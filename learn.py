from collections import defaultdict
import math
import multiprocessing
from queue import Queue
import threading
import time
from brian2 import ms
import pandas as pd
import numpy as np
import jax.numpy as jnp
import jax
from numpy import VisibleDeprecationWarning

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
    dataset_name = "mbanc"

    excluded_neurons = set()
    if dataset_name == "fafb":
        neurons_to_activate = test.neu_sugar
    elif dataset_name == "banc":
        neurons_to_activate = [720575941626500746, 720575941491992807] #walk
        # neurons_to_activate = [720575941540215501, 720575941459097503 ] #giant fiber??
    elif dataset_name == "mbanc" or dataset_name == "mbanc-no-optic":
        neurons_to_activate = [10045, 10056] #walk
        # neurons_to_activate = [] #giant fiber??

        # neu_info = pd.read_feather('../flywire/body-annotations-male-cns-v0.9-minconf-0.5.feather')
        # excluded_neurons.update(neu_info[neu_info["superclass"] == "ol_intrinsic"]["bodyId"])
        # excluded_neurons.update(neu_info[neu_info["superclass"].isnull()]["bodyId"])
    else:
        raise Exception(f"unknown dataset {dataset_name}")

    # df_neu = pd.read_csv(path_comp, index_col=0)
    # df_con = pd.read_parquet(path_con)
    df_neu, df_con = data.load(dataset_name)

    multiprocessing.freeze_support()
    mp_context = multiprocessing.get_context("spawn")
    pygame_spike_queue = mp_context.Queue()
    control_queue = mp_context.Queue()
    frame_queue = mp_context.Queue()
    # frame_queue.put((700, 700, np.empty(shape=(700*700*3,), dtype=np.int8).tobytes(), 0))
    target_func = pygame_loop.start_pygame
    render_process = mp_context.Process(target=target_func, args=[pygame_spike_queue, control_queue, frame_queue, dataset_name, neurons_to_activate])
    render_process.start()

    spike_queue = Queue()
    input_queue = Queue()
    brian_control_queue = Queue()
    neuron_thread = threading.Thread(
        target=neuron_model.start_neuron_sim, 
        args=(df_neu, df_con, dataset_name, neurons_to_activate, brian_control_queue, spike_queue, input_queue),
        kwargs={"runtime": 1000 * ms}
    )
    neuron_thread.start()

    # mjc_spikes_queue = Queue()
    # obs_queue = Queue()
    # mjc_thread = threading.Thread(target=start_mjc_thread, args=(dataset_name, mjc_spikes_queue, frame_queue, obs_queue))
    # mjc_thread.start()

    # best_learned_params = {"syn_weight_mods": np.random.normal(1, 1, len(df_con))}
    a = np.full(len(df_con), 1.0)
    learned_params = {"syn_weight_mods": a}
    learned_params["neu_weight_mods"] = np.full(len(df_neu), 1.0)
    # learned_params = best_learned_params.copy()
    best_reward = 0
    random_delta = np.zeros(len(df_neu))

    lr = .001
    while True:
        spikes, last_obs = run_sim(df_neu, df_con, neurons_to_activate, learned_params, dataset_name, brian_control_queue, spike_queue, input_queue)
        print("number of spikes:", len(spikes))

        # obs_queue.get()
        # mjc_spikes_queue.put(spikes, False)

        reward = get_reward(spikes)
        # print("learned weights", learned_params["syn_weight_mods"])

        pygame_spike_queue.put(None)
        pygame_spike_queue.put((spikes[-1][0], spikes))

        gradient, neu_gradient = get_gradient(df_neu, df_con, learned_params, neurons_to_push=reward)

        #TODO some sort of penalty for being further from the data besides just clamping it
        learned_params["syn_weight_mods"] = (learned_params["syn_weight_mods"] + np.array(gradient) * 10 * lr).clip(.5, 2)
        learned_params["neu_weight_mods"] = (learned_params["neu_weight_mods"] + np.array(neu_gradient) * 1 * lr).clip(.2, 10)
        print("supposed gradient", np.sort(gradient))
        print("learned weights 2", np.sort(learned_params["syn_weight_mods"]))

        lr = .1

def jprint(x):
    # jax.debug.print(str(x))
    pass

def get_gradient(df_neu, df_con: pd.DataFrame, learned_params, neurons_to_push = {neuron_groups.mbanc_leg_neuron_groups["rf_trochanter_extensor"][0]: 1}):
    flyid2i = {j: i for i, j in enumerate(df_neu.index)}  # flywire id: brian ID
    i2flyid = {j: i for i, j in flyid2i.items()} # brian ID: flywire ID

    neurons_to_activate = [10045, 10056] #walk

    neurons_to_activate = jnp.array([flyid2i[x] for x in neurons_to_activate])
    neurons_to_push_in = neurons_to_push
    neurons_to_push = jnp.array([flyid2i[nid] for nid in neurons_to_push_in.keys()], dtype=jnp.int32)
    neuron_weights = jnp.array([weight for weight in neurons_to_push_in.values()])

    #pre_index, post_index
    jnp_con = jnp.array(df_con.to_numpy()[..., [2, 3]])
    jnp_strengths = jnp.array(df_con.to_numpy()[..., 6], dtype=jnp.float32)

    # all_neurons = forward(jnp_con, jnp_strengths, neurons_to_activate)

    learned_syn_weights = learned_params["syn_weight_mods"] 
    learned_neu_weights = learned_params["neu_weight_mods"]

    gradient, neu_gradient = jax.grad(loss, argnums=(2, 3))(jnp_con, jnp_strengths, learned_syn_weights, learned_neu_weights, neurons_to_activate, neurons_to_push, neuron_weights) 

    gradient = gradient * .01

    return gradient, neu_gradient

def forward(jnp_con, start_synapse_weights, learned_syn_weights, learned_neu_weights, neurons_to_activate):
    global_connection_strength = .001

    all_neurons = jnp.zeros(211743)
    all_neurons = all_neurons.at[jnp.array(neurons_to_activate)].set(1)
    all_neurons_orig = all_neurons

    jprint(all_neurons[jnp_con[..., 0]].shape)
    jprint(start_synapse_weights.shape)
    # start_synapse_weights = start_synapse_weights.clip(min=-100, max=100)
    synapse_weights = jnp.tanh(learned_syn_weights * start_synapse_weights / 1000)

    jprint(jax.numpy.sum(all_neurons_orig))
    for i in range(5):
        pre_synapse_strengths = (all_neurons[jnp_con[..., 0]]).clip(min=0, max=1)
        synapse_strenghts: jnp.ndarray = (pre_synapse_strengths * synapse_weights).clip(min=0, max=1)
        # jax.debug.print("i pre_synapse_strengths * start_synapse_weights * learned_weights = step1 {}, {}, {}, {}, {}", i, pre_synapse_strengths, start_synapse_weights, learned_weights, synapse_strenghts)
        neuron_updates = jnp.zeros_like(all_neurons).at[jnp_con[..., 1]].add(synapse_strenghts) * learned_neu_weights

        all_neurons += neuron_updates

        # jax.debug.print("all neurons {}", jnp.sort(all_neurons))
        # jax.debug.print("non zero {}", jnp.count_nonzero(all_neurons))

    return all_neurons

@jax.jit
def loss(jnp_con, jnp_strengths, learned_syn_weights, learned_neu_weights, neurons_to_activate, neurons_to_push, neurons_to_push_weights):
    new_neuron_values = forward(jnp_con, jnp_strengths, learned_syn_weights, learned_neu_weights, neurons_to_activate)

    return jax.numpy.sum(neurons_to_push_weights * new_neuron_values[neurons_to_push])

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
    for neuron in all_leg_neurons:
        counter[neuron] = 0
    other_counter = defaultdict(float)
    weak_counter = defaultdict(float)
    for spike_id in spikes[:, 1]:
        if spike_id in all_leg_neurons:
            counter[spike_id] += 1
        else:
            other_counter[spike_id] += 1
        # elif spike_id in next_synapses and next_synapses[spike_id][0] > 0:
            # weak_counter[spike_id] += 1
            # counter[next_synapses[spike_id][1]] += 0
            
    # result = {key: (13 - value)**2 * (1 if (13-value) > 0 else -1) for key, value in counter.items()} 
    result = {key: (13 - value) for key, value in counter.items()} 
    # result.update({key: -0.1 * (value-3) if value > 3 else 0 for key, value in other_counter.items()})
    # result.update({key: -0.02 * value for key, value in other_counter.items()})

    # result = {key: 100 for key, value in counter.items()} 

    print("SCORE!!!", sum(result.values()) + (100 - len(result)) * 13)
    return result
    # return sum([math.sqrt(x) for x in counter.values()])
        
if __name__ == "__main__":
    wrapper_thing()
    df_neu, df_con = data.load("mbanc")
    # weights = jnp.full(len(df_con), 1.0)
    # get_gradient(df_neu, df_con, weights)
