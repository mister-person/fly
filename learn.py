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
    a: np.typing.NDArray = np.full(len(df_con), 1.0)
    best_learned_params: dict[str, np.typing.NDArray] = {"syn_weight_mods": a}
    # best_learned_params = {"neu_weight_mods": np.full(len(df_neu), 1)}
    learned_params: dict[str, np.typing.NDArray] = best_learned_params.copy()
    best_reward = 0
    random_delta = np.zeros(len(df_neu))
    while True:
        spikes, last_obs = run_sim(df_neu, df_con, neurons_to_activate, learned_params, dataset, brian_control_queue, spike_queue, input_queue)

        reward = get_reward(spikes)
        # reward = sum(learned_params["syn_weight_mods"])
        print("reward was", reward)
        print(best_learned_params["syn_weight_mods"][0])

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

        # random_delta = np.random.normal(0, .005, len(df_neu)) + momentum
        # learned_params["syn_weight_mods"] = learned_params["syn_weight_mods"] + random_delta
        learned_params["syn_weight_mods"] = learned_params["syn_weight_mods"] - np.array(get_gradient(df_neu, df_con, learned_params["syn_weight_mods"]))

def jprint(x):
    jax.debug.print(str(x))

def forward(jnp_con, jnp_strengths, weights, neurons_to_activate):
    connection_strength = .003

    all_neurons = jnp.zeros(211470)
    all_neurons = all_neurons.at[jnp.array(neurons_to_activate)].set(1)
    all_neurons_orig = all_neurons

    jprint(all_neurons[jnp_con[..., 0]].shape)
    jprint(jnp_strengths.shape)

    jprint(jax.numpy.sum(all_neurons_orig))
    for _ in range(10):
        step1 = (all_neurons[jnp_con[..., 0]] * jnp_strengths * weights * connection_strength).clip(max=1)
        all_neurons = all_neurons.at[jnp_con[..., 1]].add(step1)

        jprint(step1.shape)
        jprint(all_neurons.shape)

        jax.debug.print("{}", jax.numpy.sum(all_neurons))

    return all_neurons

@jax.jit
def loss(jnp_con, jnp_strengths, weights, neurons_to_activate, neurons_to_push):
    all_neurons = forward(jnp_con, jnp_strengths, weights, neurons_to_activate)

    return sum(all_neurons[neurons_to_push])

def get_gradient(df_neu, df_con: pd.DataFrame, weights, neurons_to_push = [neuron_groups.mbanc_leg_neuron_groups["rf_trochanter_extensor"][0]]):
    flyid2i = {j: i for i, j in enumerate(df_neu.index)}  # flywire id: brian ID
    i2flyid = {j: i for i, j in flyid2i.items()} # brian ID: flywire ID

    neurons_to_activate = [10045, 10056] #walk

    neurons_to_activate = jnp.array([flyid2i[x] for x in neurons_to_activate])
    neurons_to_push = jnp.array([flyid2i[x] for x in neurons_to_push])
    #TODO neurons to push should have weights on them

    print("neurons to push", neurons_to_push)

    #pre_index, post_index
    jnp_con = jnp.array(df_con.to_numpy()[..., [2, 3]])
    jnp_strengths = jnp.array(df_con.to_numpy()[..., 6], dtype=jnp.float32)

    # all_neurons = forward(jnp_con, jnp_strengths, neurons_to_activate)

    gradient = jax.grad(loss, argnums=2)(jnp_con, jnp_strengths, weights, neurons_to_activate, neurons_to_push) * .01

    print(gradient.sort())
    return gradient

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
    wrapper_thing()
    df_neu, df_con = data.load("mbanc")
    # weights = jnp.full(len(df_con), 1.0)
    # get_gradient(df_neu, df_con, weights)
