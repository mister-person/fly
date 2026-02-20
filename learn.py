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
from gymtest import MjcSim
import neuron_groups
import neuron_model
from pygame_loop import PygameProcess
import pygame_loop
import test

def run_sim(df_neu, df_con, neurons_to_activate, learned_params, neuron_sim: neuron_model.NeuronSim, mjc_sim: MjcSim, pygame_process: PygameProcess):
    # mjc_thread = threading.Thread(target=start_mjc_thread, args=(dataset, mjc_spikes_queue, frame_queue, obs_queue))
    
    mjc_sim.reset()

    print("sending start event")
    neuron_sim.start(learned_params)

    pygame_process.reset()

    spikes_acc = []
    times_acc = []
    obs = None
    while True:
        next_spikes = neuron_sim.spike_queue.get()
        if next_spikes is None:
            break
        update_time, spikes = next_spikes

        # obs = mjc_sim.obs_queue.get()
        neuron_sim.input_queue.put(()) #TODO actually do this and make method

        mjc_sim.put_spikes(spikes)

        spikes_acc.extend(spikes)
        # times = np.empty(len(spikes))
        # times.fill(update_time)
        times = np.full(len(spikes), update_time)
        times_acc.extend(times)

        spike_and_times = np.stack((times, spikes), 1, dtype=object)
        pygame_process.add_spikes((update_time, spike_and_times))

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

    """ #from before I put this in an object
    multiprocessing.freeze_support()
    mp_context = multiprocessing.get_context("spawn")
    pygame_spike_queue = mp_context.Queue()
    control_queue = mp_context.Queue()
    frame_queue = mp_context.Queue()
    # frame_queue.put((700, 700, np.empty(shape=(700*700*3,), dtype=np.int8).tobytes(), 0))
    target_func = pygame_loop.start_pygame
    render_process = mp_context.Process(target=target_func, args=[pygame_spike_queue, control_queue, frame_queue, dataset_name, neurons_to_activate])
    render_process.start()
    """

    color_map = {}
    for n in neurons_to_activate:
        color_map[n] = (255, 255, 0)
    pygame_process = pygame_loop.PygameProcess(dataset_name, color_map)

    '''
    spike_queue = Queue()
    input_queue = Queue()
    brian_control_queue = Queue()
    neuron_thread = threading.Thread(
        target=neuron_model.start_neuron_sim, 
        args=(df_neu, df_con, dataset_name, neurons_to_activate, brian_control_queue, spike_queue, input_queue),
        kwargs={"runtime": 1000 * ms}
    )
    neuron_thread.start()
    '''

    runtime = 200
    neuron_sim = neuron_model.NeuronSim(df_neu, df_con, dataset_name, neurons_to_activate, runtime=runtime * ms)

    mjc_sim = MjcSim(dataset_name)
    mjc_sim.frame_queue = pygame_process.frame_queue

    # best_learned_params = {"syn_weight_mods": np.random.normal(1, 1, len(df_con))}
    a = np.full(len(df_con), 1.0)
    learned_params = {"syn_weight_mods": a}
    learned_params["neu_weight_mods"] = np.full(len(df_neu), 1.0)
    # learned_params = best_learned_params.copy()
    best_reward = 0
    random_delta = np.zeros(len(df_neu))

    lr = .3
    while True:
        spikes, last_obs = run_sim(df_neu, df_con, neurons_to_activate, learned_params, neuron_sim, mjc_sim, pygame_process)
        print("number of spikes:", len(spikes))
        print(spike_stats(spikes))

        reward = get_reward(spikes, runtime)
        # print("learned weights", learned_params["syn_weight_mods"])

        # pygame_spike_queue.put(None)
        # pygame_spike_queue.put((spikes[-1][0], spikes))

        gradient, neu_gradient = get_gradient(df_neu, df_con, learned_params, neurons_to_push=reward)

        #TODO some sort of penalty for being further from the data besides just clamping it
        learned_params["syn_weight_mods"] = (learned_params["syn_weight_mods"] + np.array(gradient) * .1 * lr).clip(.2, 4)
        learned_params["neu_weight_mods"] = (learned_params["neu_weight_mods"] + np.array(neu_gradient) * 1 * lr).clip(.1, 30)
        # print("supposed gradient", np.sort(gradient))
        print("learned weights syn", np.sort(learned_params["syn_weight_mods"]))
        print("learned weights neu", np.sort(learned_params["neu_weight_mods"]))

        lr = .7

def jprint(x):
    # jax.debug.print(str(x))
    pass
def get_gradient(df_neu, df_con: pd.DataFrame, learned_params, neurons_to_push = {neuron_groups.mbanc_leg_neuron_groups["rf_trochanter_extensor"][0]: 1}):
    flyid2i = {j: i for i, j in enumerate(df_neu.index)}  # flywire id: brian ID
    # i2flyid = {j: i for i, j in flyid2i.items()} # brian ID: flywire ID

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

    start = time.monotonic()
    gradient, neu_gradient = jit_grad_loss(jnp_con, jnp_strengths, learned_syn_weights, learned_neu_weights, neurons_to_activate, neurons_to_push, neuron_weights) 
    end = time.monotonic()

    print("jax took", end - start, "seconds")
    print("gradient", gradient.sort(), neu_gradient.sort())

    return gradient, neu_gradient

def forward(jnp_con, start_synapse_weights, learned_syn_weights, learned_neu_weights, neurons_to_activate):
    # global_connection_strength = .001

    all_neurons = jnp.zeros(211743)
    all_neurons = all_neurons.at[jnp.array(neurons_to_activate)].set(1)
    all_neurons_orig = all_neurons

    # jprint(all_neurons[jnp_con[..., 0]].shape)
    # jprint(start_synapse_weights.shape)
    # start_synapse_weights = start_synapse_weights.clip(min=-100, max=100)
    # synapse_weights = jnp.tanh(learned_syn_weights * start_synapse_weights / 1000)
    synapse_weights = start_synapse_weights * learned_syn_weights

    # jprint(jax.numpy.sum(all_neurons_orig))
    for _ in range(5):
        pre_synapse_strengths = (all_neurons[jnp_con[..., 0]]).clip(min=0, max=1)
        synapse_strenghts: jnp.ndarray = (pre_synapse_strengths * synapse_weights)
        # jax.debug.print("i pre_synapse_strengths * start_synapse_weights * learned_weights = step1 {}, {}, {}, {}, {}", i, pre_synapse_strengths, start_synapse_weights, learned_weights, synapse_strenghts)
        neuron_updates = jnp.zeros_like(all_neurons).at[jnp_con[..., 1]].add(synapse_strenghts) * learned_neu_weights

        # all_neurons += jnp.tanh((neuron_updates)/1000)#.clip(min=0)
        # all_neurons += jnp.tanh((neuron_updates - 100)/250)#.clip(min=0)
        all_neurons += ((neuron_updates)/1000)#.clip(min=0)

        all_neurons = all_neurons.clip(min=0, max=1)

        # jax.debug.print("all neurons {}", jnp.sort(all_neurons))
        # jax.debug.print("non zero {}", jnp.count_nonzero(all_neurons))

    return all_neurons

def loss(jnp_con, jnp_strengths, learned_syn_weights, learned_neu_weights, neurons_to_activate, neurons_to_push, neurons_to_push_weights):
    new_neuron_values = forward(jnp_con, jnp_strengths, learned_syn_weights, learned_neu_weights, neurons_to_activate)

    return jax.numpy.sum(neurons_to_push_weights * new_neuron_values[neurons_to_push])

jit_grad_loss = jax.jit(jax.grad(loss, argnums=(2, 3)))

def get_reward(spikes, runtime):
    all_leg_neurons = set()
    # groups = neuron_groups.mbanc_leg_neuron_groups
    # for neurons in groups.values():
        # all_leg_neurons.update(neurons)

    # all_leg_neurons.update(neuron_groups.mbanc_lf_leg_neurons)
    all_leg_neurons.update(neuron_groups.mbanc_rf_leg_neurons)

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
    result = {key: (13 * (runtime / 1000) - value) for key, value in counter.items()} 
    # result.update({key: -0.1 * (value-3) if value > 3 else 0 for key, value in other_counter.items()})
    # result.update({key: -0.01 * value for key, value in other_counter.items()})

    # maybe averaging them will do something?
    average = sum(result.values()) / len(result)
    for key in result.keys():
        result[key] = average

    # result = {key: 100 for key, value in counter.items()} 

    print("SCORE!!!", sum(result.values()) + (100 - len(result)) * 13)
    return result
    # return sum([math.sqrt(x) for x in counter.values()])

def spike_stats(spikes):
    spike_counter = defaultdict(int)
    for spike in spikes[:, 1]:
        spike_counter[spike] += 1
    return spike_counter

if __name__ == "__main__":
    wrapper_thing()
    df_neu, df_con = data.load("mbanc")
    # weights = jnp.full(len(df_con), 1.0)
    # get_gradient(df_neu, df_con, weights)
