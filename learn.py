from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
import threading
import time
from brian2 import mV, ms
import pandas as pd
import numpy as np
import jax.numpy as jnp
import jax
import flygym.preprogrammed as flygym_preprogrammed

import data
from gymtest import MjcSim
import gymtest
import neuron_groups
import neuron_model
from profile_dec import profile
from pygame_loop import PygameProcess
import pygame_loop
import matplotlib.pyplot as plt

def run_sim(df_neu, df_con, neurons_to_activate, learned_params, neuron_sim: neuron_model.NeuronSim, mjc_sim: MjcSim, pygame_process: PygameProcess):
    # mjc_thread = threading.Thread(target=start_mjc_thread, args=(dataset, mjc_spikes_queue, frame_queue, obs_queue))
    
    mjc_sim.reset()

    print("sending start event")
    neuron_sim.start(learned_params)

    pygame_process.reset()

    plt.pause(1)

    spikes_acc = []
    times_acc = []
    obs = None
    while True:
        next_spikes = neuron_sim.spike_queue.get()
        if next_spikes is None:
            break
        update_time, spikes = next_spikes

        obs = mjc_sim.obs_queue.get() #comment out to uncouple mjc sim
        neuron_sim.input_queue.put(()) #TODO actually do this and make method

        mjc_sim.put_spikes(spikes)

        spikes_acc.extend(spikes)
        # times = np.empty(len(spikes))
        # times.fill(update_time)
        times = np.full(len(spikes), update_time)
        times_acc.extend(times)

        spike_and_times = np.stack((times, spikes), 1, dtype=object)
        pygame_process.add_spikes((update_time, spike_and_times))

    voltages: np.ndarray = neuron_sim.voltages

    joint_records = np.array(mjc_sim.joint_records).transpose()

    return np.array(times_acc), np.array(spikes_acc).transpose(), obs, voltages, joint_records

def wrapper_thing():
    dataset_name = "mbanc"

    excluded_neurons = set()
    if dataset_name == "fafb":
        neurons_to_activate = neuron_groups.neu_sugar
    elif dataset_name == "banc":
        # neurons_to_activate = [720575941626500746, 720575941491992807] #walk
        # neurons_to_activate = [720575941572713886, 720575941519464046] #walk 2 (dng97)
        neurons_to_activate = [720575941578692697, 720575941633779387] #walk 2 (dnp09)
        # neurons_to_activate = [720575941540215501, 720575941459097503 ] #giant fiber??
    elif dataset_name == "mbanc" or dataset_name == "mbanc-no-optic":
        neurons_to_activate = [10045, 10056] #walk
        # neurons_to_activate = [13805, 230783] #walk 2 (dng97)
        # neurons_to_activate = [10783, 11177] #walk 2 (dnp09)
        # neurons_to_activate = [] #giant fiber??

        # neu_info = pd.read_feather('./data/body-annotations-male-cns-v0.9-minconf-0.5.feather')
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
    color_map.update(setup_color_map(dataset_name))
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

    fig, ax = plt.subplots()

    walk_data, walk_data_timestep = gymtest.get_walk_data()

    #pre_index, post_index
    jnp_con = jnp.array(df_con.to_numpy()[..., [2, 3]])
    jnp_strengths = jnp.array(df_con.to_numpy()[..., 6], dtype=jnp.float32)
    neu_empty = jnp.full(len(df_neu), 0.0)

    lr = .005
    lr2 = .1
    while True:
        spike_times, spikes, last_obs, voltages, joint_records = run_sim(df_neu, df_con, neurons_to_activate, learned_params, neuron_sim, mjc_sim, pygame_process)
        print("voltages:", voltages.shape)

        print("number of spikes:", len(spikes))
        print("unique spikes:", len(spike_stats(spikes)))

        joint_diff = get_joint_diff(walk_data, walk_data_timestep, joint_records, mjc_sim.timestep)
        rewards = get_reward_from_joint_diff(joint_diff, mjc_sim.timestep, dataset_name, runtime)

        draw_the_graphs = True
        if draw_the_graphs:
            forward_result = forward(jnp_con, jnp_strengths, neu_empty, learned_params["syn_weight_mods"], learned_params["neu_weight_mods"], neurons_to_activate)
            weights = jnp.array(weight_by_time_and_voltage(voltages, rewards[16][1]))
            forward_result2 = forward(jnp_con, jnp_strengths, weights, learned_params["syn_weight_mods"], learned_params["neu_weight_mods"], neurons_to_activate)

            print("sum", jnp.sum(forward_result))
            print("sum2", jnp.sum(forward_result2))
            spikes_within_time = []
            for i, time in enumerate(spike_times):
                if time > rewards[16][1] and time < rewards[16][1] + .01:
                    spikes_within_time.append(spikes[i])
            draw_some_graphs(spikes_within_time, forward_result, df_neu, neurons_to_activate, figure=3)
            draw_some_graphs(spikes_within_time, forward_result2, df_neu, neurons_to_activate)

        def thread_func(reward_and_time):
            reward, time = reward_and_time
            v_weights = jnp.array(weight_by_time_and_voltage(voltages, time))
            print("voltage weights", np.sort(v_weights))
            gradient, gradient_neu = get_gradient(df_neu, jnp_con, jnp_strengths, learned_params, neurons_to_activate, neurons_to_push=reward, neu_weight_mods=v_weights)
            return gradient, gradient_neu

        with ThreadPoolExecutor(max_workers=12) as executor:
            print("making", len(rewards), "tasks")
            gradients = list(executor.map(thread_func, rewards))

        gradient = np.stack([x[0] for x in gradients]).sum()
        gradient_neu = np.stack([x[1] for x in gradients]).sum()
        norm = jnp.linalg.norm(gradient)
        norm_neu = jnp.linalg.norm(gradient_neu)
        print(f"gradient sizes {norm} {norm_neu}")
        if norm > 0 and norm_neu > 0:
            learned_params["syn_weight_mods"] = (learned_params["syn_weight_mods"] + np.array(gradient / norm) * .1 * lr).clip(.5, 4)
            learned_params["neu_weight_mods"] = (learned_params["neu_weight_mods"] + np.array(gradient_neu / norm_neu) * 1 * lr).clip(.2, 10)

        # pygame_spike_queue.put(None)
        # pygame_spike_queue.put((spikes[-1][0], spikes))

        reward = get_reward(spikes, runtime, excluded_neurons=neurons_to_activate, dataset_name= dataset_name)
        gradient, neu_gradient = get_gradient(df_neu, jnp_con, jnp_strengths, learned_params, neurons_to_activate, neurons_to_push=reward)
        #TODO some sort of penalty for being further from the data besides just clamping it
        learned_params["syn_weight_mods"] = (learned_params["syn_weight_mods"] + np.array(gradient) * .1 * lr2).clip(.2, 4)
        learned_params["neu_weight_mods"] = (learned_params["neu_weight_mods"] + np.array(neu_gradient) * 1 * lr2).clip(.1, 30)
        print("sorted learned params", np.sort(learned_params["syn_weight_mods"]), np.sort(learned_params["neu_weight_mods"]))

        # print("gradients", gradient, neu_gradient)

        # reward_time1, reward_time2 = get_reward_at_time(spikes, spike_times, voltages, runtime, excluded_neurons=neurons_to_activate, dataset_name= dataset_name)

        # v_weights = weight_by_time_and_voltage(voltages, .05)
        # neurons_to_push = {x[1]: 2.6 for x in neuron_groups.mbanc_by_leg["rf"]}
        # neurons_to_push = reward_time1
        # v_gradient, v_neu_gradient = get_gradient(df_neu, df_con, learned_params, neurons_to_activate, neurons_to_push, neu_weight_mods = v_weights)

        # v_weights = weight_by_time_and_voltage(voltages, .1)
        # neurons_to_push = {x[1]: -.6 for x in neuron_groups.mbanc_by_leg["rf"]}
        # neurons_to_push = reward_time2
        # v_gradient_2, v_neu_gradient_2 = get_gradient(df_neu, df_con, learned_params, neurons_to_activate, neurons_to_push, neu_weight_mods = v_weights)

        # learned_params["syn_weight_mods"] = (learned_params["syn_weight_mods"] + np.array(v_gradient + v_gradient_2) * .1 * lr).clip(.2, 4)
        # learned_params["neu_weight_mods"] = (learned_params["neu_weight_mods"] + np.array(v_neu_gradient + v_neu_gradient_2) * 1 * lr).clip(.1, 30)

        # print("supposed gradient", np.sort(gradient))
        # print("learned weights syn", np.sort(learned_params["syn_weight_mods"]))
        # print("learned weights neu", np.sort(learned_params["neu_weight_mods"]))

def draw_some_graphs(spikes, forward_result, df_neu, neurons_to_activate, figure=1):
    flyid2i = {j: i for i, j in enumerate(df_neu.index)}  # flywire id: brian ID
    i2flyid = {j: i for i, j in flyid2i.items()} # brian ID: flywire ID

    counter = defaultdict(int)
    for spike_id in spikes:
        if spike_id not in neurons_to_activate:
            counter[spike_id] += 1
    sorted_items = sorted([x for x in (counter).items() if x[1] > 0], key=lambda v: -v[1])
    # print("sorted items", sorted_items)
    # print("forward_result", forward_result)
    forward_dict = {i2flyid[i]: x for i, x in enumerate(forward_result)}
    forwards = [forward_dict[x[0]] for x in sorted_items]
    plt.figure(figure)
    plt.ion()
    plt.clf()
    plt.subplot(2, 1, 2)
    plt.plot([x[1]/10 for x in sorted_items])
    plt.plot([x for x in forwards])

    plt.subplot(2, 1, 1)
    forwards = np.array(list(filter(lambda v: v[1] > .05, forward_dict.items())))
    print("fs now numpied", forward)
    forward_sorted = forwards[forwards[:, 1].argsort()[::-1]]
    simmed = [counter[x[0]]/10 if x[0] in counter else 0 for x in forward_sorted]
    plt.plot(simmed)
    plt.plot([x[1] for x in forward_sorted])

    plt.show()
    print("start pause")
    plt.pause(.2)
    print("end pause")

def jprint(x):
    # jax.debug.print(str(x))
    pass

def get_gradient(df_neu, jnp_con, jnp_strengths, learned_params, neurons_to_activate, neurons_to_push, neu_weight_mods = None):
    flyid2i = {j: i for i, j in enumerate(df_neu.index)}  # flywire id: brian ID
    # i2flyid = {j: i for i, j in flyid2i.items()} # brian ID: flywire ID

    neurons_to_activate = jnp.array([flyid2i[x] for x in neurons_to_activate], dtype=jnp.int32)
    neurons_to_push_in = neurons_to_push
    neurons_to_push = jnp.array([flyid2i[nid] for nid in neurons_to_push_in.keys()], dtype=jnp.int32)
    neurons_to_push_weights = jnp.array([weight for weight in neurons_to_push_in.values()])

    # all_neurons = forward(jnp_con, jnp_strengths, neurons_to_activate)

    learned_syn_weights = learned_params["syn_weight_mods"] 
    learned_neu_weights = learned_params["neu_weight_mods"]

    if neu_weight_mods is None:
        neu_weight_mods = jnp.full(len(df_neu), 0.)

    # start = time.monotonic()
    gradient, neu_gradient = jit_grad_loss(jnp_con, jnp_strengths, neu_weight_mods, learned_syn_weights, learned_neu_weights, neurons_to_activate, neurons_to_push, neurons_to_push_weights) 
    # end = time.monotonic()
    # print("jax took", end - start, "seconds")
    # print("gradient", gradient.sort(), neu_gradient.sort())
    print("jaxing")

    return gradient, neu_gradient

def forward(jnp_con, start_synapse_weights, neu_weights, learned_syn_weights, learned_neu_weights, neurons_to_activate):
    # global_connection_strength = .001

    all_neurons = jnp.zeros(len(learned_neu_weights))
    # old way of using neu_weights
    # all_neurons = neu_weights
    all_neurons = all_neurons.at[jnp.array(neurons_to_activate)].set(1)
    # all_neurons_orig = all_neurons

    total_neu_weights = learned_neu_weights * (neu_weights + 1)

    # jprint(all_neurons[jnp_con[..., 0]].shape)
    # jprint(start_synapse_weights.shape)
    # start_synapse_weights = start_synapse_weights.clip(min=-100, max=100)
    # synapse_weights = jnp.tanh(learned_syn_weights * start_synapse_weights / 1000)
    synapse_weights = start_synapse_weights * learned_syn_weights

    outputs = jnp.zeros_like(all_neurons)

    # jprint(jax.numpy.sum(all_neurons_orig))
    for i in range(6):
        pre_synapse_strengths = (all_neurons[jnp_con[..., 0]]).clip(min=0, max=1)
        synapse_strenghts: jnp.ndarray = (pre_synapse_strengths * synapse_weights)
        # jax.debug.print("i pre_synapse_strengths * start_synapse_weights * learned_weights = step1 {}, {}, {}, {}, {}", i, pre_synapse_strengths, start_synapse_weights, learned_weights, synapse_strenghts)
        neuron_updates = jnp.zeros_like(all_neurons).at[jnp_con[..., 1]].add(synapse_strenghts) * total_neu_weights

        # all_neurons += jnp.tanh((neuron_updates)/1000)#.clip(min=0)
        # all_neurons += jnp.tanh((neuron_updates - 100)/250)#.clip(min=0)
        # all_neurons += jnp.log2((neuron_updates * 100 - 12.5).clip(min = 1, max=100)) * 15 / 100
        # all_neurons += jnp.log2((neuron_updates * 100 - 12.5).clip(min = 1, max=100)) * 15 / 100
        # neuron_updates *= 1
        e1 = (neuron_updates - 12.5).clip(min = 1)
        e2 = jnp.log2(e1) * 10
        # e3 = jnp.maximum(e2, (neuron_updates / 10))
        # e3 = e2.at[jnp.where(e2 == 0)].add(neuron_updates / 10)
        e3 = jax.lax.select(e2 == 0, neuron_updates / 10, e2)
        e4 = e3.clip(max=100) / 100
        # scaled_neuron_updates = jnp.maximum(jnp.log2(e1) * 10, neuron_updates).clip(max=100) / 100
        # jax.debug.print("all neurons pre update {}", jnp.sort(all_neurons))
        all_neurons = e4
        outputs += e4 / (i+1)
        # all_neurons += ((neuron_updates)/1000) / (i + 1)

        all_neurons = all_neurons.clip(min=0, max=1)
        all_neurons = all_neurons.at[jnp.array(neurons_to_activate)].set(1)

        # jax.debug.print("all neurons {} {} {} {}", jnp.sort(all_neurons), jnp.sort(neuron_updates), jnp.mean(neuron_updates), jnp.median(neuron_updates))
        # jax.debug.print("scaled {}", jnp.sort(scaled_neuron_updates))
        # jax.debug.print("{} {} {} e4: {}", jnp.sort(e1), jnp.sort(e2), jnp.sort(e3), jnp.sort(e4))
        # jax.debug.print("non zero {}", jnp.count_nonzero(all_neurons))

    return outputs

def loss(jnp_con, jnp_strengths, neuron_weights, learned_syn_weights, learned_neu_weights, neurons_to_activate, neurons_to_push, neurons_to_push_weights):
    new_neuron_values = forward(jnp_con, jnp_strengths, neuron_weights, learned_syn_weights, learned_neu_weights, neurons_to_activate)
    act_str = jnp.abs(jnp.sum(new_neuron_values) - 50000.) * .01

    return jax.numpy.sum(neurons_to_push_weights * new_neuron_values[neurons_to_push]) - act_str

def activation_strength(jnp_con, jnp_strengths, neuron_weights, learned_syn_weights, learned_neu_weights, neurons_to_activate):
    new_neuron_values = forward(jnp_con, jnp_strengths, neuron_weights, learned_syn_weights, learned_neu_weights, neurons_to_activate)
    return jnp.sum(new_neuron_values)

def jax_get_gradient(jnp_con, jnp_strengths, neu_weight_mods, learned_syn_weights, learned_neu_weights, neurons_to_activate, neurons_to_push, neurons_to_push_weights): 
    grad1_syn, grad1_neu = jax.grad(loss, argnums=(3, 4))(jnp_con, jnp_strengths, neu_weight_mods, learned_syn_weights, learned_neu_weights, neurons_to_activate, neurons_to_push, neurons_to_push_weights)
    grad2_syn, grad2_neu = jax.grad(activation_strength, argnums=(3, 4))(jnp_con, jnp_strengths, neu_weight_mods, learned_syn_weights, learned_neu_weights, neurons_to_activate)

    def thing(grad1, grad2):
        vec = grad2 * (jnp.dot(grad1, grad2) / jnp.dot(grad2, grad2))
        return grad1 - vec

    # return grad1 - jnp.dot(grad1, grad2) * normed
    return thing(grad1_syn, grad2_syn), thing(grad1_neu, grad2_neu)
    # return -grad2_syn * .001, -grad2_neu * .001

# jit_grad_loss = jax.jit(jax_get_gradient)
jit_grad_loss = jax.jit(jax.grad(loss, argnums=(3, 4)))

def get_leg_neurons(dataset_name):
    all_leg_neurons = set()
    # groups = neuron_groups.mbanc_leg_neuron_groups
    # for neurons in groups.values():
        # all_leg_neurons.update(neurons)

    # all_leg_neurons.update(neuron_groups.mbanc_lf_leg_neurons)
    if dataset_name == "mbanc" or dataset_name == "mbanc-no-optic":
        all_leg_neurons.update(neuron_groups.mbanc_leg_neurons)
        neurons_by_leg = neuron_groups.mbanc_by_leg
    elif dataset_name == "banc":
        all_leg_neurons.update(neuron_groups.banc_leg_neurons)
        neurons_by_leg = neuron_groups.banc_by_leg
    else:
        raise Exception("invalid dataset")
    return all_leg_neurons, neurons_by_leg

def weight_by_time_and_voltage(voltages, time):
    # maybe it should be recent spikes and not voltages at all?

    samples_per_second = 1000
    sample = int(time * samples_per_second)
    start_sample = max(int((time - .02) * samples_per_second), 0)
    print("sample at", start_sample, sample)
    voltages_in_window = voltages[:, start_sample:sample]
    if sample == 0 and start_sample == 0:
        weight = np.zeros_like(voltages[:, 0])
    else:
        weight = np.mean(voltages[:, start_sample:sample] / mV, axis=1)
    if np.isnan(weight).any():
        print(weight)
        raise Exception("array had a nan")
    print("getting weights by time", time, weight, weight.shape, voltages_in_window, len(voltages_in_window))
    # weight = 10 ** (weight / 7) 
    weight = weight/7
    return weight

def get_joint_diff(walk_data, walk_data_timestep, joint_records, mjc_timestep):
    print("joint records", joint_records.shape, joint_records)
    print("walk data", walk_data)
    mjc_times = np.arange(joint_records.shape[1]) * mjc_timestep
    walk_data_times = np.arange(walk_data.shape[1]) * walk_data_timestep
    result = np.zeros_like(joint_records)
    interp_walk_data = np.zeros_like(joint_records)
    zero_pose = gymtest.get_starting_pose2() # get from parameter?

    names = flygym_preprogrammed.all_leg_dofs
    plt.ion()
    plt.figure(0)
    plt.clf()
    plt.show()
    for i, record in enumerate(joint_records):
        interp_walk_joint = np.interp(mjc_times, walk_data_times, walk_data[i])
        result[i] = (record - interp_walk_joint)
        interp_walk_data[i] = interp_walk_joint - zero_pose[i]

        plt.subplot(7, 7, i + 1)
        plt.plot(mjc_times, joint_records[i], color="black")
        plt.plot(walk_data_times, walk_data[i], color="green")
        plt.plot(mjc_times, result[i], color="blue")
        plt.title(str(i) + " " + str(names[i]), fontsize=7)

    print("result", result, "walk data", interp_walk_data)

    plt.tight_layout()
    plt.pause(.1)
    plt.draw()
    plt.pause(.1)
    
    return result, interp_walk_data

def get_reward_from_joint_diff(joint_diff, mjc_timestep, dataset_name, runtime):
    if dataset_name == "banc":
        leg_neuron_groups = neuron_groups.banc_leg_neuron_groups
    elif dataset_name == "mbanc" or dataset_name == "mbanc-no-optic":
        leg_neuron_groups = neuron_groups.mbanc_leg_neuron_groups
    else:
        raise Exception("invalid dataset")

    # TODO consolidate these functions
    neuron_to_index = gymtest.get_neuron_to_muscle_index(leg_neuron_groups, gymtest.get_muscle_to_gym_muscle_index(leg_neuron_groups))
    index_to_neuron = gymtest.get_muscle_index_to_neuron(neuron_to_index)

    rewards = []

    num_seconds = runtime / 1000
    per_second = 100

    joint_diff, original_walk_data = joint_diff

    names = flygym_preprogrammed.all_leg_dofs

    plot = np.zeros((joint_diff.shape[0], int(num_seconds * per_second)))
    for i in range(int(num_seconds * per_second)):
        time = i / per_second
        diffs = np.zeros(joint_diff.shape[0])
        reward = {}
        for joint_i in range(joint_diff.shape[0]):
            mjc_times = np.arange(joint_diff.shape[1]) * mjc_timestep
            diff = np.interp(time, mjc_times, joint_diff[joint_i])
            orig = np.interp(time, mjc_times, original_walk_data[joint_i])
            diffs[joint_i] = diff

            plot[joint_i, i] = diff
            neurons = index_to_neuron[joint_i]
            for neuron, strength in neurons:
                if strength * orig > 0: # 
                    lr = 1
                elif (diff + orig) * orig < -.01:
                    lr = 1
                else: 
                    lr = 0
                # if "RMTibia" not in names[neuron_to_index[neuron][0]]:
                    # lr = 0
                reward[neuron] = strength * -diff * lr
                # print(f"doing a reward, {neuron}, {names[neuron_to_index[neuron][0]]}, {reward[neuron]}, {strength} {diff} {orig}, {(diff + orig) * orig}")
                # print(joint_i, time, neuron, reward[neuron] if neuron in reward else "not in reward")
            if len(neurons) > 0:
                pass
                # break

        rewards.append((reward, time))

    # plt.figure(2)
    # plt.plot([i[1] for i in rewards], [i[0] for i in rewards])
    # plt.show()

    print("rewards", [([(names[neuron_to_index[j][0]], j, i[0][j]) for j in i[0] if i[0][j] > 0], i[1]) for i in rewards])
    first_reward = rewards[0][0]
    print(first_reward)
    rewards1 = {(names[neuron_to_index[n][0]], neuron_to_index[n][1]): n for (n, v) in first_reward.items()}
    print(rewards1)

    plt.figure(4)
    plt.clf()
    for i, name in enumerate(rewards1.keys()):
        n = rewards1[name]
        plt.subplot(7, 7, i + 1)
        plt.plot([x[1] for x in rewards], [x[0][n] if n in x[0] else 0 for x in rewards])
        plt.title(str(name), fontsize=7)

    plt.tight_layout()

    plt.show()
    plt.pause(.1)

    return rewards

def get_reward_at_time(spikes, spike_times, voltages, runtime, excluded_neurons, dataset_name, time = .1):
    all_leg_neurons, neurons_by_leg = get_leg_neurons(dataset_name)
    
    spike1 = {neuron[1]: 0 for neuron in neurons_by_leg["rf"]}
    spike2 = {neuron[1]: 0 for neuron in neurons_by_leg["rf"]}
    for i, spike_id in enumerate(spikes):
        time = spike_times[i]
        # print(i, spike_id, time, list(spike1.keys())[0])
        if spike_id in spike1 or spike_id in spike2:
            # print("spike", spike_id)
            if time > 0 and time < .1:
                spike1[spike_id] += 1
            if time > .1 and time < .15:
                spike2[spike_id] += 1

    average1 = sum(spike1.values()) / len(spike1)
    target1 = 0
    res1 = {}
    for spike_id in spike1.keys():
        res1[spike_id] = target1 - average1

    average2 = sum(spike2.values()) / len(spike2)
    target2 = 2 * (runtime / 1000)
    res2 = {}
    for spike_id in spike2.keys():
        res2[spike_id] = target2 - average2

    print("timed reward values:", target1 - average1, target2 - average2)

    return res1, res2

def get_reward(spikes, runtime, excluded_neurons, dataset_name):
    all_leg_neurons, neurons_by_leg = get_leg_neurons(dataset_name)

    # _, synapse_map, rev_synapse_map = data.get_synapse_map("mbanc")
    # next_synapses = {}
    # for spike_id in all_leg_neurons:
        # for pre_spike_id, strength in rev_synapse_map[spike_id]:
            # next_synapses[pre_spike_id] = (strength, spike_id)

    counter = defaultdict(float)
    for neuron in all_leg_neurons:
        counter[neuron] = 0
    other_counter = defaultdict(float)
    leg_counters = {name: {neuron[1]: 0 for neuron in neurons_by_leg[name]} for name in neuron_groups.legs}
    # weak_counter = defaultdict(float)
    for spike_id in spikes:
        if spike_id in all_leg_neurons:
            counter[spike_id] += 1
        elif spike_id not in excluded_neurons:
            other_counter[spike_id] += 1
        for leg in neuron_groups.legs:
            if spike_id in leg_counters[leg].keys():
                leg_counters[leg][spike_id] += 1


    # result = {key: (13 - value)**2 * (1 if (13-value) > 0 else -1) for key, value in counter.items()} 
    result = {}
    # result = {key: (5 * (runtime / 1000) - value) for key, value in counter.items()} 
    # for n in result:
        # if counter[n] < 5 and counter[n] > 1:
            # result[n] = 0
    result.update({key: -10 * value * (runtime / 1000) for key, value in other_counter.items() if value > 150 * (runtime / 1000)})

    # maybe averaging them will do something?
    # average = sum(result.values()) / len(result)
    # for key in result.keys():
        # result[key] = average

    # result = {}
    for leg in leg_counters.keys():
        leg_counters[leg]
        average = sum(leg_counters[leg].values()) / len(leg_counters[leg])
        target = 8 * (runtime / 1000)
        for key in leg_counters[leg].keys():
            result[key] = target - average

    # print(list(map(lambda f: f"{f}, {type(f)}", other_counter.keys())))
    # max_other_neurons = 6
    # result.update({key: -1 * (value - max_other_neurons) if value > max_other_neurons else 0 for key, value in other_counter.items()})

    # result = {key: 100 for key, value in counter.items()} 

    print("1x result:", result)

    return result
    # return sum([math.sqrt(x) for x in counter.values()])

def setup_color_map(dataset_name):
    color_map = {}
    for leg in neuron_groups.legs:
        if dataset_name == "banc":
            neurons = neuron_groups.banc_by_leg[leg]
        if dataset_name == "mbanc" or dataset_name == "mbanc-no-optic":
            neurons = neuron_groups.mbanc_by_leg[leg]
        leg_index = (neuron_groups.legs.index(leg) * 5) % 7
        color_1 = (200 - leg_index * 25, 100 + leg_index * 25, 255)
        color_2 = (150 - leg_index * 25, 80 + leg_index * 25, 200)
        color_3 = (200 - leg_index * 25, 255, 100 + leg_index * 25)
        color_4 = (150 - leg_index * 25, 200, 80 + leg_index * 25)
        for i, (key, neuron) in enumerate(neurons):
            # print(key, neuron)
            if "extensor" in key or "levetator" in key:
                if i % 2 == 0:
                    color_map[neuron] = color_1
                else:
                    color_map[neuron] = color_2
            else:
                if i % 2 == 0:
                    color_map[neuron] = color_3
                else:
                    color_map[neuron] = color_4
        
    color_map[neuron_groups.mbanc_leg_neuron_groups["rf_trochanter_extensor"][0]] = (180, 180, 180)

    target_neurons = [
        720575941669833905,
        720575941560438643,
        720575941504247575,
        720575941527002649,
        720575941554038555,
        720575941552245822,

        720575941469024064,720575941487873488,720575941545638505,720575941625121674,720575941437338412,720575941519468654
    ]
    for i, n in enumerate(target_neurons):
        if i % 2 == 0:
            color_map[n] = (255, 255, 255)
        else:
            color_map[n] = (200, 200, 200)
    red_neurons = [720575941496703104,720575941503572133,720575941589995319,720575941441065919,720575941544954556,720575941687554799,
    ]
    for i, n in enumerate(red_neurons):
        if i % 2 == 0:
            color_map[n] = (255, 100, 100)
        else:
            color_map[n] = (200, 100, 100)

    yellow_neurons = [720575941569601650,720575941515421443,720575941555553363,720575941414465684,720575941461249747,720575941649246741]
    for i, n in enumerate(yellow_neurons):
        if i % 2 == 0:
            color_map[n] = (255, 255, 100)
        else:
            color_map[n] = (200, 200, 100)

    purple_neurons = [720575941626500746]
    for i, n in enumerate(purple_neurons):
        if i % 2 == 0:
            color_map[n] = (255, 100, 255)
        else:
            color_map[n] = (200, 100, 200)
    return color_map


def spike_stats(spikes):
    spike_counter = defaultdict(int)
    for spike in spikes:
        spike_counter[spike] += 1
    return spike_counter

if __name__ == "__main__":
    wrapper_thing()
    df_neu, df_con = data.load("mbanc")
    # weights = jnp.full(len(df_con), 1.0)
    # get_gradient(df_neu, df_con, weights)
