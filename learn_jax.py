from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
import dataclasses
import time
import brian2
from brian2.units import ms
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import data
import jax_spiking_model
from neuron_model import NeuronSim
from plt_thread import ThreadedPlot
import pygame_loop

def main():
    _, orig_df_con = data.load("mbanc")
    neuron_count = 1000
    syn_count = 40_000
    df_neu = pd.DataFrame(range(neuron_count))
    df_con = pd.DataFrame(range(syn_count))
    rand1 = np.random.default_rng(seed=11)
    pre = rand1.integers(neuron_count, size=syn_count)
    post = rand1.integers(neuron_count, size=syn_count)
    df_con.insert(0, "Presynaptic_Index", pre)
    df_con.insert(1, "Postsynaptic_Index", post)

    print(orig_df_con)
    orig_strs = orig_df_con["Excitatory x Connectivity"].to_numpy()
    str_indexes = rand1.integers(len(orig_strs), size=syn_count)
    strs = orig_strs[str_indexes] * 4.5
    df_con.insert(2, "Excitatory x Connectivity", strs)

    neurons_to_activate = list(range(20))

    runtime = 120
    sim = NeuronSim(df_neu, df_con, "", neurons_to_activate, runtime * ms)
    sim.voltage_resolution = .1 * ms

    pygame_process = pygame_loop.PygameProcess()

    sim.start({})
    while(True):
        spikes = sim.spike_queue.get()
        if spikes == None:
            break

        update_time, spikes = spikes
        sim.input_queue.put(())

        times = np.full(len(spikes), update_time)
        spike_and_times = np.stack((times, spikes), 1, dtype=object)
        pygame_process.add_spikes((update_time, spike_and_times))

    def count_spikes(spike_i):
        counts = defaultdict(int)
        for spike in spike_i:
            counts[spike] += 1
        return counts

    def partition_spikes(spike_t, spike_i, per_second):
        partitions = defaultdict(lambda: defaultdict(int))
        this_partition = 0
        delta = 1000/per_second
        for i in range(len(spike_t)):
            time_ms = spike_t[i] / ms
            sid = spike_i[i]
            if time_ms > this_partition + delta:
                this_partition += delta
            
            partitions[this_partition / 1000][sid] += 1

        return partitions
            
    partitions_per_second = 500
    first_spikes_t, first_spikes_i = sim.spikes
    first_voltages = sim.voltages.T
    print("first spikes", first_spikes_i, first_spikes_t)
    first_partitions = partition_spikes(first_spikes_t, first_spikes_i, partitions_per_second)
    first_spike_count = count_spikes(first_spikes_i)

    jnp_con = jnp.array(df_con.to_numpy()[..., [0, 1]], dtype=int)
    jnp_strengths = jnp.array(df_con.to_numpy()[..., 2], dtype=jnp.float32)
    neu_empty = jnp.full(len(df_neu), 0.0)

    plt = ThreadedPlot()
    # plt.ion()

    loss1 = []
    loss2 = []
    loss3 = []
    loss4 = []

    rand2 = np.random.default_rng(seed=54)
    learned_params = {
        # "neu_weight_mods": rand2.random(neuron_count) + .3,
        "neu_weight_mods": np.ones(neuron_count),
        # "syn_weight_mods": rand2.random(syn_count) + .5,
        "syn_weight_mods": rand2.random(syn_count) * .02 + .99,
    }
    i = 0
    loss = None
    lrs = .01
    lrn = .001

    params = dataclasses.replace(jax_spiking_model.default_params, steps=runtime * 10)

    while(True):
        i += 1
        pygame_process.reset()
        sim.start(learned_params)

        while(True):
            spikes = sim.spike_queue.get()
            if spikes == None:
                break

            update_time, spikes = spikes
            sim.input_queue.put(())

            times = np.full(len(spikes), update_time)
            spike_and_times = np.stack((times, spikes), 1, dtype=object)
            pygame_process.add_spikes((update_time, spike_and_times))

        cur_spikes_t, cur_spikes_i = sim.spikes

        cur_spike_count = count_spikes(cur_spikes_i)
        # print("cur spike count", cur_spike_count)

        voltages = sim.voltages
        # print("voltages", voltages.shape)
        # print("sum", np.sum(voltages.shape))
        # print(voltages)
        # print("times", first_partitions.keys())
        this_partitions = partition_spikes(cur_spikes_t, cur_spikes_i, partitions_per_second)

        target_voltages = jnp.array(first_voltages)
        
        syn_weights = jnp_strengths * learned_params["syn_weight_mods"]
        last_loss = loss if loss is not None else 10000000
        # print("getting grads")
        grads, = jax_spiking_model.get_sim_grads(params, jnp_con, neuron_count, jnp.array(syn_weights), jnp.array(neurons_to_activate), target_voltages)
        # print("grads", grads)
        gradient = grads

        # print("done getting grads")
        # print("gradient:", grads, gradient)
        # print("grad norm", jnp.linalg.norm(gradient))
        # gradient_neu = np.stack([x[1] for x in gradients]).sum()

        # norm = jnp.linalg.norm(gradient)
        # norm_neu = jnp.linalg.norm(gradient_neu)
        # print("norms", norm, norm_neu)
        norm =  1
        # norm_neu = 1
        if norm > 0:
            new_syn_weights = (learned_params["syn_weight_mods"] - np.array(gradient / norm) * lrs)#.clip(.2, 4)
            loss = jax_spiking_model.sim_loss(params, jnp_con, neuron_count, jnp_strengths * new_syn_weights, jnp.array(neurons_to_activate), target_voltages)
            print("losses:", loss, last_loss)
            learned_params["syn_weight_mods"] = new_syn_weights
        # if norm_neu > 0:
            # learned_params["neu_weight_mods"] = (learned_params["neu_weight_mods"] + np.array(gradient_neu / norm_neu) * lrn)#.clip(.1, 30)
        # diffs = {x: (first_spike_count[x] - cur_spike_count[x]) for x in (list(first_spike_count.keys()))}
        # diffs[50] = 1
        
        """
        gradient, neu_gradient = learn.get_gradient(df_neu, jnp_con, jnp_strengths, learned_params, neurons_to_activate, diffs, neu_empty)
        norm = jnp.linalg.norm(gradient)
        norm_neu = jnp.linalg.norm(neu_gradient)
        if norm > 0:
            learned_params["syn_weight_mods"] = (learned_params["syn_weight_mods"] + np.array(gradient / norm) * .1 * lrs).clip(.2, 4)
        if norm_neu > 0:
            learned_params["neu_weight_mods"] = (learned_params["neu_weight_mods"] + np.array(neu_gradient / norm_neu) * 1 * lrn).clip(.1, 30)
        """

        keys = cur_spike_count.keys()
        this_loss4 = 0
        p0 = partition_spikes(first_spikes_t, first_spikes_i, 100)
        p1 = partition_spikes(cur_spikes_t, cur_spikes_i, 100)
        for time in p0.keys():
            current_partition = this_partitions[time]
            all_sids = current_partition.keys() | first_partitions[time].keys()
            diffs = sum([first_partitions[time][sid] - current_partition[sid] for sid in all_sids])
            this_loss4 += diffs
        this_loss1 = np.linalg.norm(learned_params["syn_weight_mods"] - np.ones_like(learned_params["syn_weight_mods"]))
        this_loss2 = np.array(loss)
        this_loss3 = np.linalg.norm(np.array([cur_spike_count[x] for x in keys]) - np.array([first_spike_count[x] for x in keys]))
        # print("syn weight diff", this_loss1)
        # print("spike diff", this_loss3)
        # print("advanced spike diff", this_loss4)
        loss1.append(this_loss1)
        loss2.append(this_loss2)
        loss3.append(this_loss3)
        loss4.append(this_loss4)

        plt.figure(1)
        plt.clf()
        # print("losses", loss1, loss2, loss3)
        # print("losses", loss1[0], loss2[0], loss3[0])
        # print("losses", loss1 / loss1[0], loss2 / loss2[0], loss3 / loss3[0])
        plt.plot(loss1 / loss1[0], color="black")
        plt.plot(loss2 / loss2[0], color="red")
        plt.figure(2)
        plt.clf()
        plt.plot(loss3 / loss3[0], color="blue")
        plt.plot(np.array(loss4) / loss4[0], color="green")

def main2():
    _, orig_df_con = data.load("mbanc")
    neuron_count = 10
    syn_count = 40_000
    rand = np.random.default_rng(seed=47)
    # pre = rand.integers(neuron_count, size=syn_count)
    # post = rand.integers(neuron_count, size=syn_count)
    # jnp_con = jnp.stack((pre, post), axis=1)

    # orig_strs = orig_df_con["Excitatory x Connectivity"].to_numpy()
    # str_indexes = rand.integers(len(orig_strs), size=syn_count)
    # start_strs = orig_strs[str_indexes] * 4.5
    # neurons_to_activate = list(range(20))

    jnp_con = jnp.array([[0, 1], [1, 2]])
    start_strs = jnp.array([420, 420])
    neurons_to_activate = [0]
    syn_count = 1

    runtime = 1000
    params = dataclasses.replace(jax_spiking_model.default_params, steps=runtime * 10)

    target_voltages, target_refractory_timers, target_rise_values = jax_spiking_model.run_sim(params, jnp_con, neuron_count, start_strs, jnp.array(neurons_to_activate))

    plt = ThreadedPlot()

    # syn_weight_mods = jnp.array(rand.random(syn_count) * .2 + .9)
    syn_weight_mods = jnp.array([1.3, 1.3])

    all_spikes = jax.jit(lambda a: jnp.where(a >= params.threshold, size=10000))(target_voltages)
    spike_times = all_spikes[0] / 10000
    spike_ixs = all_spikes[1]
    all_spikes_target = jnp.stack((spike_times, spike_ixs), axis=1).at[:, 1].add(neuron_count).tolist()

    pygame_process = pygame_loop.PygameProcess(unit_height=10)
    def draw_pygame(v):
        all_spikes = jax.jit(lambda a: jnp.where(a >= params.threshold, size=10000))(v)
        spike_times = all_spikes[0] / 10000
        spike_ixs = all_spikes[1]
        all_spikes = jnp.stack((spike_times, spike_ixs), axis=1).tolist()
        end_time = params.steps / 10000
        pygame_process.reset()
        pygame_process.add_spikes((end_time, all_spikes_target))
        pygame_process.add_spikes((end_time, all_spikes))

    draw_pygame(target_voltages)

    lr = 200
    draw1 = []
    draw2 = []
    draw3 = []
    draw4 = []
    while True:
        strs = start_strs * syn_weight_mods

        voltages, refractory_timers, rise_values = jax_spiking_model.run_sim(params, jnp_con, neuron_count, strs, jnp.array(neurons_to_activate))
        print("weights", syn_weight_mods)

        # real_target = jnp.maximum(target_voltages, voltages)
        # real_target = (target_voltages) * ((voltages != 0) & (target_voltages != 0)) + voltages * ((voltages == 0) & (target_voltages == 0)) + params.threshold * ((target_voltages == 0) & (voltages != 0))
        index = 2

        spikes = (target_voltages >= params.threshold) | (voltages >= params.threshold)
        real_target_all_neurons = jnp.where((target_voltages >= params.threshold), voltages + params.threshold, voltages)
        real_target_all_neurons = jnp.where((voltages >= params.threshold), voltages - params.threshold, real_target_all_neurons)
        real_target = voltages.at[:, index].set(real_target_all_neurons[:, index])

        # rise_target = rise_values.at[:, index].set(target_rise_values[:, index])
        # rise_target = (rise_target) * ((rise_target != 0) & (target_rise_values != 0)) + rise_values * ((rise_target == 0) | (target_rise_values == 0))

        loss = jnp.sum((real_target - voltages) ** 2)
        # loss = jnp.sum((rise_target - rise_values))

        grad, = jax_spiking_model.get_sim_grads(params, jnp_con, neuron_count, strs, jnp.array(neurons_to_activate), real_target)
        # grad, = jax_spiking_model.get_sim_rise_grads(params, jnp_con, neuron_count, strs, jnp.array(neurons_to_activate), rise_target)

        syn_weight_mods -= grad * lr
        """#smoothly go to global maximum
        if jnp.linalg.norm(syn_weight_mods - 1) < .01:
            syn_weight_mods = jnp.ones_like(strs)
        else:
            syn_weight_mods -= (syn_weight_mods - 1) * .05
        print(jnp.linalg.norm(syn_weight_mods - 1))
        """

        draw_pygame(voltages)

        draw1.append(loss)
        draw2.append(jnp.linalg.norm(syn_weight_mods - jnp.array(1)))
        draw3.append(syn_weight_mods[0])
        draw4.append(syn_weight_mods[1])

        draw1np = np.array(draw1)
        draw2np = np.array(draw2)
        draw3np = np.array(draw3)
        draw4np = np.array(draw4)
        plt.figure(1)
        plt.clf()
        plt.plot(draw1np / draw1np[0], color="blue")
        plt.figure(2)
        plt.clf()
        plt.plot(draw2np / draw2np[0], color="green")
        plt.figure(3)
        plt.clf()
        plt.plot((target_voltages)[:, 2], color = "green")
        plt.plot((voltages)[:, 2], color = "blue")
        plt.figure(4)
        plt.clf()
        plt.plot((real_target)[:, 2], color = "black")
        plt.plot((voltages)[:, 2], color="red")
        # plt.plot((rise_values)[:, 2], color="red")
        plt.figure(5)
        plt.clf()
        plt.plot(draw3np, color="blue")
        plt.plot(draw4np, color="green")
        plt.figure(6)
        plt.clf()
        plt.plot((real_target - voltages)[:, 2], color = "brown")
        plt.plot((real_target - voltages)[:, 1], color = "black")

        # plt.plot((real_target)[:, 2], color = "black")
        # plt.plot((real_target - voltages)[:, 2], color="red")

if __name__ == "__main__":
    # main()
    main2()
