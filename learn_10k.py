from collections import defaultdict
import brian2
from brian2.units import ms
import pandas as pd
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

import data
import learn
from neuron_model import NeuronSim
from plt_thread import ThreadedPlot
import pygame_loop

if __name__ == "__main__":
    _, orig_df_con = data.load("mbanc")
    neuron_count = 100000
    syn_count = 10_000_000
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

    runtime = 300
    sim = NeuronSim(df_neu, df_con, "", neurons_to_activate, runtime * ms)

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
            
    first_spikes_t, first_spikes_i = sim.spikes
    print("first spikes", first_spikes_i, first_spikes_t)
    first_partitions = partition_spikes(first_spikes_t, first_spikes_i, 100)
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
    rand2.random()
    learned_params = {
        "neu_weight_mods": rand2.random(neuron_count) + .3,
        "syn_weight_mods": rand2.random(syn_count) + .3,
    }
    lr2 = 5
    while(True):
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


        total_gradient = np.zeros_like(learned_params["syn_weight_mods"])
        total_neu_gradient = np.zeros_like(learned_params["neu_weight_mods"])

        voltages = sim.voltages
        print("voltages", voltages.shape)
        print("sum", np.sum(voltages.shape))
        print(voltages)
        print("times", first_partitions.keys())
        for time in first_partitions.keys():
            this_partitions = partition_spikes(cur_spikes_t, cur_spikes_i, 100)
            diffs = first_partitions[time]
            weights = learn.weight_by_time_and_voltage(voltages, time)
            gradient, neu_gradient = learn.get_gradient(df_neu, jnp_con, jnp_strengths, learned_params, neurons_to_activate, diffs, neu_empty)
            total_gradient += gradient
            total_neu_gradient += neu_gradient

        norm = jnp.linalg.norm(total_gradient)
        norm_neu = jnp.linalg.norm(total_neu_gradient)
        if norm > 0:
            learned_params["syn_weight_mods"] = (learned_params["syn_weight_mods"] + np.array(total_gradient / norm) * .1 * lr2).clip(.2, 4)
        if norm_neu > 0:
            learned_params["neu_weight_mods"] = (learned_params["neu_weight_mods"] + np.array(total_neu_gradient / norm_neu) * 1 * lr2).clip(.1, 30)
        # diffs = {x: (first_spike_count[x] - cur_spike_count[x]) for x in (list(first_spike_count.keys()))}
        # diffs[50] = 1
        
        """
        gradient, neu_gradient = learn.get_gradient(df_neu, jnp_con, jnp_strengths, learned_params, neurons_to_activate, diffs, neu_empty)
        norm = jnp.linalg.norm(gradient)
        norm_neu = jnp.linalg.norm(neu_gradient)
        if norm > 0:
            learned_params["syn_weight_mods"] = (learned_params["syn_weight_mods"] + np.array(gradient / norm) * .1 * lr2).clip(.2, 4)
        if norm_neu > 0:
            learned_params["neu_weight_mods"] = (learned_params["neu_weight_mods"] + np.array(neu_gradient / norm_neu) * 1 * lr2).clip(.1, 30)
        """

        keys = cur_spike_count.keys()
        this_loss1 = np.linalg.norm(learned_params["syn_weight_mods"] - np.ones_like(learned_params["syn_weight_mods"]))
        this_loss2 = np.linalg.norm(learned_params["neu_weight_mods"] - np.ones_like(learned_params["neu_weight_mods"]))
        this_loss3 = np.linalg.norm(np.array([cur_spike_count[x] for x in keys]) - np.array([first_spike_count[x] for x in keys]))
        print("syn weight diff", this_loss1)
        print("neu weight diff", this_loss2)
        print("spike diff", this_loss3)
        loss1.append(this_loss1)
        loss2.append(this_loss2)
        loss3.append(this_loss3)
        loss4.append(cur_spike_count[520])

        plt.clf()
        print("losses", loss1, loss2, loss3)
        print("losses", loss1[0], loss2[0], loss3[0])
        print("losses", loss1 / loss1[0], loss2 / loss2[0], loss3 / loss3[0])
        plt.plot(loss1 / loss1[0], color="black")
        plt.plot(loss2 / loss2[0], color="red")
        plt.plot(loss3 / loss3[0], color="blue")
        # plt.plot(loss4 / np.array(25.) + np.array(1), color="green")


