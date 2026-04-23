from collections import defaultdict
from concurrent.futures.thread import ThreadPoolExecutor
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

def main():
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
            
    partitions_per_second = 500
    first_spikes_t, first_spikes_i = sim.spikes
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
    rand2.random()
    learned_params = {
        "neu_weight_mods": rand2.random(neuron_count) + .3,
        "syn_weight_mods": rand2.random(syn_count) + .3,
    }
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

        voltages = sim.voltages
        print("voltages", voltages.shape)
        print("sum", np.sum(voltages.shape))
        print(voltages)
        print("times", first_partitions.keys())
        this_partitions = partition_spikes(cur_spikes_t, cur_spikes_i, partitions_per_second)

        def thread_func(time):
            current_partition = this_partitions[time]
            all_sids = current_partition.keys() | first_partitions[time].keys()
            diffs = {sid: first_partitions[time][sid] - current_partition[sid] for sid in all_sids}
            weights = learn.weight_by_time_and_voltage(voltages, time)
            gradient, neu_gradient = learn.get_gradient(df_neu, jnp_con, jnp_strengths, learned_params, neurons_to_activate, diffs, weights)
            print("ind. grads", np.sort(gradient), np.sort(neu_gradient))
            return gradient, neu_gradient

        with ThreadPoolExecutor(max_workers=12) as executor:
            times = first_partitions.keys()
            gradients = list(executor.map(thread_func, times))
        # print("grads", gradients)

        gradient = np.stack([x[0] for x in gradients]).sum()
        gradient_neu = np.stack([x[1] for x in gradients]).sum()

        # gradient = (np.array(1) - learned_params["syn_weight_mods"]) * .01
        # gradient_neu = (np.array(1) - learned_params["neu_weight_mods"]) * .01
        # norm = norm_neu = 1

        norm = jnp.linalg.norm(gradient)
        norm_neu = jnp.linalg.norm(gradient_neu)
        print("norms", norm, norm_neu)
        lrs = .005
        lrn = .001
        if norm > 0:
            learned_params["syn_weight_mods"] = (learned_params["syn_weight_mods"] + np.array(gradient / norm) * lrs)#.clip(.2, 4)
        if norm_neu > 0:
            learned_params["neu_weight_mods"] = (learned_params["neu_weight_mods"] + np.array(gradient_neu / norm_neu) * lrn)#.clip(.1, 30)
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
        this_loss2 = np.linalg.norm(learned_params["neu_weight_mods"] - np.ones_like(learned_params["neu_weight_mods"]))
        this_loss3 = np.linalg.norm(np.array([cur_spike_count[x] for x in keys]) - np.array([first_spike_count[x] for x in keys]))
        print("syn weight diff", this_loss1)
        print("neu weight diff", this_loss2)
        print("spike diff", this_loss3)
        print("advanced spike diff", this_loss4)
        loss1.append(this_loss1)
        loss2.append(this_loss2)
        loss3.append(this_loss3)
        loss4.append(this_loss4)

        plt.clf()
        print("losses", loss1, loss2, loss3)
        print("losses", loss1[0], loss2[0], loss3[0])
        print("losses", loss1 / loss1[0], loss2 / loss2[0], loss3 / loss3[0])
        plt.plot(loss1 / loss1[0], color="black")
        plt.plot(loss2 / loss2[0], color="red")
        plt.plot(loss3 / loss3[0], color="blue")
        plt.plot(np.array(loss4) / loss4[0], color="green")

if __name__ == "__main__":
    main()
