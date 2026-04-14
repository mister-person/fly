from dataclasses import dataclass
import dataclasses
import time
from brian2 import ms
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import pandas as pd

import neuron_model

#100us
#1.8ms

# todo shape thing

@dataclass(eq=True, frozen=True)
class Params:
    neuron_decay: float
    rise_decay: float
    delay_iters: int
    threshold: float
    refractory_iters: int
    steps: int
    global_synapse_weight: float

def synapse_activation_gradient_fn(start_voltage, threshold):
    return jnp.tanh((start_voltage - threshold)/3)

@jax.custom_gradient
def synapse_activation(start_voltage, threshold):
    return start_voltage >= threshold, lambda g: jax.vmap(jax.grad(synapse_activation_gradient_fn))(g) #type: ignore

def timestep(params, connections, synapse_weights, voltages, refractory_timers, rise_values, iteration):
    neurons_current = voltages[iteration]

    neurons_pre_synapse = jnp.where(iteration - params.delay_iters >= 0, voltages[iteration - params.delay_iters], jnp.zeros_like(neurons_current))
    pre_synapse_strengths = synapse_activation(neurons_pre_synapse[connections[..., 0]], params.threshold)
    synapse_strenghts: jnp.ndarray = (pre_synapse_strengths * synapse_weights)
    neuron_updates = jnp.zeros_like(neurons_current).at[connections[..., 1]].add(synapse_strenghts)

    rise_values += neuron_updates
    rise_values = rise_values * params.rise_decay * (refractory_timers == 0)

    out = (neurons_current - rise_values) * params.neuron_decay + rise_values
    out = out * (refractory_timers == 0)

    # new_refractory_timers = (refractory_timers.at[out >= 1].set(params.refractory_iters) - 1).clip(min=0)
    new_refractory_timers = (jnp.where(out >= params.threshold, params.refractory_iters + 1, refractory_timers) - 1).clip(min=0)

    return out, new_refractory_timers, rise_values

# @jax.jit(static_argnames=["params", "num_neurons"])
def run_sim(params, connections, num_neurons, synapse_weights, neurons_to_activate):
    synapse_weights *= params.global_synapse_weight
    all_voltages = jnp.zeros((params.steps, num_neurons))
    refractory_timers = jnp.zeros(num_neurons)
    rise_values = jnp.zeros(num_neurons)
    def loop(i, x):
        (all_voltages, refractory_timers, rise_values) = x
        with_input = all_voltages.at[i, neurons_to_activate].set(((i%100)==0) * params.threshold / params.neuron_decay)
        neurons, refractory_timers, rise_values = timestep(params, connections, synapse_weights, with_input, refractory_timers, rise_values, i)
        all_voltages = all_voltages.at[i + 1].set(neurons)
        return (all_voltages, refractory_timers, rise_values)
    (all_voltages, refractory_timers, rise_values) = jax.lax.fori_loop(0, params.steps, loop, (all_voltages, refractory_timers, rise_values))

    return all_voltages

def run_full_model():
    import data
    import matplotlib.pyplot as plt
    jax.config.update("jax_compilation_cache_dir", "./tmp/jax_cache")
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

    neu, con = data.load("mbanc")
    flyid2i = {j: i for i, j in enumerate(neu.index)}  # flywire id: brian ID
    neurons_to_activate_ids = [10045, 10056] #walk
    neurons_to_activate = jnp.array([flyid2i[10045], flyid2i[10056]])

    params = dataclasses.replace(default_params, steps=300)

    num_neurons = len(neu)
    connections = jnp.array(con.to_numpy())[:, 2:4]
    synapse_weights = jnp.array(con.to_numpy())[:, 6]

    # connections = connections.at[5].set([10045, 10099])
    # synapse_weights = synapse_weights.at[5].set(500)
    # con.iloc[:, 2] = 10045
    # con.iloc[:, 3] = 10099
    # con.iloc[:, 6] = 10000

    """
    b2sim = neuron_model.NeuronSim(neu, con, "", neurons_to_activate_ids, (params.steps / 10)*ms)
    sim_start = time.monotonic()
    b2sim.start({})
    while(True):
        spikes = b2sim.spike_queue.get()
        if spikes == None:
            break
        update_time, spikes = spikes
        b2sim.input_queue.put(())

    sim_end = time.monotonic()
    print("b2 sim took", sim_end - sim_start, "seconds")

    b2v = b2sim.voltages
    """

    sim_start = time.monotonic()
    # v = compiled(params, connections, num_neurons, synapse_weights, neurons_to_activate)
    print(synapse_weights)
    v = run_sim(params, connections, num_neurons, synapse_weights, neurons_to_activate)
    print(v[-1][-1])
    sim_end = time.monotonic()
    print("jax sim took", sim_end - sim_start, "seconds")

    # pygame_process = pygame_loop.PygameProcess()
    # all_spikes = jax.jit(lambda a: jnp.where(a >= params.threshold, size=10000))(v)
    # spike_times = all_spikes[0] / 10000
    # spike_ixs = all_spikes[1]
    # all_spikes = jnp.stack((spike_times, spike_ixs), axis=1)
    # print("all spikes", all_spikes)
    # print(all_spikes.shape)
    # pygame_process.add_spikes((params.steps / 10000, all_spikes.tolist()))

    print(v.shape)
    print(v.nbytes)

    npv = np.asarray(v)
    # active_ns = jax.jit(lambda vv: jnp.where(jnp.sum(vv, axis=0) > .01, size=1000)[0])(v)
    active_ns = neurons_to_activate
    print("done getting jax v")

    # print("active neurons", [x for x in active_ns if x != 0])

    print(active_ns[:10])
    for n in active_ns[:10]:
        print(n)
        # if n in neurons_to_activate:
            # continue
        plt.plot(npv[:, n])
        print("after plot", n)

    plt.ion()
    plt.show()
    print("shown")
    plt.figure(2)

    """
    active_ns = jax.jit(lambda vv: jnp.where(jnp.sum(vv, axis=0) > .01, size=1000)[0])(b2v.T)
    print("done getting b2 v")

    # print("active neurons", [x for x in active_ns if x != 0])

    for n in active_ns[:10]:
        if n in neurons_to_activate:
            continue
        plt.plot(b2v.T[:, n])
    """

    plt.ion()
    plt.show()
    while True:
        plt.pause(1)

# dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
# t_mbr = 20 ms = .02 s
# v = 1 -> dv/dt = -v/.02 = -50
# dv = .0001 -> .005

# tau = .005 seconds = 5 ms = 50 dts
# dg/dt = -g / tau
# tau=1 -> dg/dt = -200
# dt = .0001
# dg = -.02

# synapse_weight = .1 * mV = .0001 * v

# v_th = 7 * mV,
default_params = Params(
    neuron_decay = .99497,
    rise_decay = .9803,
    threshold = .007,
    delay_iters = 18,
    refractory_iters = 22,
    steps = 500,
    global_synapse_weight=.0001,
)

def test_against_b2():
    params = dataclasses.replace(default_params, steps=1000)
    num_neurons = 100
    neurons_to_activate = jnp.array([1, 20, 21])
    # connections = jnp.array([[1, 2, 3, 4], [2, 3, 5, 9]])
    # connections = jnp.array([[1, 2], [2, 3], [2, 4], [3, 4]])
    # synapse_weights = jnp.array([300, 400, 200, 50])
    connections = jax.random.randint(jax.random.key(11), (50, 2), 0, 10)
    synapse_weights = jax.random.uniform(jax.random.key(12), (50,)) * 500
    print("b4 filter", connections)
    print("in", jnp.where(jnp.isin(connections[:, 1], neurons_to_activate)))
    connections = connections.at[jnp.where(jnp.isin(connections[:, 1], neurons_to_activate))].set([6, 7])
    print("a4ter filter", connections)
    sim_start = time.monotonic()
    v = run_sim(params, connections, num_neurons, synapse_weights, neurons_to_activate)
    sim_end = time.monotonic()
    print("jax sim took", sim_end - sim_start, "seconds")

    print(v[:, 1:3])

    df_con = pd.DataFrame(range(len(connections)))
    df_con.insert(0, "Presynaptic_Index", connections[:, 0])
    df_con.insert(1, "Postsynaptic_Index", connections[:, 1])
    df_con.insert(2, "Excitatory x Connectivity", synapse_weights)
    b2sim = neuron_model.NeuronSim(pd.DataFrame(range(num_neurons)), df_con, "", neurons_to_activate.tolist(), (params.steps / 10)*ms)
    b2sim.voltage_resolution = .1 * ms
    sim_start = time.monotonic()
    b2sim.start({})
    while(True):
        spikes = b2sim.spike_queue.get()
        if spikes == None:
            break
        update_time, spikes = spikes
        b2sim.input_queue.put(())

    sim_end = time.monotonic()
    print("b2 sim took", sim_end - sim_start, "seconds")

    b2v = b2sim.voltages

    import matplotlib.pyplot as plt

    for x in [1, 2, 3, 4, 5]:
        plt.plot(v[:, x])

    plt.ylim(top=.0072)

    plt.figure(2)
    for x in [1, 2, 3, 4, 5]:
        plt.plot(b2v.T[:, x])

    plt.ylim(top=.0072)

    print(b2v)

    plt.ion()
    plt.show()
    while True:
        plt.pause(1)

if __name__ == "__main__":
    test_against_b2()

    # run_full_model()
