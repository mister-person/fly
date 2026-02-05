from collections import defaultdict
from dataclasses import dataclass
from queue import Empty
import queue
from threading import Condition
from time import sleep
from brian2 import Hz, Network, NeuronGroup, PoissonInput, SpikeMonitor, Synapses, mV, ms, network_operation
import brian2
import numpy as np
import pandas as pd
import Drosophila_brain_model.model as model
import data
from profile_dec import profile

def start_neuron_sim(df_comp, df_con, dataset, neurons_to_activate, control_queue, spike_queue, input_queue: queue.Queue, runtime=100_000 * ms):
    start_neuron_sim_do(df_comp, df_con, dataset, neurons_to_activate, control_queue, spike_queue, input_queue, runtime)

@profile
def start_neuron_sim_do(df_comp, df_con, dataset, neurons_to_activate, control_queue, spike_queue, input_queue: queue.Queue, runtime=100_000 * ms):
    print("started neuron sim")
    # load name/id mappings
    flyid2i = {j: i for i, j in enumerate(df_comp.index)}  # flywire id: brian ID
    i2flyid = {j: i for i, j in flyid2i.items()} # brian ID: flywire ID

    i2flyid_np = np.vectorize(i2flyid.get, otypes=[np.integer])

    v_0 = -52 * mV               # resting potential
    v_rst = -52 * mV               # reset potential after spike
    v_th = -45 * mV               # threshold for spiking
    t_mbr =  20 * ms               # membrane time scale (capacitance * resistance = .002 * uF * 10. * Mohm)
    tau = 5 * ms                 # time constant 

    # params = model.default_params
    synapse_weight = .1 * mV
    poisson_rate = 250 * Hz
    model_eqs = ''' 
    dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau               : volt (unless refractory) 
    rfc                            : second
    p_weight                       : volt
    input                          : Hz
    '''
    threshold_eq = 'v > v_th or (t - lastspike) * input > 1'
    t_dly = 3.6 * ms
    reset_eq = 'v = v_rst; w = 0; g = 0 * mV'
    refractory_period = 2.2 * ms

    # model.run_trial([flyid2i[flyid] for flyid in test.neu_sugar], [], [], path_comp, path_con, params)

    # neu, syn, spk_mon = model.create_model(df_comp, df_con, params)

    neu = NeuronGroup( # create neurons
        N = len(df_comp),
        model = model_eqs,
        method = 'linear',
        threshold = threshold_eq,
        reset = reset_eq,
        refractory = 'rfc',
        name = 'default_neurons',
        # namespace = params,
    )

    syn = Synapses(neu, neu, 'w : volt', on_pre='g += w', delay=t_dly, name='default_synapses')
    i_pre = df_con.loc[:, 'Presynaptic_Index'].values
    i_post = df_con.loc[:, 'Postsynaptic_Index'].values
    syn.connect(i=i_pre, j=i_post)

    spk_mon = SpikeMonitor(neu) 

    _, synapse_map, reverse_synapse_map = data.get_synapse_map(dataset)

    neurons_indexes = [flyid2i[flyid] for flyid in neurons_to_activate]

    # poi_inp, neu = model.poi(neu, neurons_i, [], params)
    pois = []
    for i in neurons_indexes:
        print("creating poi for", i2flyid[i])
        p = PoissonInput(
            target=neu[i], 
            target_var='v', 
            N=1, 
            rate=poisson_rate ,
            weight="p_weight"
            # weight=params['w_syn']*params['f_poi'],
            # weight = 68.75 * mV
            )
        neu[i].rfc = 0 * ms # no refractory period for Poisson targets
        pois.append(p)

    poi_inp = pois

    STATE_POI = 1
    STATE_RATE = 2

    @dataclass
    class State:
        state: int
        _rate: int
        @property
        def rate(self):
            return self._rate * Hz

    control_states: list[State] = [State(STATE_POI, 250)]
    neu.input[neurons_indexes] = 100 * Hz

    # spikes_processed = 0
    last_spike_index = 0

    @network_operation() # type: ignore
    def fast_update(time):
        nonlocal last_spike_index
        # nonlocal spikes_processed
        spikes = spk_mon.i[last_spike_index:]

        spike_queue.put((time[:] / brian2.second, i2flyid_np(spikes)))
        last_spike_index = len(spk_mon.i)
        input_queue.get()

    last_time = 0

    started = False

    @network_operation(dt=16.6666*ms) # type: ignore 
    # @network_operation(dt=1*ms) # type: ignore 
    def update(time):
        nonlocal last_time
        nonlocal started
        # print(len(spk_mon.i), spikes_processed)

        if not started:
            print("started!")
            started = True
        # volt_array = neu.v
        # dif = np.where(volt_array != -52 * mV)
        # if len(dif) > 0:
            # print(dif)
        # print(np.sum(spk_mon.count)) # type: ignore
        # print(spk_mon.i[-10:])
        # print(spk_mon.t[-10:])
        # index = np.searchsorted(spk_mon.t, last_time)
        # spike_indices = spk_mon.i[index:]
        # spike_times = spk_mon.t[index:]
        # if len(spike_indices) > 0:
            # points = np.stack((spike_times, i2flyid_np(spike_indices)), 1, dtype=object)
            # spike_queue.put((time[:] / brian2.second, points))
        # else:
            # spike_queue.put((time[:] / brian2.second, []))

        paused = False
        while True:
            try:
                while event := control_queue.get_nowait():
                    if event[0] == "start":
                        print("how the fuck")
                    if event[0] == "group1":
                        if event[1]:
                            if control_states[0].state == STATE_POI:
                                neu.p_weight[neurons_indexes] = 65 * mV
                                neu.input[neurons_indexes] = 0 * Hz
                            elif control_states[0].state == STATE_RATE:
                                neu.p_weight[neurons_indexes] = 0 * mV
                                neu.input[neurons_indexes] = control_states[0].rate
                            print("input on")
                        else:
                            neu.p_weight[neurons_indexes] = 0 * mV
                            neu.input[neurons_indexes] = 0 * Hz
                            print("input off")
                    elif event[0] == "pause":
                        paused = event[1]
                    elif event[0] == "rate1":
                        # neu.input[neurons_indexes] = event[2]
                        control_states[0]._rate = event[1]
                        control_states[0].state = STATE_RATE

                        neu.p_weight[neurons_indexes] = 0 * mV
                        neu.input[neurons_indexes] = control_states[0].rate
                    else:
                        print(f"received event {event[0]} with value {event[1]}")
            except Empty:
                pass

            if not paused:
                break
            else:
                sleep(.05)

        last_time = time[:]

    net = Network(neu, syn, spk_mon, *poi_inp, update, fast_update)

    while True:
        while True:
            event = control_queue.get()
            if event[0] == "start":
                learned_params = event[1]
                syn_weight_mods = 1 if "syn_weight_mods" not in learned_params else learned_params["syn_weight_mods"]

                if "neu_weight_mods" in learned_params:
                    neu_weight_mods = learned_params["neu_weight_mods"]
                    neu_weights_by_syn = neu_weight_mods[df_con.loc[:,'Presynaptic_Index']]
                else:
                    neu_weights_by_syn = 1

                syn.w = df_con.loc[:,'Excitatory x Connectivity'].values * syn_weight_mods * synapse_weight * neu_weights_by_syn
                print(syn.w)
                neu.v = v_0
                neu.g = 0
                neu.rfc = refractory_period
                break

        # run simulation
        net.run(duration=runtime)

        spike_queue.put(None)
