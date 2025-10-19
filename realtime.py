from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import multiprocessing
from queue import Empty, Queue
import threading
from time import sleep
from brian2 import Hz, Mvolt, Network, NeuronGroup, PoissonInput, Quantity, SpikeGeneratorGroup, SpikeMonitor, Synapses, mV, ms, network_operation, second, volt
import brian2
from brian2.core.variables import VariableView
from drawutils import SpikeDrawer
import sys
import os
import Drosophila_brain_model.model as model
from gymtest import setup_fly
import neuron_groups
import pygame_loop
import test
import pandas as pd
import numpy as np
import flygym
import flygym.state as flygym_state
import flygym.preprogrammed as flygym_preprogrammed
from flygym.preprogrammed import all_leg_dofs

import testbanc

def get_synapse_map(dataset):
    if dataset == "banc":
        SYNAPSES_FILENAME = "./data/banc_connectivity.parquet"
    else:
        SYNAPSES_FILENAME = "./Drosophila_brain_model/Connectivity_783.parquet"

    synapses_df = pd.read_parquet(SYNAPSES_FILENAME)
    synapses = synapses_df.to_numpy()
    synapse_map = defaultdict(list)
    reverse_synapse_map = defaultdict(list)
    for synapse in synapses[1:]:
        synapse_map[synapse[0]].append((synapse[1], synapse[6]))
        reverse_synapse_map[synapse[1]].append((synapse[0], synapse[6]))

    return synapses, synapse_map, reverse_synapse_map

def start_sim(df_comp, df_con, neurons_to_activate):
    # load name/id mappings
    df_comp = pd.read_csv(path_comp, index_col=0) # load completeness dataframe
    flyid2i = {j: i for i, j in enumerate(df_comp.index)}  # flywire id: brian ID
    i2flyid = {j: i for i, j in flyid2i.items()} # brian ID: flywire ID

    i2flyid_np = np.vectorize(i2flyid.get, otypes=[np.integer])

    params = model.default_params
    params['w_syn'] = .175 * mV
    params['r_poi'] = 250 * Hz
    params['t_run'] = 100_000 * ms
    params['n_run'] = 1
    params['eqs'] = ''' 
    dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau               : volt (unless refractory) 
    rfc                            : second
    p_weight                       : volt
    input                          : Hz
    '''
    params['eq_th'] = 'v > v_th or (t - lastspike) * input > 1'
    # params['t_dly'] = 3.6 * ms
    params['t_rfc'] = 2.2 * ms

    # model.run_trial([flyid2i[flyid] for flyid in test.neu_sugar], [], [], path_comp, path_con, params)

    neu, syn, spk_mon = model.create_model(df_comp, df_con, params)

    _, synapse_map, reverse_synapse_map = get_synapse_map("banc" if banc else "fafb")

    neurons_indexes = [flyid2i[flyid] for flyid in neurons_to_activate]

    # poi_inp, neu = model.poi(neu, neurons_i, [], params)
    pois = []
    for i in neurons_indexes:
        print("creating poi for", i2flyid[i])
        p = PoissonInput(
            target=neu[i], 
            target_var='v', 
            N=1, 
            rate=params['r_poi'], 
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
    # neu.p_weight = 65 * mV

    # poi_inp, neu = model.poi(neu, [], [flyid2i[flyid] for flyid in test.neu_sugar], params)
    # silence neurons
    # syn = model.silence([], syn)
    # collect in Network object

    last_time = 0

    started = False

    multiprocessing.freeze_support()
    mp_context = multiprocessing.get_context("spawn")
    spike_queue = mp_context.Queue()
    input_queue = mp_context.Queue()
    frame_queue = mp_context.Queue()
    render_thread = mp_context.Process(target=pygame_loop.start_pygame, args=[spike_queue, input_queue, frame_queue, banc, neurons_to_activate])
    render_thread.start()
    
    sim, obs, info = setup_fly()
    actuated_joints = flygym_preprogrammed.all_leg_dofs
    starting_pose: flygym_state.kinematic_pose.KinematicPose = flygym_preprogrammed.get_preprogrammed_pose("tripod")
    print(starting_pose.joint_pos)
    starting_positions = np.zeros(len(actuated_joints))
    joints = {}
    for i, joint in enumerate(actuated_joints):
        starting_positions[i] = starting_pose[joint]
        joints[joint] = i

    action_size = len(flygym.preprogrammed.all_leg_dofs) # type: ignore

    spikes_processed = 0
    last_spike_index = 0
    neurons = set(neuron_groups.rf_leg_motor_neurons)

    joint_state = starting_positions.copy()

    leg_neuron_groups = neuron_groups.leg_neuron_groups

    neuron_to_muscle = {}

    muscle_to_gym_name = {
        "trochanter_flexor": ("Femur", 1),
        "trochanter_extensor": ("Femur", -1),
        "tibia_flexor": ("Tibia", 1),
        "tibia_extensor": ("Tibia", -1),
        "tarsus_depressor": ("Tarsus1", 1),
        "tarsus_levetator": ("Tarsus1", -1),
        # "long_tendon": ("": 1),
    }

    for neuron in leg_neuron_groups.keys():
        side = leg = None
        if neuron[0] == 'r':
            side = "R"
        if neuron[0] == 'l':
            side = "L"
        if neuron[1] == 'f':
            leg = 'F'
        if neuron[1] == 'm':
            leg = 'M'
        if neuron[1] == 'h':
            leg = 'H'

        if side is None or leg is None:
            print("error building map thing")
            print(neuron, side, leg)
            exit()

        for muscle in muscle_to_gym_name.keys():
            if muscle in neuron: # example: joint_LHFemu
                gym_name, sign = muscle_to_gym_name[muscle]
                neuron_to_muscle[neuron] = ("joint_" + side + leg + gym_name, sign)

    @network_operation() # type: ignore 
    def fast_update(time):
        nonlocal last_spike_index
        nonlocal spikes_processed
        spikes = spk_mon.i[last_spike_index:]
        last_spike_index = len(spk_mon.i)
        for spike_index in spikes:
            spikes_processed += 1
            spike = i2flyid[spike_index]
            if spike in neurons:
                if spike in neuron_groups.rf_trochanter_flexor:
                    joint_state[joints["joint_RFFemur"]] += 1
                    print("rf femur")
                elif spike in neuron_groups.rf_trochanter_extensor:
                    joint_state[joints["joint_RFFemur"]] -= 1
                    print("rf femur -")
                elif spike in neuron_groups.rf_tibia_extensor:
                    joint_state[joints["joint_RFTibia"]] += 1
                    print("rf tibia")
                elif spike in neuron_groups.rf_tibia_flexor:
                    joint_state[joints["joint_RFTibia"]] -= 1
                    print("rf tibia -")
                elif spike in neuron_groups.rf_tarsus_depressor:
                    joint_state[joints["joint_RFTarsus1"]] += 1
                    print("rf tarsus")
                elif spike in neuron_groups.rf_tarsus_levetator:
                    joint_state[joints["joint_RFTarsus1"]] -= 1
                    print("rf tarsus -")
                print(joint_state[joints["joint_RFFemur"]])

        action = {"joints": joint_state}
        joint_state[:] = (starting_positions * .05 + joint_state * .95)
        obs, reward, terminated, truncated, info = sim.step(action)

        frame = sim.render()[0]
        if frame is not None:
            frame_queue.put((frame.shape[1], frame.shape[0], frame.tobytes()))
                

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
        index = np.searchsorted(spk_mon.t, last_time)
        spike_indices = spk_mon.i[index:]
        spike_times = spk_mon.t[index:]
        if len(spike_indices) > 0:
            points = np.stack((spike_times, i2flyid_np(spike_indices)), 1, dtype=object)
            spike_queue.put((time[:] / brian2.second, points))
        else:
            spike_queue.put((time[:] / brian2.second, []))


        '''
        # test_neurons = [720575941496703104, 720575941625121674]
        test_neurons = [720575941669833905,720575941560438643,720575941504247575,720575941527002649,720575941554038555,720575941552245822]
        # test_neurons = [720575941469024064,720575941487873488,720575941545638505,720575941625121674,720575941437338412,720575941519468654]
        for i, spike in enumerate(spike_indices):
            n_id = i2flyid[spike]
            inp = next((x for x in synapse_map[n_id] if x[0] in test_neurons), None)
            # out = next((x for x in reverse_synapse_map[n_id] if x[0] in test_neurons), None)
            if n_id in test_neurons:
                print("neuron spiked! time", spike_times[i], "id", test_neurons.index(n_id), inp)
            # out = next((x for x in lst if ...), None)
            if inp != None:
                if n_id in test_neurons:
                    print("in from", test_neurons.index(n_id), "to", inp)
                else:
                    print("in from", n_id, "to", inp)
            # if out != None:
                # print("out", n_id)

            # if test_neuron in synapse_map[n_id]:
                # print(synapse_map[n_id]
            # if test_neuron in reverse_synapse_map:

        '''
                
        # print("network ooooOOOooo", time, neu.indices[0], neu.v)

        paused = False
        while True:
            try:
                while event := input_queue.get_nowait():
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

    print('syn', syn)
    net = Network(neu, syn, spk_mon, *poi_inp, update, fast_update)

    # run simulation
    net.run(duration=params['t_run'])

    # spike times 
    # spk_trn: SpikeMonitor = model.get_spk_trn(spk_mon)# type: ignore
    assert(type(spk_mon.count) == VariableView)
    print(np.sum(spk_mon.count))

if __name__ == "__main__":
    banc = True

    if not banc:
        config = {
             'path_res'  : './results',                              # directory to store results,
             'path_comp' : './Drosophila_brain_model/Completeness_783.csv',        # csv of the complete list of Flywire neurons,
             'path_con'  : './Drosophila_brain_model/Connectivity_783.parquet',    # connectivity data,
             'n_proc'    : -1,                                               # number of CPU cores (-1: use all),
        }
        neurons_to_activate = test.neu_sugar
    else:
        config = {
             'path_res'  : './results',                              # directory to store results,
             'path_comp' : './data/banc_completeness.csv',        # csv of the complete list of Flywire neurons,
             'path_con'  : './data/banc_connectivity.parquet',    # connectivity data,
             'n_proc'    : -1,                                               # number of CPU cores (-1: use all),
        }
        neurons_to_activate = [720575941626500746] #walk
        # neurons_to_activate = [720575941540215501, 720575941459097503 ] #giant fiber??

    path_comp = config["path_comp"]
    path_con = config["path_con"]

    df_comp = pd.read_csv(path_comp, index_col=0)
    df_con = pd.read_parquet(path_con)
    
    df_con.add

    filter_neurons = []
    filter_neurons += [720575941669833905,720575941560438643,720575941504247575,720575941527002649,720575941554038555,720575941552245822] #e1
    filter_neurons += [720575941469024064,720575941487873488,720575941545638505,720575941625121674,720575941437338412,720575941519468654] #e2
    filter_neurons += [720575941496703104,720575941503572133,720575941589995319,720575941441065919,720575941544954556,720575941687554799] #i1
    filter_neurons += [720575941569601650,720575941515421443,720575941555553363,720575941414465684,720575941461249747,720575941649246741] #i2
    filter_neurons += [720575941626500746]
    filtered_con = df_con.iloc[:0,:].copy()
    for i, a in enumerate(df_con.to_numpy()):
        if a[0] in filter_neurons and a[1] in filter_neurons:
            filtered_con.loc[i] = a

    print("filtered_connections", filtered_con)

    start_sim(df_comp, df_con, neurons_to_activate)
    
