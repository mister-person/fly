from brian2 import Hz, Network, NeuronGroup, SpikeMonitor, Synapses, mV, ms, second
import numpy as np
import matplotlib.pyplot as plt
from pygame_loop import PygameProcess

if __name__ == "__main__":
    v_0 = 0 * mV               # resting potential
    v_rst = 0 * mV
    v_th = 7 * mV               # threshold for spiking
    t_mbr =  20 * ms               # membrane time scale (capacitance * resistance = .002 * uF * 10. * Mohm)
    tau = 5 * ms                 # time constant 

    synapse_weight = .1 * mV
    poisson_rate = 250 * Hz
    model_eqs = ''' 
    v_rst                            : volt
    dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau               : volt (unless refractory) 
    rfc                            : second
    p_weight                       : volt
    input                          : Hz
    '''
    threshold_eq = 'v > v_th or (t - lastspike) * input > 1'
    t_dly = 1.8 * ms
    reset_eq = 'v = v_rst; w = 0; g = 0 * mV'
    refractory_period = 2.2 * ms

    n = 1000
    weight_step = .125

    neu = NeuronGroup( # create neurons
        N = n+1,
        model = model_eqs,
        method = 'linear',
        threshold = threshold_eq,
        reset = reset_eq,
        refractory = 'rfc',
        name = 'default_neurons',
        # namespace = params,
    )
    neu.input[0] = 100 * Hz

    weights = (np.arange(n) + 1 - n//2) * weight_step

    syn = Synapses(neu, neu, 'w : volt', on_pre='g += w', delay=t_dly, name='default_synapses')

    syn.connect(i=np.full(n, 0), j=np.arange(n) + 1)
    # syn.w = weights * mV
    # neu.v = np.concatenate([np.array([0]), weights]) * mV

    vrst_weight = .03
    vrst_weights = np.minimum((np.arange(n) + 1 - n//2) * vrst_weight, 7)
    neu.v_rst = np.concatenate([np.array([0]), vrst_weights]) * mV
    syn_w = 40
    syn.w = syn_w * mV

    spk_mon = SpikeMonitor(neu) 

    runtime = 1 * second

    net = Network(neu, syn, spk_mon)
    net.run(duration=runtime)

    model_synapse_weight = .1 * mV

    print(spk_mon.num_spikes)
    
    print([(s[0] * weight_step, len(s[1])) for s in spk_mon.spike_trains().items()])

    t = spk_mon.t
    j = spk_mon.i
    pygame = PygameProcess(None, {})
    spike_and_times = np.stack((t, j), 1, dtype=object)
    pygame.add_spikes((runtime / second, spike_and_times[:1000000]))
    
    # weights = np.array([(s[0] / second * weight) for s in spk_mon.spike_trains().items()])
    print(weights)
    weights /= 100

    plt.plot(weights, [len(s[1]) / runtime for s in spk_mon.spike_trains().items()][1:], color="blue")
    plt.plot(weights, np.log2((weights - 12.5).clip(min = 1, max=100)), color="red")
    plt.plot(weights, np.maximum(np.log2((weights * 100 - 12.5).clip(min = 0)) * 10, weights).clip(max=100), color="black")

    '''
    plt.plot(vrst_weights, [len(s[1]) / runtime for s in spk_mon.spike_trains().items()][1:], color="blue")
    plt.plot(vrst_weights, np.log2((weights - 12.5).clip(min = 1, max=100)), color="red")
    # plt.plot(vrst_weights, np.maximum(np.log2((weights * 100 - 12.5).clip(min = 1)) * 10, weights).clip(max=100), color="black")
    plt.plot(vrst_weights, np.maximum(np.log2((np.array([syn_w / 100]) * 100 - 12.5).clip(min = 1)) * 10, weights).clip(max=100), color="black")
    # plt.plot(weights, (weights * 10))
    # plt.plot(weights, np.tanh((weights)*10))
    '''
    '''
    plt.plot(weights, [len(s[1]) / runtime for s in spk_mon.spike_trains().items()])
    plt.plot(weights, np.log2((weights - 12.5).clip(min = 1, max=100)) * 15)
    plt.plot(weights, np.maximum(np.log2((weights - 12.5).clip(min = 1)) * 15, weights/10).clip(max=100))
    plt.plot(weights, (weights / 10))
    plt.plot(weights, np.tanh((weights)/10))
    '''
    plt.show()


