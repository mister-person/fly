import openvino
import torch
import torch.nn as nn
import time
import jax
import jax.numpy as jnp

import data
import neuron_groups
import learn

def forward(con, start_synapse_weights, learned_syn_weights, learned_neu_weights, neurons_to_activate):
    device = start_synapse_weights.device
    print(device)

    all_neurons = torch.zeros(211743, device=device)
    all_neurons[torch.tensor(neurons_to_activate, device=device)] = 1.0

    print(all_neurons[con[..., 0]].shape)
    print(start_synapse_weights.shape)

    synapse_weights = torch.tanh(learned_syn_weights * start_synapse_weights / 1000)

    # assert synapse_weights.requires_grad, "synapse_weights lost grad connection"

    for _ in range(50):
        pre_synapse_strengths = all_neurons[con[..., 0]].clamp(min=0, max=1)
        synapse_strengths = (pre_synapse_strengths * synapse_weights).clamp(min=0, max=1)

        # assert synapse_strengths.requires_grad, "synapse_strengths lost grad connection"

        neuron_updates = torch.zeros_like(all_neurons)
        neuron_updates = neuron_updates.scatter_add(0, con[..., 1], synapse_strengths)
        neuron_updates = neuron_updates * learned_neu_weights

        # assert neuron_updates.requires_grad, "neuron_updates lost grad connection"

        all_neurons = all_neurons + neuron_updates

    return all_neurons

# @torch.compile(backend="openvino", options = {"device": "GPU"})
@torch.compile()
def loss(con, start_synapse_weights, learned_syn_weights, learned_neu_weights, neurons_to_activate, neurons_to_push, neurons_to_push_weights):
    new_neuron_values = forward(con, start_synapse_weights, learned_syn_weights, learned_neu_weights, neurons_to_activate)
    return torch.sum(neurons_to_push_weights * new_neuron_values[neurons_to_push])

@torch.compile()
# @torch.compile(backend="openvino", options = {"device": "GPU"})
def get_gradients(con, start_synapse_weights, learned_syn_weights, learned_neu_weights, 
                  neurons_to_activate, neurons_to_push, neurons_to_push_weights):
    
    # Ensure leaf tensors have gradients enabled
    # con = con.detach().requires_grad_(True)
    # start_synapse_weights = start_synapse_weights.detach().requires_grad_(True)
    learned_syn_weights = learned_syn_weights.detach().requires_grad_(True)
    learned_neu_weights = learned_neu_weights.detach().requires_grad_(True)
    # neurons_to_activate = neurons_to_activate.detach().requires_grad_(True)
    # neurons_to_push = neurons_to_push.detach().requires_grad_(True)
    # neurons_to_push_weights = neurons_to_push_weights.detach().requires_grad_(True)

    synapse_weights = torch.tanh(learned_syn_weights * start_synapse_weights / 1000)

    assert synapse_weights.requires_grad, "synapse_weights lost grad connection"
    
    loss_val = loss(con, start_synapse_weights, learned_syn_weights, learned_neu_weights,
                    neurons_to_activate, neurons_to_push, neurons_to_push_weights)
    
    loss_val.backward()
    
    return learned_syn_weights.grad, learned_neu_weights.grad

core = openvino.Core()
print(core.get_available_devices())

df_neu, df_con = data.load("mbanc")
flyid2i = {j: i for i, j in enumerate(df_neu.index)}  # flywire id: brian ID
i2flyid = {j: i for i, j in flyid2i.items()} # brian ID: flywire ID

neurons_to_activate = [10045, 10056] #walk
neurons_to_push = neuron_groups.mbanc_leg_neuron_groups["rf_trochanter_extensor"]

neurons_to_activate = torch.tensor([flyid2i[x] for x in neurons_to_activate])
neurons_to_push = torch.tensor([flyid2i[x] for x in neurons_to_push])
neurons_to_push_weights = torch.full_like(neurons_to_push, 1.0, dtype=torch.float32)

learned_syn_weights = torch.full((len(df_con),), 1.0, dtype=torch.float32)
learned_neu_weights = torch.full((len(df_neu),), 1.0, dtype=torch.float32)

con = torch.tensor(df_con.to_numpy()[..., [2, 3]])
start_synapse_weights = torch.tensor(df_con.to_numpy()[..., 6], dtype=torch.float32)

start1 = time.monotonic()
a = get_gradients(con, start_synapse_weights, learned_syn_weights, learned_neu_weights, neurons_to_activate, neurons_to_push, neurons_to_push_weights)
start2 = time.monotonic()
b = get_gradients(con, start_synapse_weights, learned_syn_weights, learned_neu_weights, neurons_to_activate, neurons_to_push, neurons_to_push_weights)
start3 = time.monotonic()
c = get_gradients(con, start_synapse_weights, learned_syn_weights, learned_neu_weights, neurons_to_activate, neurons_to_push, neurons_to_push_weights)
end = time.monotonic()

print(a)
print(b)

print("a took", start2 - start1, "seconds")
print("b took", start3 - start2, "seconds")
print("c took", end - start3, "seconds")

con = jnp.array(con)
start_synapse_weights = jnp.array(start_synapse_weights, dtype=jnp.float32)
learned_syn_weights = jnp.array(learned_syn_weights, dtype=jnp.float32)
learned_neu_weights = jnp.array(learned_neu_weights, dtype=jnp.float32)
neurons_to_activate = jnp.array(neurons_to_activate)
neurons_to_push = jnp.array(neurons_to_push)
neurons_to_push_weights = jnp.array(neurons_to_push_weights, dtype=jnp.float32)

# ov_model = openvino.convert_model(loss)
jloss = jax.jit(jax.grad(learn.loss, (2, 3)))
start1 = time.monotonic()
a = jloss(con, start_synapse_weights, learned_syn_weights, learned_neu_weights, neurons_to_activate, neurons_to_push, neurons_to_push_weights)
start2 = time.monotonic()
b = jloss(con, start_synapse_weights, learned_syn_weights, learned_neu_weights, neurons_to_activate, neurons_to_push, neurons_to_push_weights)
start3 = time.monotonic()
c = jloss(con, start_synapse_weights, learned_syn_weights, learned_neu_weights, neurons_to_activate, neurons_to_push, neurons_to_push_weights)
end = time.monotonic()

print(a)
print(b)

print("jax a took", start2 - start1, "seconds")
print("jax b took", start3 - start2, "seconds")
print("jax c took", end - start3, "seconds")

# loss_jaxpr = jax.make_jaxpr(jax.grad(learn.loss, (2, 3)))(con, start_synapse_weights, learned_syn_weights, learned_neu_weights, neurons_to_activate, neurons_to_push, neurons_to_push_weights)

# ov_model = openvino.convert_model(loss_jaxpr)
