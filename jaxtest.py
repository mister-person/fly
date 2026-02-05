import jax
import jax.numpy as jnp
import numpy as np

import data
import neuron_groups

def jprint(x):
    jax.debug.print(str(x))

df_neu, df_con = data.load("mbanc")
flyid2i = {j: i for i, j in enumerate(df_neu.index)}  # flywire id: brian ID
i2flyid = {j: i for i, j in flyid2i.items()} # brian ID: flywire ID

neurons_to_activate = [10045, 10056] #walk
neurons_to_push = neuron_groups.mbanc_leg_neuron_groups["rf_trochanter_extensor"]

neurons_to_activate = jnp.array([flyid2i[x] for x in neurons_to_activate])
neurons_to_push = jnp.array([flyid2i[x] for x in neurons_to_push])

print("neurons to push", neurons_to_push)

#pre_index, post_index, strength
jnp_con = jnp.array(df_con.to_numpy()[..., [2, 3]])
jnp_strengths = jnp.array(df_con.to_numpy()[..., 6], dtype=jnp.float32)
print(jnp_strengths)

def forward(jnp_con, jnp_strengths, weights, neurons_to_activate):
    connection_strength = .003

    all_neurons = jnp.zeros(211470)
    all_neurons = all_neurons.at[jnp.array(neurons_to_activate)].set(1)
    all_neurons_orig = all_neurons

    jprint(all_neurons[jnp_con[..., 0]].shape)
    jprint(jnp_strengths.shape)

    jprint(jax.numpy.sum(all_neurons_orig))
    for _ in range(5):
        step1 = (all_neurons[jnp_con[..., 0]] * jnp_strengths * weights * connection_strength).clip(max=1)
        all_neurons = all_neurons.at[jnp_con[..., 1]].add(step1)

        jprint(step1.shape)
        jprint(all_neurons.shape)

        jax.debug.print("{}", jax.numpy.sum(all_neurons))

    return all_neurons

@jax.jit
def loss(jnp_con, jnp_strengths, weights, neurons_to_activate, neurons_to_push):
    all_neurons = forward(jnp_con, jnp_strengths, weights, neurons_to_activate)

    return sum(all_neurons[neurons_to_push])

weights = jnp.full_like(jnp_strengths, 1.0)

print(jax.make_jaxpr(forward)(jnp_con, jnp_strengths, weights, neurons_to_activate))

all_neurons = forward(jnp_con, jnp_strengths, weights, neurons_to_activate)

print(all_neurons[neurons_to_push[0].item()])

asdf = jax.grad(loss, argnums=1)(jnp_con, jnp_strengths, weights, neurons_to_activate, neurons_to_push)
asdf2 = jax.grad(loss, argnums=2)(jnp_con, jnp_strengths, weights, neurons_to_activate, neurons_to_push)

print("w/o weights", asdf.sort())
print("with weights", asdf2.sort())
