import jax
import jax.numpy as jnp
import numpy as np
import openvino

import data
import neuron_groups
from learn import forward
from learn import loss
import time

def jprint(x):
    jax.debug.print(str(x))

df_neu, df_con = data.load("mbanc")
flyid2i = {j: i for i, j in enumerate(df_neu.index)}  # flywire id: brian ID
i2flyid = {j: i for i, j in flyid2i.items()} # brian ID: flywire ID

neurons_to_activate = [10045, 10056] #walk
neurons_to_push = neuron_groups.mbanc_leg_neuron_groups["rf_trochanter_extensor"]

neurons_to_activate = jnp.array([flyid2i[x] for x in neurons_to_activate])
neurons_to_push = jnp.array([flyid2i[x] for x in neurons_to_push])
neurons_to_push_weights = jnp.full_like(neurons_to_push, 1.0)

learned_syn_weights = np.full(len(df_con), 1.0)
learned_neu_weights = np.full(len(df_neu), 1.0)

print("neurons to push", neurons_to_push)

#pre_index, post_index, strength
jnp_con = jnp.array(df_con.to_numpy()[..., [2, 3]])
jnp_strengths = jnp.array(df_con.to_numpy()[..., 6], dtype=jnp.float32)
print(jnp_strengths)

all_neurons = forward(jnp_con, jnp_strengths, learned_syn_weights, learned_neu_weights, neurons_to_activate)

print(all_neurons[neurons_to_push[0].item()])

start1 = time.monotonic()
asdf = jax.jit(jax.grad(loss, argnums=2))(jnp_con, jnp_strengths, learned_syn_weights, learned_neu_weights, neurons_to_activate, neurons_to_push, neurons_to_push_weights)
start2 = time.monotonic()
asdf2 = jax.jit(jax.grad(loss, argnums=2))(jnp_con, jnp_strengths, learned_syn_weights, learned_neu_weights, neurons_to_activate, neurons_to_push, neurons_to_push_weights)
end = time.monotonic()

print("first took", start2-start1, "seconds")
print("second took", end-start2, "seconds")

print("w/o weights", asdf.sort())
print("with weights", asdf2.sort())

asdf3 = jax.make_jaxpr(jax.grad(loss, argnums=2))(jnp_con, jnp_strengths, learned_syn_weights, learned_neu_weights, neurons_to_activate, neurons_to_push, neurons_to_push_weights)
# print(asdf3)
# ov_model = openvino.convert_model(asdf3)

# print(ov_model.parameters)

# print(jax.make_jaxpr(loss)(jnp_con, jnp_strengths, learned_syn_weights, learned_neu_weights, neurons_to_activate, neurons_to_push, neurons_to_push_weights))
