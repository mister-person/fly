import matplotlib.pyplot as plt
from jax import config

import jaxley as jx
from jaxley.channels import HH
import jaxley.synapses

config.update("jax_platform_name", "cpu")  # Or "gpu" / "tpu".

cell = jx.Cell()  # Define cell.
cell.insert(HH())  # Insert channels.

current = jx.step_current(i_delay=1.0, i_dur=2000.0, i_amp=0.005, delta_t=0.025, t_max=160.0)
# cell.stimulate(current)  # Stimulate with step current.

cell2 = jx.Cell() 
cell2.insert(HH())

# v = jx.integrate(cell)  # Run simulation.

net = jx.Network([cell, cell2])
# jx.fully_connect(net.cell(range(2)), net.cell(range(2)), jaxley.synapses.IonotropicSynapse())
jx.connect(net.cell(0), net.cell(1), jaxley.synapses.IonotropicSynapse())
net.set("IonotropicSynapse_gS", 0.00004)
net.set("IonotropicSynapse_e_syn", 0)
net.set("IonotropicSynapse_k_minus", .25)

net.cell(1).set("v", -65)
# set m, n, h to resting values
net.cell(1).set("HH_m", .053)
net.cell(1).set("HH_n", .318)
net.cell(1).set("HH_h", .595)

net.record("v")
net.record("HH_m")
net.record("HH_n")
net.record("HH_h")

print("x", net.nodes.to_string()) #type: ignore
print("s", net.shape)
print("e", net.edges.to_string())
print(f"cell 0\n{net.cell(0).nodes}")
print(f"cell 1\n{net.cell(1).nodes}")

# net.delete_stimuli()
net.cell(0).stimulate(current)


net.make_trainable("IonotropicSynapse_gS")

print("trainables", net.get_parameters())

v = jx.integrate(net, params=net.get_parameters(), t_max=200)
print("v shape", v.shape)
print(net.edges)

plt.plot(v[0], color="blue")  # Plot voltage trace.
plt.plot(v[1], color="yellow")  # Plot voltage trace.

plt.figure(2)
plt.plot(v[3], color="green")  # Plot voltage trace.
plt.plot(v[5], color="red")  # Plot voltage trace.
plt.plot(v[7], color="blue")  # Plot voltage trace.

print("last m, n, h", v[3][-1], v[5][-1], v[7][-1])

net.compute_xyz()
net.arrange_in_layers(layers=[1, 1], within_layer_offset=150, between_layer_offset=200)

fig, ax = plt.subplots(1, 1, figsize=(3, 6))
_ = net.vis(ax=ax, detail="full")

plt.show()
