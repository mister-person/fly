import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import plt_thread

from jax import config
# config.update("jax_debug_nans", True)
# config.update("jax_debug_infs", True)
# config.update("jax_disable_jit", True)

h_size = 6

key = jax.random.key(32)

inp = jnp.array([1, 2, 3, 4], dtype=float)

subkey, key = jax.random.split(key)
mlp1 = jax.random.uniform(subkey, (h_size, 4), dtype=float)
mlp2 = jnp.zeros((4, h_size), dtype=float)
b1 = jnp.zeros((h_size))

def act_grad(inp):
    # return jax.nn.sigmoid(inp)
    return jnp.tanh(inp) * .01
    # return inp * 10
    # return 0.0

@jax.custom_gradient
def act_func(inp) -> jax.Array:
    h1 = jnp.heaviside(inp, 1)
    h2 = jnp.heaviside(inp + .5, 1)
    h3 = jnp.heaviside(inp - .5, 1)
    return h1+h2+h3 - 1.5, lambda g: jax.vmap(jax.grad(act_grad))(g) #type: ignore

#26.6 tahn @40

def foo(inp, mlp1, b1, mlp2):
    m1out = mlp1 @ inp
    # act = jax.nn.relu(m1out + b1)
    act = act_func(m1out + b1)
    # act = jnp.heaviside(m1out + b1, 1)
    m2out = jnp.sum(jax.vmap(act_func)(mlp2 * act), axis=1)
    return m2out * .1

def loss(inp, mlp1, b1, mlp2):
    out = foo(inp, mlp1, b1, mlp2)
    # return jnp.sum(jnp.abs(out + 1 - (inp[0]**2 + inp[1]**2 - (inp[2] - inp[3])**2)))
    return jnp.sum(jnp.abs(out + 1 - (inp[0] + inp[1] - (inp[2] - inp[3]))))

get_grad = jax.grad(loss, argnums = (1, 2, 3))

plt = plt_thread.ThreadedPlot()

lr = 1
losses1 = []
losses2 = []
losses3 = []
graph4 = []
average_loss = []
x = 0

jitd = jax.jit(jax.vmap(get_grad, in_axes=[0, None, None, None]))
# jitd = jax.vmap(get_grad, in_axes=[0, None, None, None])
while True:
    if lr > .001:
        lr *= .9999
    batch_size = 1000
    # lr *= .99999999 ** batch_size
    x += 1

    key, subkey = jax.random.split(key)
    examples = jax.random.uniform(subkey, (batch_size, 4), float, -10, 10)

    grad1, gradb1, grad2 = jitd(examples, mlp1, b1, mlp2)
    if not (jnp.any(jnp.isnan(grad1)) or jnp.any(jnp.isnan(grad2))):
        mlp1 -= jnp.sum(grad1, axis=0) * lr / batch_size
        b1 -= jnp.sum(gradb1, axis=0) * lr / batch_size
        mlp2 -= jnp.sum(grad2, axis=0) * lr / batch_size

    actual_loss = jax.vmap(loss, in_axes = [0, None, None, None])(examples, mlp1, b1, mlp2)
    average_loss.append(actual_loss)

    if x % 100 == 0:
        # print(mlp1)
        print("learning rate", lr)
        # print("mlp2", mlp2)
        print("loss", jnp.mean(jnp.array(average_loss)))
        losses1.append(jnp.mean(jnp.array(average_loss)))
        average_loss.clear()

        ones_loss = loss(jnp.ones(4), mlp1, b1, mlp2)
        losses2.append(ones_loss)
        loss_1234 = loss(1 + jnp.arange(4), mlp1, b1, mlp2)
        losses3.append(loss_1234)

        graph4.append(jnp.max(jnp.abs(mlp1)))
        plt.cla()
        plt.plot(jnp.array(losses1))
        plt.plot(jnp.array(losses2))
        plt.plot(jnp.array(losses3))
        # plt.plot(graph4)
