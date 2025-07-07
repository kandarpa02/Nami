try:
    import jax
    import jax.numpy as jnp
    import hauku as hk
except:
    raise ImportError("JAX/Haiku is not installed")

def nami(x, params=None, a=1.0, b=1.5, w=0.3):
    x = x.astype(jnp.float32)
    a = jnp.array(a, dtype=jnp.float32)
    b = jnp.array(b, dtype=jnp.float32)
    w = jnp.array(w, dtype=jnp.float32)

    if params is not None:
        if isinstance(params, dict):
            a, b, w = params['a'], params['b'], params['w']
        elif isinstance(params, (list, tuple)):
            a, b, w = params
        a = jnp.array(a, dtype=jnp.float32)
        b = jnp.array(b, dtype=jnp.float32)
        w = jnp.array(w, dtype=jnp.float32)

    w = jnp.clip(w, 0.1, 0.5)
    a = jnp.clip(a, 0.5, 3.0)
    b = jnp.clip(b, 0.5, 3.0)

    out = jnp.where(x > 0, jax.nn.tanh(x * a), a * jnp.sin(x * w) / b)
    return out.astype(x.dtype)


class Nami(hk.Module):
    def __init__(self, name='nami'):
        super().__init__(name=name)

    def __call__(self, x):
        a = hk.get_parameter("a", shape=(), init=hk.initializers.Constant(1.0))
        b = hk.get_parameter("b", shape=(), init=hk.initializers.Constant(1.5))
        w = hk.get_parameter("w", shape=(), init=hk.initializers.Constant(0.3))
        return nami(x, params={'a': a, 'b': b, 'w': w})
