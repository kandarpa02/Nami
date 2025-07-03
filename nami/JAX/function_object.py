import jax
import jax.numpy as jnp

class Nami:
    def __init__(self):
        pass
    
    @staticmethod
    def __call__(_x, params=None, _a=1.0, _b=1.5, _w=0.3):
        _x = _x.astype(jnp.float32)
        _a, _b, _w = map(lambda x: jnp.array(x, dtype=jnp.float32), (_a, _b, _w))

        if params is not None:
            if isinstance(params, dict):
                a, b, w = params['a'], params['b'], params['w']
            elif isinstance(params, list):
                a, b, w = params[0], params[1], params[2]
            a = a.astype(jnp.float32)
            b = b.astype(jnp.float32)
            w = w.astype(jnp.float32)
        else:
            a, b, w = _a, _b, _w

        w = jnp.clip(w, 0.1, 0.5)
        a = jnp.clip(a, 0.5, 3.0)
        b = jnp.clip(b, 0.5, 3.0)

        result = jnp.where(_x > 0, jax.nn.tanh(_x * a), a * jnp.sin(_x * w) / b)
        return result.astype(_x.dtype)
