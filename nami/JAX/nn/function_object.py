import jax.numpy as jnp
import jax

class Nami:
    def __init__(self):
        pass
    
    @staticmethod
    def __call__(_x, params, _a = 1.0, _b = 1.5, _w = 0.3, learnable=True):
        if learnable==True:
            if isinstance(params, dict):
                a, b, w = params['a'], params['b'], params['w']
            elif isinstance(params, list):
                a, b, w = params[0], params[1], params[2]
        else:
             a, b, w = jnp.array(_a), jnp.array(_b), jnp.array(_w)
        
        w = jnp.clip(w, min=0.1, max=0.5)
        a = jnp.clip(a, min=0.5, max=3.0)
        b = jnp.clip(b, min=0.5, max=3.0)
    

        return jnp.where(_x > 0, jax.nn.tanh(_x * a) , a * jnp.sin(_x * w)/b)

