
try:
    import tensorflow as tf
except:
    raise ImportError('TensorFlow is not installed')

class Nami(tf.keras.layers.Layer):
    def __init__(self, w_init=0.3, a_init=1.0, b_init=1.5, learnable=True, **kwargs):
        super().__init__(**kwargs)
        self.learnable = learnable

        def create_weight(name, init_val):
            return self.add_weight(
                name=name,
                shape=(),
                initializer=tf.constant_initializer(init_val),
                trainable=self.learnable,
            )

        self._w = create_weight("w", w_init)
        self._a = create_weight("a", a_init)
        self._b = create_weight("b", b_init)

    def call(self, x):
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float32)

        w = tf.clip_by_value(tf.cast(self._w, tf.float32), 0.1, 0.5)
        a = tf.clip_by_value(tf.cast(self._a, tf.float32), 0.5, 3.0)
        b = tf.clip_by_value(tf.cast(self._b, tf.float32), 0.5, 3.0)

        out = tf.where(
            x > 0,
            tf.math.tanh(x * a),
            a * tf.math.sin(x * w) / b
        )

        return tf.cast(out, orig_dtype)
