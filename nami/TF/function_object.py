
try:
    import tensorflow as tf
except:
    raise ImportError('TensorFlow is not installed')

class Nami(tf.keras.layers.Layer):
    """
    Nami Activation Function (TF Version)

    Nami means wave in Japanese, the name came from its wavy nature in the negative domain
    due to the `sin` function, rather than tending to one value like other functions
    `Nami` oscillates in the negative side, and has the smoothness of `tanh`. According to
    the training data the oscilation is maintained by three learnable parameters: `w`, `a`, `b`.

    Parameters:
        w: Controls wavelength of sin (smoothing)
        a: Controls amplitude of wave
        b: Regulates overfitting by suppressing a
        learnable: Whether parameters are trainable
    """

    def __init__(self, w_init=0.3, a_init=1.0, b_init=1.5, learnable=True, **kwargs):
        super().__init__(**kwargs)
        self.learnable = learnable

        def create_weight(name, init_val, trainable):
            return self.add_weight(
                name=name,
                shape=(),
                initializer=tf.constant_initializer(init_val),
                trainable=trainable,
            )

        self._w = create_weight("w", w_init, learnable)
        self._a = create_weight("a", a_init, learnable)
        self._b = create_weight("b", b_init, learnable)

    def call(self, x):
        w = tf.clip_by_value(self._w, 0.1, 0.5)
        a = tf.clip_by_value(self._a, 0.5, 3.0)
        b = tf.clip_by_value(self._b, 0.5, 3.0)

        return tf.where(
            x > 0,
            tf.math.tanh(x * a),
            a * tf.math.sin(w * x) / b
        )
