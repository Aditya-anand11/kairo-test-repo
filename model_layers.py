"""
model_layers.py — custom Keras layers shared by training (03) and inference (04).
"""
import tensorflow as tf
from tensorflow.keras import layers


class L2NormalizeLayer(layers.Layer):
    """Projects embeddings onto the unit hypersphere so the dot product
    between user/car embeddings is exactly cosine similarity."""
    def call(self, x):
        return tf.math.l2_normalize(x, axis=-1)


class ScaledCosineSimilarity(layers.Layer):
    """Learnable-temperature scaled dot product between two L2-normalized
    embeddings, squashed through a sigmoid to produce a match probability."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.log_temp = self.add_weight(name="log_temp", shape=(), initializer="zeros", trainable=True)

    def call(self, inputs):
        user_emb, car_emb = inputs
        cosine = tf.reduce_sum(user_emb * car_emb, axis=-1, keepdims=True)
        temp = tf.exp(self.log_temp) + 1e-3
        return tf.sigmoid(cosine * temp)