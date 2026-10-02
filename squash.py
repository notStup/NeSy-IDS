import tensorflow as tf
import numpy as np


def squash(s, axis=-1, epsilon=1e-7, name=None):

    with tf.name_scope(name or "squash"):

        original_dtype = s.dtype

        # Usiamo float64 internamente per evitare overflow
        # con i valori molto grandi presenti in CICIDS2017.
        s64 = tf.cast(s, tf.float64)

        squared_norm = tf.reduce_sum(
            tf.square(s64),
            axis=axis,
            keepdims=True
        )

        safe_norm = tf.sqrt(
            squared_norm + epsilon
        )

        squash_factor = squared_norm / (
            1.0 + squared_norm
        )

        unit_vector = s64 / safe_norm

        output = squash_factor * unit_vector

        return tf.cast(
            output,
            original_dtype
        )


def safe_norm(
    s,
    axis=-1,
    epsilon=1e-7,
    keep_dims=False,
    name=None
):

    with tf.name_scope(name or "safe_norm"):

        original_dtype = s.dtype

        s64 = tf.cast(s, tf.float64)

        squared_norm = tf.reduce_sum(
            tf.square(s64),
            axis=axis,
            keepdims=keep_dims
        )

        norm = tf.sqrt(
            squared_norm + epsilon
        )

        return tf.cast(
            norm,
            original_dtype
        )


def squash_arr(s, axis=-1, epsilon=1e-7):

    squared_norm = np.sum(
        np.square(s),
        axis=axis
    )

    safe_norm = np.sqrt(
        squared_norm + epsilon
    )

    squash_factor = squared_norm / (
        1.0 + squared_norm
    )

    unit_vector = s / safe_norm

    return squash_factor * unit_vector