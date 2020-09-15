# coding:utf-8
# Implementation of activation function
from enum import Enum
import tensorflow as tf

class activation(Enum):
    MISH = 1        # mish activation
    LEAKY_RELU = 2  # leaky_relu
    RELU = 3        # relu activation

def activation_fn(inputs, name, alpha=0.1):
    if name is activation.MISH:        
        MISH_THRESH = 20.0        # thresh in yolov4
        tmp = inputs

        inputs = tf.where(
                                    tf.math.logical_and(tf.less(tmp, MISH_THRESH), tf.greater(inputs, -MISH_THRESH)),
                                    tf.log(1 + tf.exp(tmp)), 
                                    tf.zeros_like(tmp)
                                )
        inputs = tf.where(tf.less(tmp, -MISH_THRESH), 
                                                tf.exp(tmp), 
                                                inputs)
        inputs = tf.where(tf.greater(tmp, MISH_THRESH),
                                                tmp,
                                                inputs)
        # Mish = x*tanh(ln(1+e^x))
        inputs = tmp * tf.tanh(inputs)
        return inputs
    elif name is activation.LEAKY_RELU:
        return tf.nn.leaky_relu(inputs, alpha=alpha)
    elif name is activation.RELU:
        return tf.nn.relu(inputs)
    elif name is None:
        return inputs
    else:
        ValueError("can not find activation named "+str(name) + "'")
    return None