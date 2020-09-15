# coding:utf-8
# optimizer
import tensorflow as tf
from src import Log

# configuration optimizer
def config_optimizer(optimizer_name, lr, momentum=0.99):
    Log.add_log("message: configuration optimizer:'" + str(optimizer_name) + "'")
    if optimizer_name == 'momentum':
        return tf.compat.v1.train.MomentumOptimizer(lr, momentum=momentum)
    elif optimizer_name == 'adam':
        return tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
    elif optimizer_name == 'sgd':
        return tf.compat.v1.train.GradientDescentOptimizer(learning_rate=lr)
    else:
        Log.add_log("error:unsupported type of optimizer:'" + str(optimizer_name) + "'")
        raise ValueError(str(optimizer_name) + ":unsupported type of optimizer")