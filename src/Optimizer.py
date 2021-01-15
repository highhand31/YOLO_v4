# coding:utf-8
# optimizer
import tensorflow
from src import Log
if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
    # import tensorflow.contrib.slim as slim
else:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    # import tf_slim as slim

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