# coding:utf-8
# learning rate
from src import Log
import tensorflow as tf

# configuration learning rate
def config_lr(lr_type, lr_init, lr_lower=1e-6, piecewise_boundaries=None, piecewise_values=None, epoch=0):
    Log.add_log("message:configuration lr:'" + str(lr_type) + "', initial lr:"+str(lr_init))
    if lr_type == 'piecewise':
        lr = tf.compat.v1.train.piecewise_constant(epoch, 
                                                    piecewise_boundaries, piecewise_values)
    elif lr_type == 'exponential':
        lr = tf.compat.v1.train.exponential_decay(learning_rate=lr_init,
                                                    global_step=epoch, decay_steps=10, decay_rate=0.99, staircase=True)
    elif lr_type =='constant':
        lr = lr_init
    else:
        Log.add_log("error:unsupported lr type:'" + str(lr_type) + "'")
        raise ValueError(str(lr_type) + ": unsupported type of learning rate")

    return tf.maximum(lr, lr_lower)