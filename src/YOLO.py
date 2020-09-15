# coding:utf-8
# implement YOLO
import tensorflow as tf
from src import module
import numpy as np
slim = tf.contrib.slim

class YOLO():
    def __init__(self):
        pass

    def forward(self, inputs, class_num, batch_norm_decay=0.9, weight_decay=0.0005, isTrain=True, reuse=False):
        # set batch norm params
        batch_norm_params = {
            'decay': batch_norm_decay,
            'epsilon': 1e-05,
            'scale': True,
            'is_training': isTrain,
            'fused': None,  # Use fused batch norm if possible.
        }

        with slim.arg_scope([slim.conv2d, slim.batch_norm], reuse=reuse):
            # [N, 19, 19, 512], [N, 38, 38, 256], [N, 76, 76, 128]
            route_1, route_2, route_3 = module.extraction_feature(inputs, batch_norm_params, weight_decay)
            
            with slim.arg_scope([slim.conv2d], 
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params,
                                biases_initializer=None,
                                activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                                weights_regularizer=slim.l2_regularizer(weight_decay)):
                with tf.variable_scope('yolo'):
                    # features of y1
                    # [N, 76, 76, 128] => [N, 76, 76, 256]
                    net = module.conv(route_1, 256)
                    # [N, 76, 76, 256] => [N, 76, 76, 255]
                    net = slim.conv2d(net, 3*(4+1+class_num), 1,
                                                        stride=1, normalizer_fn=None,
                                                        activation_fn=None, biases_initializer=tf.zeros_initializer())  # ,scope="feature_y3"
                    feature_y3 = net    # yolo/Conv_1/BiasAdd:0

                    # features of  y2
                    # [N, 76, 76, 128] => [N, 38, 38, 256]
                    net = module.conv(route_1, 256, stride=2)
                    # [N, 38, 38, 512]
                    net = tf.concat([net, route_2], -1)
                    net = module.yolo_conv_block(net, 512, 2, 1)
                    route_147 = net
                    # [N, 38, 38, 256] => [N, 38, 38, 512]
                    net = module.conv(net, 512)
                    # [N, 38, 38, 512] => [N, 38, 38, 255]
                    net = slim.conv2d(net, 3*(4+1+class_num), 1,
                                                        stride=1, normalizer_fn=None,
                                                        activation_fn=None, biases_initializer=tf.zeros_initializer())  # , scope="feature_y2"
                    feature_y2 = net    # yolo/Conv_9/BiasAdd:0

                    # features of  y3
                    # [N, 38, 38, 256] => [N, 19, 19, 512]
                    net = module.conv(route_147, 512, stride=2)
                    net = tf.concat([net, route_3], -1)
                    net = module.yolo_conv_block(net, 1024, 3, 0)
                    # [N, 19, 19, 1024] => [N, 19, 19, 255]
                    net = slim.conv2d(net, 3*(4+1+class_num), 1,
                                                        stride=1, normalizer_fn=None,
                                                        activation_fn=None, biases_initializer=tf.zeros_initializer())  #, scope="feature_y1"
                    feature_y1 = net    # yolo/Conv_17/BiasAdd:0

        return feature_y1, feature_y2, feature_y3