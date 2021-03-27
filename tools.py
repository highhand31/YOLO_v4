import os
import tensorflow
from src.YOLO import YOLO
from utils.misc_utils import load_weights

#----tensorflow version check
if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
else:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
print("Tensorflow version of {}: {}".format(__file__,tf.__version__))

def convert_weight(model_path,output_dir,size=608):
    save_path = os.path.join(output_dir,'YOLO_v4_' + str(size) + '.ckpt')
    class_num = 80
    yolo = YOLO()
    with tf.Session() as sess:
        tf_input = tf.placeholder(tf.float32, [1, size, size, 3])

        feature = yolo.forward(tf_input, class_num, isTrain=False)

        saver = tf.train.Saver(var_list=tf.global_variables())

        load_ops = load_weights(tf.global_variables(), model_path)
        sess.run(load_ops)
        saver.save(sess, save_path=save_path)
        print('YOLO v4 weights have been transformed to {}'.format(save_path))


if __name__ == "__main__":
    #----convert_weight
    model_path = r"C:\Users\JohnnyKavnie\Desktop\yolo_test\yolov4.weights"
    output_dir = r"C:\Users\JohnnyKavnie\Desktop\yolo_test"
    size = 416
    convert_weight(model_path, output_dir, size=size)







