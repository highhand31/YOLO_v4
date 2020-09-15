import os
import tensorflow as tf
from src.YOLO import YOLO
from utils.misc_utils import load_weights


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
    model_path = r"C:\Users\User\Downloads\yolov4 (1).weights"
    output_dir = r"G:\我的雲端硬碟\Python\Code\Pycharm\YOLO_V4\yolo_weights"
    size = 416
    convert_weight(model_path, output_dir, size=size)








