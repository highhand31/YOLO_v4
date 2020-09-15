# coding:utf-8
# 
import tensorflow as tf

# 得到预测的box
def get_predict_result(feature_1, feature_2, feature_3, anchor_1, anchor_2, anchor_3, width, height, class_num, score_thresh=0.5, iou_thresh=0.5, max_box=20):
    '''
    feature_13:[batch_size, 13, 13, 3*5]
    feature_26:[batch_size, 26, 26, 3*5]
    return:
        boxes:[V, 4]    item [x_min, y_min, x_max, y_max]
        score:[V, 1]
        label:[V, 1]
    '''
    boxes, conf, prob = __get_pred_box(feature_1, feature_2, feature_3, anchor_1, anchor_2, anchor_2, width, height)
    score = conf * prob
    boxes, score, label = __nms(boxes, score, class_num, max_boxes=max_box, score_thresh=score_thresh, iou_threshold=iou_thresh)
    return boxes, score, label

# feature parse
def __decode_feature(feature, anchor, width, height):
    shape = tf.shape(feature) 
    shape = tf.cast(shape, tf.float32)
    # [batch_size, 13, 13, 3, 5+class_num]
    yi_pred = tf.reshape(feature, [shape[0], shape[1], shape[2], 3, -1])
    # shape : [batch_size,13,13,3,2] [batch_size,13,13,3,2] [batch_size,13,13,3,1] [batch_size,13,13,3, class_num]
    xy, wh, conf, prob = tf.split(yi_pred, [2, 2, 1, -1], axis=-1)

    ''' compute offset of x and y ''' 
    offset_x = tf.range(shape[2], dtype=tf.float32) #width
    offset_y = tf.range(shape[1], dtype=tf.float32) # height
    offset_x, offset_y = tf.meshgrid(offset_x, offset_y)
    offset_x = tf.reshape(offset_x, (-1, 1))
    offset_y = tf.reshape(offset_y, (-1, 1))
    offset_xy = tf.concat([offset_x, offset_y], axis=-1)
    # [13, 13, 1, 2]
    offset_xy = tf.reshape(offset_xy, [shape[1], shape[2], 1, 2])

    xy = tf.math.sigmoid(xy) + offset_xy    
    xy = xy / [shape[2], shape[1]]

    wh = tf.math.exp(wh) * anchor
    wh = wh / [width, height]

    conf = tf.math.sigmoid(conf)
    prob = tf.math.sigmoid(prob)

    return xy, wh, conf, prob

# get all predicts boxes
def __get_pred_box(feature1, feature2, feature3, anchor1, anchor2, anchor3, width, height):
    '''
    feature:[1, 13, 13, 3*5]
    return:
        boxes:[1, V, 4]:[x_min, y_min, x_max, y_max] 相对于原始图片大小的浮点数
        conf:[1, V, 1]
        prob:[1, V, class_num]
    '''
    # decode
    xy1, wh1, conf1, prob1 = __decode_feature(feature1, anchor1, width, height)
    xy2, wh2, conf2, prob2 = __decode_feature(feature2, anchor2, width, height)
    xy3, wh3, conf3, prob3 = __decode_feature(feature3, anchor3, width, height)

    # gather box
    def _reshape(xy, wh, conf, prob):
        # [1, 13, 13, 3, 1]
        x_min = xy[..., 0: 1] - wh[..., 0: 1] / 2.0
        x_max = xy[..., 0: 1] + wh[..., 0: 1] / 2.0
        y_min = xy[..., 1: 2] - wh[..., 1: 2] / 2.0
        y_max = xy[..., 1: 2] + wh[..., 1: 2] / 2.0

        # [1, 13, 13, 3, 4]
        boxes = tf.concat([x_min, y_min, x_max, y_max], -1)
        shape = tf.shape(boxes)
        # [1, 13*13*3, 4]
        boxes = tf.reshape(boxes, (shape[0], shape[1] * shape[2]* shape[3], shape[4]))

        # [1, 13 * 13 * 3, 1]
        conf = tf.reshape(conf, (shape[0], shape[1] * shape[2]* shape[3], 1))

        # [1, 13*13*3, class_num]
        prob = tf.reshape(prob, (shape[0], shape[1] * shape[2]* shape[3], -1))
    
        return boxes, conf, prob

    # reshape
    # [batch_size, 13*13*3, 4], [batch_size, 13*13*3, 1], [batch_size, 13*13*3, class_num]
    boxes_1, conf_1, prob_1 = _reshape(xy1, wh1, conf1, prob1)
    boxes_2, conf_2, prob_2 = _reshape(xy2, wh2, conf2, prob2)
    boxes_3, conf_3, prob_3 = _reshape(xy3, wh3, conf3, prob3)

    # gather
    # [1, 13*13*3, 4] & [1, 26*26*3, 4] ==> [1,  V, 4]
    boxes = tf.concat([boxes_1, boxes_2, boxes_3], 1)
    conf = tf.concat([conf_1, conf_2, conf_3], 1)
    prob = tf.concat([prob_1, prob_2, prob_3], 1)

    return boxes, conf, prob


# NMS
def __nms(boxes, scores, class_num, max_boxes=50, score_thresh=0.5, iou_threshold=0.5):
    '''
    boxes:[1, V, 4]
    score:[1, V, class_num]
    return:????
        boxes:[V, 4]
        score:[V,]
    '''
    boxes_list, label_list, score_list = [], [], []
    max_boxes = tf.constant(max_boxes, dtype='int32')

    # [V, 4]
    boxes = tf.reshape(boxes, [-1, 4])
    # [V, class_num]
    score = tf.reshape(scores, [-1, class_num])

    mask = tf.greater_equal(score, tf.constant(score_thresh))
    for i in range(class_num):
        # Step 3: Apply the mask to scores, boxes and pick them out
        filter_boxes = tf.boolean_mask(boxes, mask[:,i])
        filter_score = tf.boolean_mask(score[:,i], mask[:,i])
        nms_indices = tf.image.non_max_suppression(boxes=filter_boxes,
                                                scores=filter_score,
                                                max_output_size=max_boxes,
                                                iou_threshold=iou_threshold, name='nms_indices')
        label_list.append(tf.ones_like(tf.gather(filter_score, nms_indices), 'int32')*i)
        boxes_list.append(tf.gather(filter_boxes, nms_indices))
        score_list.append(tf.gather(filter_score, nms_indices))

    # stack
    boxes = tf.concat(boxes_list, axis=0)
    score = tf.concat(score_list, axis=0)
    label = tf.concat(label_list, axis=0)

    return boxes, score, label

    