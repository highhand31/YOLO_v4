# coding:utf-8
# loss function of yolo
import tensorflow as tf

class Loss():
    def __init__(self):
        pass

    def yolo_loss(self, ls_features, ls_labels, ls_anchors, \
                                    width, height, class_num,cls_normalizer=1.0,  \
                                    iou_normalizer=0.07, iou_thresh=0.5, \
                                    prob_thresh=0.25, score_thresh=0.25):
        loss_total = 0
        loss_xy = 0
        loss_wh = 0
        loss_conf = 0
        loss_prob = 0
        loss_ciou = 0
            
        xy, wh, conf, prob = self.__decode_feature(ls_features[0], ls_anchors[0], width, height, class_num)
        total_loss, xy_loss, wh_loss, conf_loss, class_loss, ciou_loss = self.__compute_loss_v4(\
                                                        xy, wh, conf, prob, ls_labels[0], class_num, \
                                                        cls_normalizer=cls_normalizer, \
                                                        iou_normalizer=iou_normalizer, \
                                                        iou_thresh=iou_thresh, \
                                                        prob_thresh=prob_thresh, \
                                                        score_thresh=score_thresh)
        loss_total += total_loss
        loss_xy += xy_loss
        loss_wh += wh_loss
        loss_conf += conf_loss
        loss_prob += class_loss
        loss_ciou += ciou_loss
            
        xy, wh, conf, prob = self.__decode_feature(ls_features[1], ls_anchors[1], width, height, class_num)
        total_loss, xy_loss, wh_loss, conf_loss, class_loss, ciou_loss = self.__compute_loss_v4(\
                                                        xy, wh, conf, prob, ls_labels[1], class_num, \
                                                        cls_normalizer=cls_normalizer, \
                                                        iou_normalizer=iou_normalizer, \
                                                        iou_thresh=iou_thresh, \
                                                        prob_thresh=prob_thresh, \
                                                        score_thresh=score_thresh)
        loss_total += total_loss
        loss_xy += xy_loss
        loss_wh += wh_loss
        loss_conf += conf_loss
        loss_prob += class_loss
        loss_ciou += ciou_loss
            
        xy, wh, conf, prob = self.__decode_feature(ls_features[2], ls_anchors[2], width, height, class_num)
        total_loss, xy_loss, wh_loss, conf_loss, class_loss, ciou_loss = self.__compute_loss_v4(\
                                                        xy, wh, conf, prob, ls_labels[2], class_num, \
                                                        cls_normalizer=cls_normalizer, \
                                                        iou_normalizer=iou_normalizer, \
                                                        iou_thresh=iou_thresh, \
                                                        prob_thresh=prob_thresh, \
                                                        score_thresh=score_thresh)
        loss_total += total_loss
        loss_xy += xy_loss
        loss_wh += wh_loss
        loss_conf += conf_loss
        loss_prob += class_loss
        loss_ciou += ciou_loss

        return loss_total

    # 计算最大的 IOU, GIOU
    def __IOU(self, pre_xy, pre_wh, valid_yi_true):
        # [13, 13, 3, 2] ==> [13, 13, 3, 1, 2]
        pre_xy = tf.expand_dims(pre_xy, -2)
        pre_wh = tf.expand_dims(pre_wh, -2)

        # [V, 2]
        yi_true_xy = valid_yi_true[..., 0:2]
        yi_true_wh = valid_yi_true[..., 2:4]

        # 相交区域左上角坐标 : [13, 13, 3, 1, 2] & [V, 2] ==> [13, 13, 3, V, 2]
        intersection_left_top = tf.maximum((pre_xy - pre_wh / 2), (yi_true_xy - yi_true_wh / 2))
        # 相交区域右下角坐标
        intersection_right_bottom = tf.minimum((pre_xy + pre_wh / 2), (yi_true_xy + yi_true_wh / 2))
        # 并集区域左上角坐标 
        combine_left_top = tf.minimum((pre_xy - pre_wh / 2), (yi_true_xy - yi_true_wh / 2))
        # 并集区域右下角坐标
        combine_right_bottom = tf.maximum((pre_xy + pre_wh / 2), (yi_true_xy + yi_true_wh / 2))

        # 相交区域宽高 [13, 13, 3, V, 2] == > [13, 13, 3, V, 2]
        intersection_wh = tf.maximum(intersection_right_bottom - intersection_left_top, 0.0)
        # 并集区域宽高 : 这里其实不用 tf.max 也可以，因为右下坐标一定大于左上
        combine_wh = tf.maximum(combine_right_bottom - combine_left_top, 0.0)
        
        # 相交区域面积 : [13, 13, 3, V]
        intersection_area = intersection_wh[..., 0] * intersection_wh[..., 1]
        # 预测box面积 : [13, 13, 3, 1]
        pre_area = pre_wh[..., 0] * pre_wh[..., 1]
        # 真值 box 面积 : [V]
        true_area = yi_true_wh[..., 0] * yi_true_wh[..., 1]
        # [V] ==> [1, V]
        true_area = tf.expand_dims(true_area, axis=0)
        # iou : [13, 13, 3, V]
        iou = intersection_area / (pre_area + true_area - intersection_area + 1e-10)    # 防止除0

        # 并集区域面积 : [13, 13, 3, V, 1] ==> [13, 13, 3, V] 
        combine_area = combine_wh[..., 0] * combine_wh[..., 1]
        # giou : [13, 13, 3, V]
        giou = (intersection_area+1e-10) / combine_area # 加上一个很小的数字，确保 giou 存在
        
        return iou, giou

    # 计算 CIOU 损失
    def __my_CIOU_loss(self, pre_xy, pre_wh, yi_box):
        # [batch_size, 13, 13, 3, 2]
        yi_true_xy = yi_box[..., 0:2]
        yi_true_wh = yi_box[..., 2:4]

        # 上下左右
        pre_lt = pre_xy - pre_wh/2
        pre_rb = pre_xy + pre_wh/2
        truth_lt = yi_true_xy - yi_true_wh / 2
        truth_rb = yi_true_xy + yi_true_wh / 2

        # 相交区域坐标 : [batch_size, 13, 13, 3,2]
        intersection_left_top = tf.maximum(pre_lt, truth_lt)
        intersection_right_bottom = tf.minimum(pre_rb, truth_rb)
        # 相交区域宽高 : [batch_size, 13, 13, 3, 2]
        intersection_wh = tf.maximum(intersection_right_bottom - intersection_left_top, 0.0)
        # 相交区域面积 : [batch_size, 13, 13, 3, 1]
        intersection_area = intersection_wh[..., 0:1] * intersection_wh[..., 1:2]
        # 并集区域左上角坐标 
        combine_left_top = tf.minimum(pre_lt, truth_lt)
        # 并集区域右下角坐标
        combine_right_bottom = tf.maximum(pre_rb, truth_rb)
        # 并集区域宽高 : 这里其实不用 tf.max 也可以，因为右下坐标一定大于左上
        combine_wh = tf.maximum(combine_right_bottom - combine_left_top, 0.0)

        # 并集区域对角线 : [batch_size, 13, 13, 3, 1]
        C = tf.square(combine_wh[..., 0:1]) + tf.square(combine_wh[..., 1:2])
        # 中心点对角线:[batch_size, 13, 13, 3, 1]
        D = tf.square(yi_true_xy[..., 0:1] - pre_xy[..., 0:1]) + tf.square(yi_true_xy[..., 1:2] - pre_xy[..., 1:2])

        # box面积 : [batch_size, 13, 13, 3, 1]
        pre_area = pre_wh[..., 0:1] * pre_wh[..., 1:2]
        true_area = yi_true_wh[..., 0:1] * yi_true_wh[..., 1:2]

        # iou : [batch_size, 13, 13, 3, 1]
        iou = intersection_area / (pre_area + true_area - intersection_area)

        pi = 3.14159265358979323846

        # 衡量长宽比一致性的参数:[batch_size, 13, 13, 3, 1]
        v = 4 / (pi * pi) * tf.square( 
                                    tf.subtract(
                                        tf.math.atan(yi_true_wh[..., 0:1] / yi_true_wh[..., 1:2]),
                                        tf.math.atan(pre_wh[...,0:1] / pre_wh[..., 1:2])
                                        )
                                    )

        # trade-off 参数
        # alpha
        alpha = v / (1.0 - iou + v)
        ciou_loss = 1.0 - iou + D / C +  alpha * v
        return ciou_loss

    # 得到低iou的地方
    def __get_low_iou_mask(self, pre_xy, pre_wh, yi_true, iou_thresh=0.5):
        # 置信度:[batch_size, 13, 13, 3, 1]
        conf_yi_true = yi_true[..., 4:5]

        # iou小的地方
        low_iou_mask = tf.TensorArray(tf.bool, size=0, dynamic_size=True)
        # batch_size
        N = tf.shape(yi_true)[0]
        
        def loop_cond(index, low_iou_mask):
            return tf.less(index, N)        
        def loop_body(index, low_iou_mask):
            # 一张图片的全部真值 : [13, 13, 3, class_num+5] & [13, 13, 3, 1] == > [V, class_num + 5]
            mask = conf_yi_true[index, ..., 0] > 0.5
            valid_yi_true = tf.boolean_mask(yi_true[index], mask)
            # 计算 iou/ giou : [13, 13, 3, V]
            iou, _ = self.__IOU(pre_xy[index], pre_wh[index], valid_yi_true)

            # [13, 13, 3]
            best_giou = tf.reduce_max(iou, axis=-1)
            # [13, 13, 3]
            low_iou_mask_tmp = best_giou < iou_thresh
            # [13, 13, 3, 1]
            low_iou_mask_tmp = tf.expand_dims(low_iou_mask_tmp, -1)
            # 写入
            low_iou_mask = low_iou_mask.write(index, low_iou_mask_tmp)
            return index+1, low_iou_mask

        _, low_iou_mask = tf.while_loop(cond=loop_cond, body=loop_body, loop_vars=[0, low_iou_mask])
        # 拼接:[batch_size, 13, 13, 3, 1]
        low_iou_mask = low_iou_mask.stack()
        return low_iou_mask

    # 得到预测物体概率低的地方的掩码
    def __get_low_prob_mask(self, prob, prob_thresh=0.25):
        # [batch_size, 13, 13, 3, 1]
        max_prob = tf.reduce_max(prob, axis=-1, keepdims=True)
        low_prob_mask = max_prob < prob_thresh        
        return low_prob_mask

    # 对预测值进行解码
    def __decode_feature(self, yi_pred, curr_anchors, width, height, class_num):
        '''
        yi_pred:[batch_size, 13, 13, 3 * (class_num + 5)]
        curr_anchors:[3,2], 这一层对应的 anchor, 真实值
        return:
            xy:[batch_size, 13, 13, 3, 2], 相对于原图的中心坐标
            wh:[batch_size, 13, 13, 3, 2], 相对于原图的宽高
            conf:[batch_size, 13, 13, 3, 1]
            prob:[batch_size, 13, 13, 3, class_num]
        '''
        shape = tf.shape(yi_pred) 
        shape = tf.cast(shape, tf.float32)
        # [batch_size, 13, 13, 3, class_num + 5]
        yi_pred = tf.reshape(yi_pred, [shape[0], shape[1], shape[2], 3, 5 + class_num])
        # 分割预测值
        # shape : [batch_size,13,13,3,2] [batch_size,13,13,3,2] [batch_size,13,13,3,1] [batch_size,13,13,3, class_num]
        xy, wh, conf, prob = tf.split(yi_pred, [2, 2, 1, class_num], axis=-1)

        ''' 计算 x,y 的坐标偏移 '''
        offset_x = tf.range(shape[2], dtype=tf.float32) #宽
        offset_y = tf.range(shape[1], dtype=tf.float32) # 高
        offset_x, offset_y = tf.meshgrid(offset_x, offset_y)
        offset_x = tf.reshape(offset_x, (-1, 1))
        offset_y = tf.reshape(offset_y, (-1, 1))
        # 把 offset_x, offset_y 合并成一个坐标网格 [13*13, 2], 每个元素是当前坐标 (x, y)
        offset_xy = tf.concat([offset_x, offset_y], axis=-1)
        # [13, 13, 1, 2]
        offset_xy = tf.reshape(offset_xy, [shape[1], shape[2], 1, 2])
        
        xy = tf.math.sigmoid(xy) + offset_xy    
        xy = xy / [shape[2], shape[1]]

        wh = tf.math.exp(wh) * curr_anchors
        wh = wh / [width, height]

        return xy, wh, conf, prob

    # 计算损失, yolov4
    def __compute_loss_v4(self, xy, wh, conf, prob, yi_true, class_num, \
                                                            cls_normalizer=1.0, iou_thresh=0.5, \
                                                            prob_thresh=0.25, score_thresh=0.25, \
                                                            iou_normalizer=0.07):
        # mask of low iou 
        low_iou_mask = self.__get_low_iou_mask(xy, wh, yi_true, iou_thresh=iou_thresh)
        # mask of low prob
        low_prob_mask = self.__get_low_prob_mask(prob, prob_thresh=prob_thresh)        
        # mask of low iou or low prob
        low_iou_prob_mask = tf.math.logical_or(low_iou_mask, low_prob_mask)
        low_iou_prob_mask = tf.cast(low_iou_prob_mask, tf.float32)

        # batch_size
        N = tf.shape(xy)[0]
        N = tf.cast(N, tf.float32)

        # [batch_size, 13, 13, 3, 1]
        conf_scale = wh[..., 0:1] * wh[..., 1:2]    # scale = tf.square(tf.minimum(2-wh[]^2, 2))
        conf_scale = tf.where(tf.math.greater(conf_scale, 0),
                                                        tf.math.sqrt(conf_scale), conf_scale)
        conf_scale = tf.math.square(conf_scale * cls_normalizer)
        # [batch_size, 13, 13, 3, 1]
        no_obj_mask = yi_true[..., 4:5] < 0.5
        no_obj_mask = tf.cast(no_obj_mask, tf.float32)
        conf_loss_no_obj = tf.nn.sigmoid_cross_entropy_with_logits(
                                                            labels=yi_true[:,:,:,:,4:5], logits=conf
                                                            ) * conf_scale * no_obj_mask * low_iou_prob_mask
        # [batch_size, 13, 13, 3, 1]
        obj_mask = 1.0 - no_obj_mask
        conf_loss_obj = tf.nn.sigmoid_cross_entropy_with_logits(
                                                            labels=yi_true[:,:,:,:,4:5], logits=conf
                                                            )* tf.square(cls_normalizer) * obj_mask        
        # 置信度损失
        conf_loss = conf_loss_obj + conf_loss_no_obj
        conf_loss = tf.clip_by_value(conf_loss, 0.0, 1e3)
        conf_loss = tf.reduce_sum(conf_loss) / N

        # ciou_loss
        yi_true_ciou = tf.where(tf.math.less(yi_true[..., 0:4], 1e-10),
                                                tf.ones_like(yi_true[..., 0:4]), yi_true[..., 0:4])
        pre_xy = tf.where(tf.math.less(xy, 1e-10),
                                                tf.ones_like(xy), xy)
        pre_wh = tf.where(tf.math.less(wh, 1e-10),
                                                tf.ones_like(wh), wh)
        ciou_loss = self.__my_CIOU_loss(pre_xy, pre_wh, yi_true_ciou)
        ciou_loss = tf.where(tf.math.greater(obj_mask, 0.5), ciou_loss, tf.zeros_like(ciou_loss))
        ciou_loss = tf.square(ciou_loss * obj_mask) * iou_normalizer
        ciou_loss = tf.clip_by_value(ciou_loss, 0, 1e3)
        ciou_loss = tf.reduce_sum(ciou_loss) / N
        ciou_loss = tf.clip_by_value(ciou_loss, 0, 1e4)

        # xy 损失
        xy = tf.clip_by_value(xy, 1e-10, 1e4)
        xy_loss = obj_mask * tf.square(yi_true[..., 0: 2] - xy)
        xy_loss = tf.clip_by_value(xy_loss, 0.0, 1e3)
        xy_loss = tf.reduce_sum(xy_loss) / N
        xy_loss = tf.clip_by_value(xy_loss, 0.0, 1e4)

        # wh 损失
        wh_y_true = tf.where(condition=tf.math.less(yi_true[..., 2:4], 1e-10),
                                        x=tf.ones_like(yi_true[..., 2: 4]), y=yi_true[..., 2: 4])
        wh_y_pred = tf.where(condition=tf.math.less(wh, 1e-10),
                                        x=tf.ones_like(wh), y=wh)
        wh_y_true = tf.clip_by_value(wh_y_true, 1e-10, 1e10)
        wh_y_pred = tf.clip_by_value(wh_y_pred, 1e-10, 1e10)
        wh_y_true = tf.math.log(wh_y_true)
        wh_y_pred = tf.math.log(wh_y_pred)

        wh_loss = obj_mask * tf.square(wh_y_true - wh_y_pred)
        wh_loss = tf.clip_by_value(wh_loss, 0.0, 1e3)
        wh_loss = tf.reduce_sum(wh_loss) / N
        wh_loss = tf.clip_by_value(wh_loss, 0.0, 1e4)
        
        # prob 损失
        # [batch_size, 13, 13, 3, class_num]
        prob_score = prob * conf
        
        high_score_mask = prob_score > score_thresh
        high_score_mask = tf.cast(high_score_mask, tf.float32)
        
        class_loss_no_obj = tf.nn.sigmoid_cross_entropy_with_logits(
                                                        labels=yi_true[...,5:5+class_num],
                                                        logits=prob 
                                                    ) * low_iou_prob_mask * no_obj_mask * high_score_mask
        
        class_loss_obj = tf.nn.sigmoid_cross_entropy_with_logits(
                                                        labels=yi_true[...,5:5+class_num],
                                                        logits=prob
                                                    ) * obj_mask

        class_loss = class_loss_no_obj + class_loss_obj        
        class_loss = tf.clip_by_value(class_loss, 0.0, 1e3)
        class_loss = tf.reduce_sum(class_loss) / N

        loss_total = xy_loss + wh_loss + conf_loss + class_loss + ciou_loss
        # return loss_total
        return loss_total, xy_loss, wh_loss, conf_loss, class_loss, ciou_loss
