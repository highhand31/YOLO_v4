# coding:utf-8
# load own dataset
import numpy as np
from src import Log
from utils import tools
import random
import cv2

class Data():
    def __init__(self, data_file, class_num, batch_size, anchors, width=608, height=608, data_debug=False):
        self.data_file = data_file  # data dictionary
        self.class_num = class_num  # classify number
        self.batch_size = batch_size
        self.anchors = np.asarray(anchors).astype(np.float32).reshape([-1, 2]) / [width, height]     #[9,2]
        print("anchors:\n", self.anchors)

        self.data_debug = data_debug

        self.imgs_path = []
        self.labels_path = []

        self.num_batch = 0      # total batch
        self.num_imgs = 0       # total image number
        self.steps_per_epoch = 1

        self.width = width
        self.height = height

        # 初始化数据增强策略的参数
        self.flip_img = 0    # probility of flip image
        self.is_flip = False    # 
        self.gray_img = 0.02        # probility to gray picture
        self.smooth_delta = 0.001 # label smooth delta
        self.erase_img = 0        # probility of random erase some area
        self.gasuss = 0.2       # probility of gasuss noise

        self.__init_args()
    
    # initial all parameters
    def __init_args(self):
        Log.add_log("message: begin to initial images path")
        # init imgs path
        self.imgs_path = tools.read_file(self.data_file)
        if not self.imgs_path:
            Log.add_log("error:imgs_path not exist")
            raise ValueError("no file find in :" + str(self.data_file))
        self.num_imgs = len(self.imgs_path)
        self.steps_per_epoch = int(self.num_imgs / self.batch_size)
        Log.add_log("message:there are "+str(self.num_imgs) + " pictures in all")

        # init labels path
        for img in self.imgs_path:
            label_path = img.replace("JPEGImages", "labels")
            label_path = label_path.replace(img.split('.')[-1], "txt")
            self.labels_path.append(label_path)
        Log.add_log("message: initialize images path and corresponding labels complete")
        
        return
        
    # read image and do data augment
    def read_img(self, img_file):
        '''
        read img_file, and resize image
        return:img, RGB & float
        '''
        img = tools.read_img(img_file)
        if img is None:
            return None
        ori_h, ori_w, _ = img.shape
        img = cv2.resize(img, (self.width, self.height))

        # flip image
        if np.random.random() < self.flip_img:
            self.is_flip = True
            img = cv2.flip(img, 1)
                
        # gray
        if np.random.random() < self.gray_img:
            tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # convert to 3 channel
            img = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
        
        # random erase
        if np.random.random() < self.erase_img:
            erase_w = random.randint(20, 100)
            erase_h = random.randint(20, 100)
            x = random.randint(0, self.width - erase_w)
            y = random.randint(0, self.height - erase_h)
            value = random.randint(0, 255)
            img[y:y+erase_h, x:x+erase_w, : ] = value

        test_img = img

        img = img.astype(np.float32)
        img = img/255.0
        
        # gasuss noise
        if np.random.random() < self.gasuss:
            noise = np.random.normal(0, 0.01, img.shape)
            img = img+noise
            img = np.clip(img, 0, 1.0)
        return img, ori_w, ori_h, test_img
    
    # read labels
    def read_label(self, label_file, anchors, new_w, new_h):
        '''
        parsement label_file, and generates label_y1, label_y2, label_y3
        return:label_y1, label_y2, label_y3
        '''
        contents = tools.read_file(label_file)
        if not contents:
            return None, None, None

        label_y1 = np.zeros((self.height // 32, self.width // 32, 3, 5 + self.class_num), np.float32)
        label_y2 = np.zeros((self.height // 16, self.width // 16, 3, 5 + self.class_num), np.float32)
        label_y3 = np.zeros((self.height // 8, self.width // 8, 3, 5 + self.class_num), np.float32)

        delta = self.smooth_delta
        if delta:
            label_y1[:, :, :, 4] = delta  / self.class_num
            label_y2[:, :, :, 4] = delta  / self.class_num
            label_y3[:, :, :, 4] = delta  / self.class_num

        y_true = [label_y3, label_y2, label_y1]
        ratio = {0:8, 1:16, 2:32}

        test_result = []

        for label in contents:
            label = label.split()
            if len(label) != 5:
                Log.add_log("error: in file '" + str(label_file) + "', the number of parameter does not match")
                raise ValueError(str(label_file) + ": the number of label parameter does not match")
            label_id = int(label[0])
            box = np.asarray(label[1: 5]).astype(np.float32)   # what saved in label are x,y,w,h
            # flip the label
            if self.is_flip:
                box[0] = 1.0 - box[0] 
            
            test_result.append([box[0]-box[2]/2, box[1]-box[3]/2, box[0]+box[2]/2, box[1]+box[3]/2])
            
            best_giou = 0
            best_index = 0
            for i in range(len(anchors)):
                min_wh = np.minimum(box[2:4], anchors[i])
                max_wh = np.maximum(box[2:4], anchors[i])
                giou = (min_wh[0] * min_wh[1]) / (max_wh[0] * max_wh[1])
                if giou > best_giou:
                    best_giou = giou
                    best_index = i
            
            # 012->0, 345->1, 678->2
            x = int(np.floor(box[0] * self.width / ratio[best_index // 3]))
            y = int(np.floor(box[1] * self.height / ratio[best_index // 3]))
            k = best_index % 3

            y_true[best_index // 3][y, x, k, 0:4] = box
            # label smooth
            label_value = 1.0  if not delta else (1-delta)
            y_true[best_index // 3][y, x, k, 4:5] = label_value
            y_true[best_index // 3][y, x, k, 5:] = delta/self.class_num
            y_true[best_index // 3][y, x, k, 5 + label_id] = label_value
        if self.is_flip:
            self.is_flip = False
        return label_y1, label_y2, label_y3, test_result

    # remove broken file
    def __remove(self, img_file, xml_file):
        self.imgs_path.remove(img_file)
        self.labels_path.remove(xml_file)
        self.num_imgs -= 1
        if not len(self.imgs_path) == len(self.labels_path):
            print("after delete file: %s，the number of label and picture is not equal" %(img_file))
            assert(0)
        return 

    # load batch_size images and labels
    def __get_data(self):
        '''
        load batch_size labels and images data
        return:imgs, label_y1, label_y2, label_y3
        '''        
        imgs = []
        labels_y1, labels_y2, labels_y3 = [], [], []

        count = 0
        while count < self.batch_size:
            curr_index = random.randint(0, len(self.imgs_path) - 1)
            img_name = self.imgs_path[curr_index]
            label_name = self.labels_path[curr_index]

            img, ori_w, ori_h, test_img = self.read_img(img_name)
            if img is None:
                Log.add_log("img file'" + img_name + "'reading exception, will be deleted")
                self.__remove(img_name, label_name)
                continue

            label_y1, label_y2, label_y3, test_result = self.read_label(label_name, self.anchors, self.width, self.height)
            if label_y1 is None:
                Log.add_log("label file'" + label_name + "'reading exception, will be deleted")
                self.__remove(img_name, label_name)
                continue
            # show data augment result
            if self.data_debug:
                test_img = tools.draw_img(test_img, test_result, None, None, None, None)
                cv2.imshow("letterbox_img", test_img)
                cv2.waitKey(0)
            imgs.append(img)
            labels_y1.append(label_y1)
            labels_y2.append(label_y2)
            labels_y3.append(label_y3)

            count += 1

        self.num_batch += 1
        imgs = np.asarray(imgs)
        labels_y1 = np.asarray(labels_y1)
        labels_y2 = np.asarray(labels_y2)
        labels_y3 = np.asarray(labels_y3)
        
        return imgs, labels_y1, labels_y2, labels_y3

    # Iterator
    def __next__(self):
        return self.__get_data()

    


