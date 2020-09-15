# coding:utf-8
# load voc dataset
import numpy as np
from src import Log
from utils import tools
import random
import cv2
import os
from os import path

class Data():
    def __init__(self, voc_root_dir, voc_dir_ls, voc_names, class_num, batch_size, anchors, width=608, height=608, data_debug=False):
        self.data_dirs = [path.join(path.join(voc_root_dir, voc_dir), "JPEGImages") for voc_dir in voc_dir_ls] 
        self.class_num = class_num  # classify number
        self.batch_size = batch_size
        self.anchors = np.asarray(anchors).astype(np.float32).reshape([-1, 2]) / [width, height]     #[9,2]
        print("anchors:\n", self.anchors)

        self.imgs_path = []
        self.labels_path = []

        self.num_batch = 0      # total batch number
        self.num_imgs = 0       # total number of images
        self.steps_per_epoch = 1

        self.data_debug = data_debug

        self.width = width
        self.height = height

        self.names_dict = tools.word2id(voc_names)    # dictionary of name to id

        # 初始化数据增强策略的参数
        self.flip_img = 0.5    # probility of flip image
        self.is_flip = False    # 
        self.gray_img = 0.02        # probility to gray picture
        self.smooth_delta = 0.001 # label smooth delta
        self.erase_img = 0        # probility of random erase some area
        self.gasuss = 0.0       # probility of gasuss noise

        self.__init_args()
    
    # initial all parameters
    def __init_args(self):
        Log.add_log("message: begin to initial images path")

        # init imgs path
        for voc_dir in self.data_dirs:
            for img_name in os.listdir(voc_dir):
                img_path = path.join(voc_dir, img_name)
                #print("img_path:",img_path)
                label_path = img_path.replace("JPEGImages", "Annotations")
                label_path = label_path.replace(img_name.split('.')[-1], "xml")
                if not path.isfile(img_path):
                    Log.add_log("warning:VOC image'"+str(img_path)+"'is not a file")
                    continue
                if not path.isfile(label_path):
                    Log.add_log("warning:VOC label'"+str(label_path)+"'if not a file")
                    continue
                self.imgs_path.append(img_path)
                self.labels_path.append(label_path)
                self.num_imgs += 1        
        self.steps_per_epoch = int(self.num_imgs / self.batch_size)
        Log.add_log("message:initialize VOC dataset complete,  there are "+str(self.num_imgs)+" pictures in all")
        
        if self.num_imgs <= 0:
            raise ValueError("there are 0 pictures to train in all")
        
        return
        
    # read image 
    def read_img(self, img_file):
        '''
        read img_file, and resize it
        return:img, RGB & float
        '''
        img = tools.read_img(img_file)
        if img is None:
            return None, None, None, None
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
    
    # read label file
    def read_label(self, label_file, names_dict, anchors):
        '''
        parsement label_file, and generates label_y1, label_y2, label_y3
        return:label_y1, label_y2, label_y3
        '''
        contents = tools.parse_voc_xml(label_file, names_dict)  
        if not contents:
            return None, None, None

        # flip the label
        if self.is_flip:
            self.is_flip = False
            for i in range(len(contents)):
                contents[i][1] = 1.0 - contents[i][1]

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
            label_id = int(label[0])
            box = np.asarray(label[1: 5]).astype(np.float32)   # the value saved in label is x,y,w,h

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
            y_true[best_index // 3][y, x, k, 5:] = delta/self.class_num 
            y_true[best_index // 3][y, x, k, 4:5] = label_value
            y_true[best_index // 3][y, x, k, 5 + label_id] = label_value
        
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

    # load batch_size images
    def __get_data(self):
        '''
        load  batch_size labels and images
        return:imgs, label_y1, label_y2, label_y3
        '''        
        imgs = []
        labels_y1, labels_y2, labels_y3 = [], [], []

        count = 0
        while count < self.batch_size:
            curr_index = random.randint(0, self.num_imgs - 1)
            img_name = self.imgs_path[curr_index]
            label_name = self.labels_path[curr_index]

            img, ori_w, ori_h, test_img = self.read_img(img_name)
            if img is None:
                Log.add_log("img file'" + img_name + "'reading exception, will be deleted")
                self.__remove(img_name, label_name)
                continue

            label_y1, label_y2, label_y3, test_result = self.read_label(label_name, self.names_dict, self.anchors)
            if label_y1 is None:
                Log.add_log("label file'" + label_name + "'reading exception, will be deleted")
                self.__remove(img_name, label_name)
                continue
            
            # show data agument result
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
        '''    get batch images    '''
        return self.__get_data()

    


