# coding:utf-8
# tools

import os
from os import path
import time
import cv2
import random
from xml.dom.minidom import parse

'''
##################### about file #####################
'''
# read file content
def read_file(file_name):
    '''
    read all content in file_name
    return: list 
    '''
    if not path.isfile(file_name):
        return None
    result = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n').strip()
            if len(line) == 0:
                continue
            result.append(line)
    return result

# write file
def write_file(file_name, line, write_time=False):
    '''
    file_name: name
    line: content to write
    write_time: write current time before this line
    '''
    with open(file_name,'a') as f:
        if write_time:
            line = get_curr_date() + '\n' + str(line)
        f.write(str(line) + '\n')
    return None

# rewrite a list to file_name 
def rewrite_file(file_name, ls_line):
    '''
    rewrite file in file_name
    '''
    with open(file_name, 'w') as f:
        for line in ls_line:
            f.write(str(line) + '\n')
    return

# parameter voc xml file
def parse_voc_xml(file_name, names_dict):
    '''
    return [ [id1, x1, y1, w1, h1], [id2, x2, y2, w2, h2], ... ]
    '''
    # print(file_name)
    # print(names_dict)
    result = []
    if not os.path.isfile(file_name):
        return None
    doc = parse(file_name)
    root = doc.documentElement
    size = root.getElementsByTagName('size')[0]
    width = int(size.getElementsByTagName('width')[0].childNodes[0].data)
    height = int(size.getElementsByTagName('height')[0].childNodes[0].data)

    objs = root.getElementsByTagName('object')
    for obj in objs:
        name = obj.getElementsByTagName('name')[0].childNodes[0].data
        name_id = names_dict[name]

        bndbox = obj.getElementsByTagName('bndbox')[0]
        xmin = int(float(bndbox.getElementsByTagName('xmin')[0].childNodes[0].data))
        ymin = int(float(bndbox.getElementsByTagName('ymin')[0].childNodes[0].data))
        xmax = int(float(bndbox.getElementsByTagName('xmax')[0].childNodes[0].data))
        ymax = int(float(bndbox.getElementsByTagName('ymax')[0].childNodes[0].data))

        x = (xmax + xmin) / 2.0 / width
        w = (xmax - xmin) / width
        y = (ymax + ymin) / 2.0 / height
        h = (ymax - ymin) / height

        result.append([name_id, x, y, w, h])
    return result

'''
######################## about time ####################
'''
# get current time
def get_curr_date():
    '''
    return : year-month-day-hours-minute-second
    '''
    t = time.gmtime()
    time_str = time.strftime("%Y-%m-%d-%H-%M-%S",t)
    return time_str

'''
######################## about image ####################
'''
# read image
def read_img(file_name):
    '''
    read image as BGR
    return:BGR image
    '''
    if not path.exists(file_name):
        return None
    img = cv2.imread(file_name)
    return img

# draw some box on image
def draw_img(img, boxes, score, label, word_dict, color_table,):
    '''
    img : cv2.img [416, 416, 3]
    boxes:[V, 4], x_min, y_min, x_max, y_max
    score:[V], score of corresponding box 
    label:[V], label of corresponding box
    word_dict: dictionary of  id=>name
    return : a image after draw the boxes
    '''
    w = img.shape[1]
    h = img.shape[0]
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        boxes[i][0] = constrait(boxes[i][0], 0, 1)
        boxes[i][1] = constrait(boxes[i][1], 0, 1)
        boxes[i][2] = constrait(boxes[i][2], 0, 1)
        boxes[i][3] = constrait(boxes[i][3], 0, 1)
        x_min, x_max = int(boxes[i][0] * w), int(boxes[i][2] * w)
        y_min, y_max = int(boxes[i][1] * h), int(boxes[i][3] * h)
        
        curr_label = label[i] if label is not None else 0
        curr_color = color_table[curr_label] if color_table is not None else (0, 125, 255)
        
        # draw box
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), curr_color)
        # draw font
        if word_dict is not None:
            text_name = "{}".format(word_dict[curr_label])
            cv2.putText(img, text_name, (x_min, y_min + 25), font, 1, curr_color)
        if score is not None:
            text_score = "{:2d}%".format(int(score[i] * 100))
            cv2.putText(img, text_score, (x_min, y_min), font, 1, curr_color)
    return img


'''
######################## others ####################
'''
def get_word_dict(name_file):
    '''
    dictionary of id to name
    return:{}
    '''
    word_dict = dict()
    if not os.path.exists(name_file):
        print("Name file:{} doesn't exist".format(name_file))
    else:
        contents = read_file(name_file)
        for i in range(len(contents)):
            word_dict[i] = str(contents[i])
    return word_dict

# name => id 
def word2id(names_file):
    '''
    dictionary of name to id
    return {}
    '''
    id_dict = {}
    contents = read_file(names_file)
    for i in range(len(contents)):
        id_dict[str(contents[i])] = i
    return id_dict

def constrait(x, start, end):
    '''    
    return:x    ,start <= x <= end
    '''
    if x < start:
        return start
    elif x > end:
        return end
    else:
        return x

# get a list of color of corresponding name 
def get_color_table(class_num):
    '''
    return :  list of (r, g, b) color
    '''
    color_table = []
    for i in range(class_num):
        r = random.randint(128, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        color_table.append((b, g, r))
    return color_table


