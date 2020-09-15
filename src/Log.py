# coding:utf-8
# log
import config
from os import path
import os
from utils import tools

log_dir = config.log_dir
loss_name = config.loss_name

# add a log message
def add_log(content):
    if not path.isdir(log_dir):
        os.mkdir(log_dir)
        add_log("message:create folder '{}'".format(log_dir))
    log_file = path.join(log_dir, config.log_name)
    print(content)
    tools.write_file(log_file, content, True)
    return

# add a loss value
def add_loss(value):
    if not path.isdir(log_dir):
        os.mkdir(log_dir)
        add_log("create folder '{}'".format(log_dir))
    loss_file = path.join(log_dir, loss_name)
    tools.write_file(loss_file, value, False)
    return