# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 19:25:39 2020

@author: MMM
"""

import os
from glob import glob
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2

def load_data(path, split=0.2):
     images=sorted(glob(os.path.join(path,"images/*")))
     labels=sorted(glob(os.path.join(path,"labels/*")))
     
     total_size=len(images)
     validation_size=int(split*total_size)
     test_size=int(split*total_size)
     #print(" total_size: ", total_size,"validation_size: ",validation_size,"test_size: ",test_size)
     
     train_x, valid_x=train_test_split(images,test_size=validation_size,random_state=42)
     train_y, valid_y=train_test_split(labels,test_size=validation_size,random_state=42)
     
     train_x, test_x=train_test_split(train_x,test_size=test_size,random_state=42)
     train_y, test_y=train_test_split(train_y,test_size=test_size,random_state=42)
     
     
     return (train_x,train_y),(valid_x,valid_y),(test_x,test_y)

def read_image(path):
    path=path.decode()
    x=cv2.imread(path,cv2.IMREAD_COLOR)
    x=cv2.resize(x,(256,256))
    x=x/255.0
    return x


def read_label(path):
    path=path.decode()
    x=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    x=cv2.resize(x,(256,256))
    x=x/255.0
    x=np.expand_dims(x,axis=-1)
    return x

def tf_parse(x,y):
    def _parse(x,y):
         x=read_image(x)
         y=read_label(y)
         
         return x,y
    x,y= tf.numpy_function(_parse,[x, y], [tf.float64,tf.float64])
    x.set_shape([256,256,3])
    y.set_shape([256,256,1])
    return x,y
    
def tf_dataset(x,y,batch=8):
    dataset=tf.data.Dataset.from_tensor_slices((x,y))
    dataset=dataset.map(tf_parse)
    dataset=dataset.batch(batch)
    dataset=dataset.repeat()
    return dataset
    
if __name__=="__main__":
    
    path="CVC-ClinicDB"
    (train_x,train_y),(valid_x,valid_y),(test_x,test_y)=load_data(path)
    print(" train_x: ", len(train_x)," train_y: ", len(train_y))
    print(" valid_x: ", len(valid_x)," valid_y: ", len(valid_y))
    print(" test_x: ", len(test_x)," test_y: ", len(test_y))
    
    ds=tf_dataset(test_x,test_y)
    for x,y in ds:
        print(x.shape,y.shape)
        break
