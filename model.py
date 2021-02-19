# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 14:30:31 2020

@author: MMM
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation,Input, MaxPooling2D, Concatenate, LeakyReLU, Dropout, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow import keras

def conv_block(x,num_filters):
    
    x=Conv2D(num_filters,(3,3),padding='same')(x)
    x=BatchNormalization()(x)
    x=LeakyReLU(0.02)(x)
    x= Dropout(0.1)(x)
    x=Conv2D(num_filters,(3,3),padding='same')(x)
    x=BatchNormalization()(x)
    x=LeakyReLU(0.02)(x)
    x= Dropout(0.1)(x)
    return x
    

def build_model():
    size=256
    num_filters=[16,32,64,128]
    
    inputs=Input(shape=(size,size,3))
    skip_x=[]
    skip_pp=[]
    x=inputs
    
    #Encoder
    for f in num_filters:
        x=conv_block(x,f)
        skip_x.append(x)
        x=MaxPooling2D(2,2)(x)
        

    #bottleneck
    x04=conv_block(x,num_filters[-1]*2)
    num_filters.reverse()
   
    
    #skip connection
    x01=UpSampling2D((2,2))(skip_x[1])
    x01=Concatenate()([skip_x[0],x01])
    x01=conv_block(x01,16)
   
    x11=UpSampling2D((2,2))(skip_x[2])
    x11=Concatenate()([skip_x[1],x11])
    x11=conv_block(x11,32)
    
    x21=UpSampling2D((2,2))(skip_x[3])
    x21=Concatenate()([skip_x[2],x21])
    x21=conv_block(x21,32)
    ##########################################################################
    x02=UpSampling2D((2,2))(x11)
    x02=Concatenate()([skip_x[0],x01,x02])
    x02=conv_block(x02,16)
    
    x12=UpSampling2D((2,2))(x21)
    x12=Concatenate()([skip_x[1],x12,x11])
    x12=conv_block(x12,16)
    ##########################################################################
    x03=UpSampling2D((2,2))(x12)
    x03=Concatenate()([skip_x[0],x01,x02,x03])
    x03=conv_block(x03,16)
    
    
    skip_pp.append(Concatenate()([skip_x[0],x01,x02,x03]))
    skip_pp.append(Concatenate()([skip_x[1],x11,x12]))
    skip_pp.append(Concatenate()([skip_x[2],x21]))
    skip_pp.append(skip_x[3])
    
    skip_pp.reverse()
    
    #Decoder
    """x31= UpSampling2D((2,2))(x04)
    x31=Concatenate()([skip_pp[0],x31])
    x31=conv_block(x31,num_filters[0])
    
    x22= UpSampling2D((2,2))(x31)
    x22=Concatenate()([skip_pp[1],x22])
    x22=conv_block(x22,num_filters[1])
    
    x13= UpSampling2D((2,2))(x22)
    x13=Concatenate()([skip_pp[2],x13])
    x13=conv_block(x13,num_filters[2])
    
    x03= UpSampling2D((2,2))(x13)
    x03=Concatenate()([skip_pp[3],x03])
    x03=conv_block(x03,num_filters[3])"""
    
    x=x04
    for i,f in enumerate(num_filters):
        x=UpSampling2D((2,2))(x)
        xs=skip_pp[i]
        x=Concatenate()([x,xs])
        x=conv_block(x,f)

    #output
    x=Conv2D(1,(1,1),padding='same')(x)
    x=Activation('sigmoid')(x)
    return Model(inputs,x)

if __name__=="__main__":
    
    model=build_model()
    model.summary()
    dot_img_file = 'model_1.png'
    tf.keras.utils.plot_model(model, to_file=dot_img_file)
    
    
    