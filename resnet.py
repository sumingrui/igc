'''
resnet
'''

import keras
from keras.layers import Conv3D, Dense, BatchNormalization, Activation,Dropout,MaxPooling3D
from keras.layers import AveragePooling3D, Input, Flatten
from keras.regularizers import l2
from keras.models import Model

# 两种pooling
def pool(x,pooling,pool_size):
    if pooling=='max':
        x=MaxPooling3D(pool_size=pool_size)(x)
    elif pooling=='avg':
        x=AveragePooling3D(pool_size=pool_size)(x)
    return x


# 普通卷积
def conv_block(name,inputs,kout,kernel_size,strides,padding,relu=True,batch_normalization=True):
    # Conv-BN-Relu
    x=inputs
    x=Conv3D(name=name+'_conv',filters=kout,kernel_size=kernel_size,strides=strides,padding=padding,
                kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.001))(x)
    x=BatchNormalization(name=name+'_bn',axis=4)(x)
    if relu:
        x=Activation('relu',name=name+'_relu')(x)
    return x


# 两层卷积
def two_conv(name,inputs,kout,kernel_size,strides,padding):
    x=inputs
    x=conv_block(name=name+'_res1',inputs=x,kout=kout,kernel_size=kernel_size,strides=strides)

