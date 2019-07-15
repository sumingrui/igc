'''
plain network
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
def conv_block(name,inputs,kout,kernel_size,strides,padding,activation='relu',batch_normalization=True):
    # Conv-BN-Relu
    x=inputs
    x=Conv3D(name=name+'_conv',filters=kout,kernel_size=kernel_size,strides=strides,padding=padding,
                kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.001))(x)
    x=BatchNormalization(name=name+'_bn',axis=4)(x)
    x=Activation(activation,name=name+'_relu')(x)
    return x


# plain组
def plain_group(name,data,num_block,kout,kernel_size,strides,padding):
    data=conv_block(name=name+'_b1',inputs=data,kout=kout,kernel_size=kernel_size,strides=strides,padding=padding)
    for idx in range(num_block-1):
        data=conv_block(name=name+'_b%d'%(idx+2),inputs=data,kout=kout,kernel_size=kernel_size,strides=strides,padding='same')
    return data


# plain网络
def plain_net(input_shape,num_classes,net_depth,primary_partition,secondary_partition,dropout,pooling):
    #三个stage
    block3_num=int((net_depth-2)/3)
    block2_num=int((net_depth-2)/3)
    block1_num=int((net_depth-2)/3)
    blocks_num=(block1_num,block2_num,block3_num)
    if net_depth!=((block1_num+block2_num+block3_num)*1+2) or block3_num<=0:
        print('invalid depth number: %d'%net_depth,', blocks numbers: ',blocks_num)
        return
    
    # 第一个conv之后需要的channel数量
    channel=secondary_partition*primary_partition
    x_inputs = Input(input_shape)
    # 第一个普通conv
    x=conv_block('g0', x_inputs, kout=channel, kernel_size=(1,1,11), strides=(1,1,1), padding='valid')
    x=pool(x,pooling,pool_size=(1,1,2))
    
    # stage 1
    x=Dropout(dropout)(x)
    x=plain_group('g1',x,blocks_num[0],kout=channel*2,kernel_size=(3,3,7),strides=(1,1,1),padding='valid')
    x=pool(x,pooling,pool_size=(1,1,2))

    # stage 2
    x=Dropout(dropout)(x)
    x=plain_group('g2',x,blocks_num[1],kout=channel*4,kernel_size=(3,3,5),strides=(1,1,1),padding='valid')
    x=pool(x,pooling,pool_size=(1,1,2))

    # stage 3
    x=Dropout(dropout)(x)
    x=plain_group('g3',x,blocks_num[2],kout=channel*8,kernel_size=(2,2,5),strides=(1,1,1),padding='valid')
    
    # 对应不同块有不同的pooling大小
    x=AveragePooling3D(name='avg_pool',pool_size=2)(x)
    x=Flatten()(x)
    x=Dense(num_classes, activation='softmax', name='fc1')(x)
    
    model = Model(inputs = x_inputs, outputs = x, name = 'plain')
    model.summary()

    return model