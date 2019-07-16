'''
resnet
'''

import keras
from keras.layers import Conv3D, Dense, BatchNormalization, Activation,Dropout,MaxPooling3D
from keras.layers import AveragePooling3D, Input, Flatten,add
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
    x=conv_block(name=name+'_two1',inputs=x,kout=kout,kernel_size=kernel_size,strides=strides,padding=padding)
    x=conv_block(name=name+'_two2',inputs=x,kout=kout,kernel_size=kernel_size,strides=(1,1,1),padding='same',relu=False)
    return x

# shortcut
def shortcut(name,inputs,kin,kout,kernel_size,strides,padding):
    x=inputs
    if kin!=kout:
        x=conv_block(name=name+'_line',inputs=x,kout=kout,kernel_size=kernel_size,strides=strides,padding=padding,relu=False)
    return x


# resnet block
def res_block(name,inputs,kin,kout,kernel_size,strides,padding):
    x=inputs
    one=shortcut(name=name+'_l',inputs=x,kin=kin,kout=kout,kernel_size=kernel_size,strides=strides,padding=padding)
    two=two_conv(name=name+'_r',inputs=x,kout=kout,kernel_size=kernel_size,strides=strides,padding=padding)
    x=add([one,two])
    x=Activation('relu',name=name+'_relu')(x)
    return x

# resnet group
def res_group(name,inputs,count,kin,kout,kernel_size,strides,padding):
    x=inputs
    x=res_block(name=name+'_b1',inputs=x,kin=kin,kout=kout,kernel_size=kernel_size,strides=strides,padding=padding)
    kin=kout
    for idx in range(count-1):
        x=res_block(name=name+'_b%d'%(idx+2),inputs=x,kin=kin,kout=kout,kernel_size=kernel_size,strides=(1,1,1),padding='same')
    return x

# resnet net
def res_net(input_shape,num_classes,net_depth,primary_partition,secondary_partition):
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
    x=conv_block('g0', x_inputs, kout=channel, kernel_size=(1,1,11), strides=(1,1,2), padding='valid')
    
    # resnet三个group
    x=res_group('g1',x,blocks_num[0],kin=channel*1,kout=channel*2,kernel_size=(3,3,7),strides=(1,1,2),padding='valid')
    x=res_group('g2',x,blocks_num[1],kin=channel*2,kout=channel*4,kernel_size=(3,3,5),strides=(1,1,2),padding='valid')
    x=res_group('g3',x,blocks_num[2],kin=channel*4,kout=channel*8,kernel_size=(2,2,5),strides=(1,1,1),padding='valid')

    x=AveragePooling3D(name='avg_pool',pool_size=(2,2,2))(x)
    x=Flatten()(x)
    x=Dense(num_classes, activation='softmax', name='fc1')(x)
    
    model = Model(inputs = x_inputs, outputs = x, name = 'resnet')
    model.summary()
    return model
