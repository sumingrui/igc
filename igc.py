'''
plain network with IGC
'''

import keras
from keras.layers import Conv3D, Dense, BatchNormalization, Activation,Dropout,MaxPooling3D
from keras.layers import AveragePooling3D, Input, Flatten, add, Reshape, Permute,Lambda,concatenate
from keras.regularizers import l2
from keras.models import Model
from keras.utils import plot_model
import keras.backend as K
import traceback


# 获得模型名称
def get_model_name():
    return traceback.extract_stack()[-2][2]


def get_igc_wrong():
    input_shape = (10,10,64)
    x0=Input(input_shape)
    x=Reshape((10,10,2,32),input_shape=input_shape)(x0)
    x = Permute((1,2,4,3))(x)   # 1,2维度的顺序没有改变
    x1 = Lambda(lambda z: K.expand_dims(z[:, :, :, :, 0], axis=-1))(x)
    x2 = Lambda(lambda z: K.expand_dims(z[:, :, :, :, 1], axis=-1))(x)
    x1 = Conv3D(2, kernel_size=(3, 3, 2), strides=(1, 1, 2), padding='same')(x1)
    x2 = Conv3D(2, kernel_size=(3, 3, 2), strides=(1, 1, 2), padding='same')(x2)
    x1 = Reshape((10,10,32), input_shape=(10, 10, 16, 2))(x1)
    x2 = Reshape((10,10,32), input_shape=(10, 10, 16, 2))(x2)
    x = concatenate([x1, x2],axis = -1)
    x = Reshape((10,10,2,32), input_shape = (10,10,64))(x)
    x = Permute((1,2,4,3))(x)
    x = Conv3D(2, kernel_size=(3, 3, 1), strides=(1, 1, 1), padding='same')(x)
    x = Reshape((10,10,64), input_shape = (10, 10, 32, 2))(x)
    model = Model(inputs = x0, outputs = x)
    model.summary()
    plot_model(model, to_file='igc.png', show_shapes=True)
    return model


# 普通卷积
def conv_block(name,inputs,kout,kernel_size,strides,padding,activation='relu',batch_normalization=True):
    # Conv-BN-Relu
    x=inputs
    x=Conv3D(name=name+'_conv',filters=kout,kernel_size=kernel_size,strides=strides,padding=padding,
                kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.001))(x)
    x=BatchNormalization(name=name+'_bn',axis=4)(x)
    x=Activation(activation,name=name+'_relu')(x)
    return x


def divide1(data,start,end):
    return data[:,:,:,:,start:end]

def divide2(data,slice):
    return data[:,:,:,:,:,slice]

def exp_dims(data):
    return K.expand_dims(data,axis=-1)


'''
igc块
'''
def igc_block(name,inputs,kin,kout,primary_partition,secondary_partition,activation='relu',batch_normalization=True):
    #x_input=Input(inputs,name='inputsss')  # 7,7,200,16
    x=inputs
    x1=[]
    x2=[]

    # 1th step: 生成需要的宽度 primary_partition*secondary_partition 之前的conv_block已经做好
    # 2th step: 分成primary_partition组
    for i in range(primary_partition):
        c=int(int(x.shape[4])/primary_partition)
        x1.append(Lambda(lambda z:z[:,:,:,:,i*c:(i+1)*c],name=name+'_L%d'%(i+1))(x))
        # 3th step: 第一部分做卷积
        x1[i]=Conv3D(name=name+'_L%dconv'%(i+1),filters=secondary_partition,kernel_size=(3,3,3),strides=(1,1,1) if kin==kout else (2,2,2),
                        padding='same',kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.001))(x1[i])
        x1[i]=Lambda(lambda z:K.expand_dims(z,axis=-1))(x1[i])

    # 5th+6th step: 组合+permute
    x = concatenate(x1,axis=-1,name=name+'_concat1')
    x = Permute((1,2,3,5,4),name=name+'_permute1')(x)
    
    # 7th step: 分组conv
    for j in range(secondary_partition):
        x2.append(Lambda(lambda z:z[:,:,:,:,:,j],name=name+'_M%d'%(j+1))(x))
        x2[j]=Conv3D(name=name+'_M%dconv'%(j+1),filters=primary_partition,kernel_size=(1,1,1),strides=(1,1,1),
                        padding='same',kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.001))(x2[j])
        x2[j]=Lambda(lambda z:K.expand_dims(z,axis=-1))(x2[j])

    # 8th step: 组合+恢复permute
    x = concatenate(x2,axis=-1,name=name+'_concat2')

    a=int(x.shape[1])
    b=int(x.shape[2])
    c=int(x.shape[3])
    x = Reshape((a,b,c,primary_partition*secondary_partition),name=name+'_permute2')(x)
    x=BatchNormalization(axis=-1,name=name+'_bn')(x)
    x=Activation(activation,name=name+'_relu')(x)
    return x


'''
igc_1块
采用和自定义卷积核
一般卷积操作在第一组内完成
第二组卷积核固定
'''
def igc1_block(name,inputs,primary_partition,secondary_partition,kernel_size,strides,padding,activation='relu',batch_normalization=True):
    #x_input=Input(inputs,name='inputsss')  # 7,7,200,16
    x=inputs
    x1=[]
    x2=[]

    # 1th step: 生成需要的宽度 primary_partition*secondary_partition 之前的conv_block已经做好
    # 2th step: 分成primary_partition组
    for i in range(primary_partition):
        c=int(int(x.shape[4])/primary_partition)
        x1.append(Lambda(divide1,arguments={'start':i*c,'end':(i+1)*c},name=name+'_L%d'%(i+1))(x))
        # 3th step: 第一部分做卷积
        x1[i]=Conv3D(name=name+'_L%dconv'%(i+1),filters=secondary_partition,kernel_size=kernel_size,strides=strides,
                        padding=padding,kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.001))(x1[i])
        x1[i]=Lambda(exp_dims)(x1[i])

    # 5th+6th step: 组合+permute
    if len(x1)!=1: 
        x = concatenate(x1,axis=-1,name=name+'_concat1')
    x = Permute((1,2,3,5,4),name=name+'_permute1')(x)
    
    # 7th step: 分组conv
    for j in range(secondary_partition):
        x2.append(Lambda(divide2,arguments={'slice':j},name=name+'_M%d'%(j+1))(x))
        x2[j]=Conv3D(name=name+'_M%dconv'%(j+1),filters=primary_partition,kernel_size=(1,1,1),strides=(1,1,1),
                        padding='same',kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.001))(x2[j])
        x2[j]=Lambda(exp_dims)(x2[j])

    # 8th step: 组合+恢复permute
    if len(x2)!=1:
        x = concatenate(x2,axis=-1,name=name+'_concat2')

    a=int(x.shape[1])
    b=int(x.shape[2])
    c=int(x.shape[3])
    x = Reshape((a,b,c,primary_partition*secondary_partition),name=name+'_permute2')(x)
    x=BatchNormalization(axis=-1,name=name+'_bn')(x)
    x=Activation(activation,name=name+'_relu')(x)
    return x


# igc组
def igc_group(name,data,num_block,primary_partition, secondary_partition,kernel_size,strides,padding):
    data=igc1_block(name=name+'_b1',inputs=data,primary_partition=primary_partition,secondary_partition=secondary_partition,
                        kernel_size=kernel_size,strides=strides,padding=padding)
    for idx in range(num_block-1):
        data=igc1_block(name=name+'_b%d'%(idx+2),inputs=data,primary_partition=primary_partition,secondary_partition=secondary_partition,
                            kernel_size=kernel_size,strides=(1,1,1),padding='same')
    return data


# igc1组    用自定义方法降维，再接恒等
def igc1_group(name,data,num_block,primary_partition, secondary_partition,kernel_size,strides,padding):
    data=igc1_block(name=name+'_b1',inputs=data,primary_partition=primary_partition,secondary_partition=secondary_partition,
                        kernel_size=kernel_size,strides=strides,padding=padding)
    for idx in range(num_block-1):
        data=igc1_block(name=name+'_b%d'%(idx+2),inputs=data,primary_partition=primary_partition,secondary_partition=secondary_partition,
                            kernel_size=kernel_size,strides=(1,1,1),padding='same')
    return data


# 两种pooling
def pool(x,pooling,pool_size):
    if pooling=='max':
        x=MaxPooling3D(pool_size=pool_size)(x)
    elif pooling=='avg':
        x=AveragePooling3D(pool_size=pool_size)(x)
    return x


# igc网络:不用pooling 用stride来做下采样
def igc_net(input_shape,num_classes, net_depth,primary_partition,secondary_partition):
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
    
    # igc 3 stage
    x=igc1_group('g1',x,blocks_num[0],primary_partition=primary_partition,secondary_partition=secondary_partition*2,
                    kernel_size=(3,3,7),strides=(1,1,2),padding='valid')
    x=igc1_group('g2',x,blocks_num[1],primary_partition=primary_partition,secondary_partition=secondary_partition*4,
                    kernel_size=(3,3,5),strides=(1,1,2),padding='valid')
    x=igc1_group('g3',x,blocks_num[2],primary_partition=primary_partition,secondary_partition=secondary_partition*8,
                    kernel_size=(2,2,5),strides=(1,1,1),padding='valid')
      
    # 对应不同块有不同的pooling大小
    x=AveragePooling3D(name='avg_pool',pool_size=(2,2,2))(x)
    x=Flatten()(x)
    x=Dense(num_classes, activation='softmax', name='fc1')(x)
    
    model = Model(inputs = x_inputs, outputs = x, name = 'igc')
    model.summary()
    return model


# igc1网络
def igc1_net(input_shape,num_classes, net_depth,primary_partition,secondary_partition,dropout,pooling):
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
    x=igc1_group('g1',x,blocks_num[0],primary_partition=primary_partition,secondary_partition=secondary_partition*2,
                    kernel_size=(3,3,7),strides=(1,1,1),padding='valid')
    x=pool(x,pooling,pool_size=(1,1,2))


    # stage 2
    x=Dropout(dropout)(x)
    x=igc1_group('g2',x,blocks_num[1],primary_partition=primary_partition,secondary_partition=secondary_partition*4,
                    kernel_size=(3,3,5),strides=(1,1,1),padding='valid')
    x=pool(x,pooling,pool_size=(1,1,2))

    # stage 3
    x=Dropout(dropout)(x)
    x=igc1_group('g3',x,blocks_num[2],primary_partition=primary_partition,secondary_partition=secondary_partition*8,
                    kernel_size=(2,2,5),strides=(1,1,1),padding='valid')
    
    # 对应不同块有不同的pooling大小
    x=AveragePooling3D(name='avg_pool',pool_size=2)(x)
    x=Flatten()(x)
    x=Dense(num_classes, activation='softmax', name='fc1')(x)
    
    model = Model(inputs = x_inputs, outputs = x, name = 'igc1')
    model.summary()

    return model

