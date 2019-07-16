'''
resnet igc net
'''

import keras
from keras.layers import Conv3D, Dense, BatchNormalization, Activation,Dropout,MaxPooling3D
from keras.layers import AveragePooling3D, Input, Flatten,add, Reshape, Permute,Lambda,concatenate
from keras.regularizers import l2
from keras.models import Model
import keras.backend as K


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


def divide1(data,start,end):
    return data[:,:,:,:,start:end]

def divide2(data,slice):
    return data[:,:,:,:,:,slice]

def exp_dims(data):
    return K.expand_dims(data,axis=-1)


def igc_block(name,inputs,primary_partition,secondary_partition,kernel_size,strides,padding,relu=True,activation='relu',batch_normalization=True):
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
    if relu:
        x=Activation(activation,name=name+'_relu')(x)
    return x


# 两层卷积
def two_conv(name,inputs,kin,kout,kernel_size,strides,padding,primary_partition, secondary_partition):
    x=inputs
    x=igc_block(name+'_two1',inputs=x,primary_partition=primary_partition,secondary_partition=int(kin/primary_partition),
                        kernel_size=kernel_size,strides=strides,padding=padding) 
    x=igc_block(name+'_two2',inputs=x,primary_partition=primary_partition,secondary_partition=int(kout/primary_partition),
                        kernel_size=kernel_size,strides=(1,1,1),padding='same',relu=False)
    return x


# shortcut
def shortcut(name,inputs,kin,kout,kernel_size,strides,padding):
    x=inputs
    if kin!=kout:
        x=conv_block(name=name+'_line',inputs=x,kout=kout,kernel_size=kernel_size,strides=strides,padding=padding,relu=False)
    return x


# resnet block
def res_block(name,inputs,kin,kout,primary_partition,secondary_partition,kernel_size,strides,padding):
    x=inputs
    one=shortcut(name=name+'_l',inputs=x,kin=kin,kout=kout,kernel_size=kernel_size,strides=strides,padding=padding)
    two=two_conv(name=name+'_r',inputs=x,kin=kin,kout=kout,kernel_size=kernel_size,strides=strides,padding=padding,primary_partition=primary_partition,secondary_partition=secondary_partition)
    x=add([one,two])
    x=Activation('relu',name=name+'_relu')(x)
    return x

# resnet group
def res_group(name,inputs,count,kin,kout,primary_partition,secondary_partition,kernel_size,strides,padding):
    x=inputs
    x=res_block(name=name+'_b1',inputs=x,kin=kin,kout=kout,primary_partition=primary_partition,secondary_partition=secondary_partition,kernel_size=kernel_size,strides=strides,padding=padding)
    kin=kout
    for idx in range(count-1):
        x=res_block(name=name+'_b%d'%(idx+2),inputs=x,kin=kin,kout=kout,primary_partition=primary_partition,secondary_partition=secondary_partition,kernel_size=kernel_size,strides=(1,1,1),padding='same')
    return x

# resnet net
def igc_res_net(input_shape,num_classes,net_depth,primary_partition,secondary_partition):
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
    x=res_group('g1',x,blocks_num[0],kin=channel*1,kout=channel*2,primary_partition=primary_partition, secondary_partition=secondary_partition,kernel_size=(3,3,7),strides=(1,1,2),padding='valid')
    x=res_group('g2',x,blocks_num[1],kin=channel*2,kout=channel*4,primary_partition=primary_partition, secondary_partition=secondary_partition,kernel_size=(3,3,5),strides=(1,1,2),padding='valid')
    x=res_group('g3',x,blocks_num[2],kin=channel*4,kout=channel*8,primary_partition=primary_partition, secondary_partition=secondary_partition,kernel_size=(2,2,5),strides=(1,1,1),padding='valid')

    x=AveragePooling3D(name='avg_pool',pool_size=(2,2,2))(x)
    x=Flatten()(x)
    x=Dense(num_classes, activation='softmax', name='fc1')(x)
    
    model = Model(inputs = x_inputs, outputs = x, name = 'resnet')
    model.summary()
    return model
