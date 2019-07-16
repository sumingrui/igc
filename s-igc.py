#!/usr/bin/env python
# coding: utf-8

import keras
from keras import backend as K
from keras.utils import np_utils
from keras.callbacks import LearningRateScheduler
from keras.utils import plot_model
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model

import tensorflow as tf
import numpy as np
from data_pretreat import handle_data
from igc import igc1_net,igc_net
from plain import plain1_net,plain_net
from resnet import res_net
from resnet_igc import igc_res_net
import options
from load_model import data_process,draw_pic

import traceback
from functools import partial
import datetime
import sys


# 设置学习率
def lr_schedule(epoch,lr_init,lr_by_epoch,lr_scale):
    # lr = 1e-3
    # base = 1/3
    # if epoch >0:
    #     t = int(epoch / 30)
    #     lr = lr * base**t
    # print('Learning rate: ', lr)

    # return lr

    lr = lr_init
    base = lr_scale
    if epoch >0:
        t = int(epoch / lr_by_epoch)
        lr = lr * base**t
    print('Learning rate: ', lr)

    return lr


# 计算flops,params
def get_flops():
    run_meta = tf.RunMetadata()

    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.profiler.profile(graph=K.get_session().graph,run_meta=run_meta, cmd='op', options=opts)

    opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()    
    params = tf.profiler.profile(graph=K.get_session().graph,run_meta=run_meta, cmd='op', options=opts)
    
    return flops.total_float_ops,params.total_parameters



def run_model(data,model,args,save_dir):
    # ReduceLROnPlateau和LearningRateScheduler选一个
    if args.lr_callback == 0:
        lr_s = partial(lr_schedule,lr_init=args.lr, lr_by_epoch=args.lr_byepoch, lr_scale=args.lr_factor)
        reduce_lr = LearningRateScheduler(lr_s)

    elif args.lr_callback == 1:
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=args.lr_factor, patience=6, verbose=1, mode='auto')

    tensorboard = TensorBoard(log_dir=save_dir+'logs/')

    checkpoint = ModelCheckpoint(filepath=save_dir+'best_acc.h5', monitor='val_acc', verbose=1,
                                    save_best_only='True', mode='max', period=1)

    callback_lists = [tensorboard, checkpoint, reduce_lr]
    # callback_lists = [lr_scheduler]

    if args.optimizer=='adam':
        print('Use Adam optimizer...')
        model.compile(loss=keras.losses.categorical_crossentropy,
                        #optimizer=keras.optimizers.Adadelta(),
                        optimizer=keras.optimizers.Adam(lr=args.lr),
                        metrics=['accuracy'])

    # 获得训练时间以及每一个epoch时间
    begin = datetime.datetime.now()

    x_train=data[0]
    x_test=data[1]
    y_train=data[2]
    y_test=data[3]

    flops,params=get_flops()
    f = open(save_dir+'network_config.txt','a')
    f.writelines('flops: %d\n'%flops)
    f.writelines('params: %d\n'%params)

    model.fit(x_train, y_train,
                batch_size=args.batch_size,
                epochs=args.num_epochs,
                callbacks=callback_lists,
                validation_data=(x_test, y_test),
                verbose=1)
    
    end = datetime.datetime.now()
    train_time = (end-begin).total_seconds()
    epoch_time = train_time/args.num_epochs
    f.writelines('whole_training_time: %f\n'%train_time)
    f.writelines('each_epoch_time: %f\n'%epoch_time)
    
        
    print("Saving model to disk \n")
    path=save_dir+'final_model.h5'
    model.save(path)

    score = model.evaluate(x_test, y_test, verbose=0)
    f.writelines('Test loss: %f\n'%score[0])
    f.writelines('Test accuracy: %f\n'%score[1])
    f.close()


def analysis_consequence(data,y_eval,save_dir,dataset,spatial_size,bf):
    K.set_learning_phase(0)
    best_model=load_model(save_dir+'best_acc.h5')

    x_test=data[1]
    #y_test=data[3]

    data_process(best_model,save_dir,x_test,y_eval)
    draw_pic(best_model,dataset,save_dir,spatial_size,bf)



def main(argv):
    K.set_image_data_format('channels_last')
    args,save_dir=options.get_args(argv)

    if args.dataset=='indian_pines' or 'salinas':
        num_classes=16
    elif args.dataset=='pavia_university':
        num_classes=9

    # 获得数据
    x_train,y_train,x_test,y_test=handle_data(args.dataset,args.train_scale,args.spatial_size,args.bf)
    
    # expand_dims
    x_train=np.expand_dims(x_train,axis=4)
    x_test=np.expand_dims(x_test,axis=4)

    # y_eval给后处理
    y_eval=y_test

    #one-hot
    y_train=np_utils.to_categorical(y_train)[:,1:num_classes+1]
    y_test =np_utils.to_categorical(y_test)[:,1:num_classes+1]
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    if args.network=='igc1':
        s_model = igc1_net(x_train[0].shape,num_classes,args.depth,args.firstgroup,args.secondgroup,args.dropout,args.pooling)
    elif args.network=='igc':
        s_model = igc_net(x_train[0].shape,num_classes,args.depth,args.firstgroup,args.secondgroup)
    elif args.network=='plain1':
        s_model = plain1_net(x_train[0].shape,num_classes,args.depth,args.firstgroup,args.secondgroup,args.dropout,args.pooling)
    elif args.network=='plain':
        s_model = plain_net(x_train[0].shape,num_classes,args.depth,args.firstgroup,args.secondgroup)
    elif args.network=='resnet':
        s_model =res_net(x_train[0].shape,num_classes,args.depth,args.firstgroup,args.secondgroup)
    elif args.network=='resnet_igc':
        s_model = igc_res_net(x_train[0].shape,num_classes,args.depth,args.firstgroup,args.secondgroup)
    
    #data=(x_train,x_test,y_train,y_test)
    plot_model(s_model, to_file=save_dir+'model.png', show_shapes=True)
    #run_model(data,s_model, args, save_dir)
    # 后处理
    #analysis_consequence(data,y_eval,save_dir,args.dataset,args.spatial_size,args.bf)

if __name__ == '__main__':
    main(sys.argv[1:])