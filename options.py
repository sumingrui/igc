'''
train args

'''

import argparse
import datetime
import os

def get_time():
    str=(datetime.datetime.now()+datetime.timedelta(hours=8)).strftime('%Y-%m-%d_%H:%M:%S')
    return str


# 创建子文件夹
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def get_args(argv,parse=True):
    parser=argparse.ArgumentParser(description='train an deep cnn')

    parser.add_argument('--network',type=str,help='network name')
    
    # network structure
    parser.add_argument('--depth',type=int,help='depth of the corresponding network')
    parser.add_argument('--firstgroup',type=int,help='primary partition number')
    parser.add_argument('--secondgroup',type=int,help='secondary partition number')
    #parser.add_argument('--data-dir', type=str, help='the input data directory')

    parser.add_argument('--dataset',type=str,default='indian_pines',choices=['indian_pines','pavia_university','salinas'])

    # data pretreat
    parser.add_argument('--spatial-size',type=int,default=7, help='data cube spatial size')
    parser.add_argument('--bf',type=int, default=0, help='is data bf')
    # training strategy
    parser.add_argument('--train-scale', type=float, default=0.3, help='the training size scale')
    parser.add_argument('--batch-size', type=int, default=32, help='the training batch size')
    parser.add_argument('--num-epochs', type=int, default=200, help='the number of training epochs')
    
    # learning rate
    parser.add_argument('--lr', type=float, default=0.001, help='the initial learning rate')
    parser.add_argument('--lr-factor', type=float, default=0.333, help='reduce the lr by a factor')
    parser.add_argument('--lr-callback', type=int, default=1, help='0: by epoch  1: by val_acc')
    parser.add_argument('--lr-byepoch', type=int, help='reduce the lr by epoch')

    # optimizer
    parser.add_argument('--optimizer', type=str, default='adam',help='optimizer:adam....')

    # others
    parser.add_argument('--dropout', type=float, default=0.1,help='dropout scale')
    # parser.add_argument('--filters', type=str, help='eg. --filters=16,32,64,128')
    parser.add_argument('--pooling', type=str, default='max',help='maxpooling or average pooling')

    args=parser.parse_args(argv)
    save_dir=resolve_args(args,parse)
    return args, save_dir

# 解析参数
def resolve_args(args,parse=True):
    save_dir=args.network+'_'+args.dataset+'_'+str(args.depth)+'_'+str(args.firstgroup)+'_'+str(args.secondgroup)+'_'+'/'+get_time()+'/'
    mkdir(save_dir)
    if parse==True:
        # 打印参数
        f = open(save_dir+'network_config.txt','a')
        f.writelines('---------------------------- Start Report ----------------------------')
        f.writelines('network: %s\n'%args.network)
        f.writelines('depth: %d\n'%args.depth)
        f.writelines('firstgroup: %d\n'%args.firstgroup)
        f.writelines('secondgroup: %d\n'%args.secondgroup)
        f.writelines('dataset: %s\n\n'%args.dataset)

        f.writelines('train_scale: %f\n'%args.train_scale)
        f.writelines('spatial_size: %d\n'%args.spatial_size)
        f.writelines('training_batch_size: %d\n'%args.batch_size)
        f.writelines('training_epoch: %d\n'%args.num_epochs)
        f.writelines('is data bf: %s\n\n'%('True' if(args.bf==1) else 'False'))
        
        f.writelines('lr: %f\n'%args.lr)
        f.writelines('lr_factor: %f\n'%args.lr_factor) 
        f.writelines('lr_callback:{}\n'.format('LearningRateScheduler' if(args.lr_callback==0) else 'ReduceLROnPlateau'))
        f.writelines('lr_reduce_by_epoch: %d\n\n'%args.lr_byepoch)
        
        f.writelines('optimizer: %s\n'%args.optimizer)
        f.writelines('dropout: %f\n'%args.dropout)
        # f.writelines('filters: %s\n'%args.filters)
        f.writelines('pooling: %s\n'%args.pooling)
        f.close()

    return save_dir