'''
对indian pines, salinas, pavia university三个数据集进行处理
    不双边滤波
    可以选择数据块的大小
    可以选择训练集比例

'''
from scipy.io import loadmat
import numpy as np
import cv2

# pad function
def zero_pad(X,pad):
    X_pad = np.pad(X, ((pad,pad), (pad,pad), (0,0)), 'constant')
    return X_pad

# normalization
def normalize(X):
    print('import normalize ... ...')
    X = X.astype('float32')
    return (X-np.min(X))/(np.max(X)-np.min(X))

# 双边滤波(不采用)
def bf(x_raw):
    x_bi = np.zeros((x_raw.shape))
    print('import bf ... ...')
    for i in range(x_raw.shape[2]):
        x_bi[:,:,i] = cv2.bilateralFilter(x_raw[:,:,i], 7, 50, 50)
    return x_bi


# 选择不同的spatial size和数据集
def handle_data(dataset,train_scale, spatial_size, b_bf=0):
    if dataset=='indian_pines':
        data_corrected=loadmat('dataset_indian_pines/Indian_pines_corrected.mat')
        data_gt=loadmat('dataset_indian_pines/Indian_pines_gt.mat')
        data_x=data_corrected['indian_pines_corrected']
        data_y=data_gt['indian_pines_gt']
        num_class=16
    elif dataset=='pavia_university':
        data_corrected=loadmat('dataset_pavia_university/PaviaU.mat')
        data_gt=loadmat('dataset_pavia_university/PaviaU_gt.mat')
        data_x=data_corrected['paviaU']
        data_y=data_gt['paviaU_gt']
        num_class=9
    elif dataset=='salinas':
        data_corrected=loadmat('dataset_salinas/Salinas_corrected.mat')
        data_gt=loadmat('dataset_salinas/Salinas_gt.mat')
        data_x=data_corrected['salinas_corrected']
        data_y=data_gt['salinas_gt']
        num_class=16

    pad = (spatial_size - 1) // 2

    #归一化
    data_x=normalize(data_x)

    # 双边滤波
    if b_bf==True:
        data_x = bf(data_x)

    #pad
    data_x_pad = zero_pad(data_x,pad)

    x_data_all = np.zeros((data_x.shape[0]*data_x.shape[1],spatial_size,spatial_size,data_x.shape[2]))
    y_data_all = data_y.reshape(-1,1)

    # print(data_x.shape)
    # print(data_y.shape)

    for i in range (0,data_x.shape[0]):
        for j in range (data_x.shape[1]):
            x_data_all[i*data_x.shape[1]+j,:,:,:]=data_x_pad[i:i+spatial_size,j:j+spatial_size,:]
    # print(x_data_all.shape)
    # print(y_data_all.shape)

    x_train=np.zeros((0,spatial_size,spatial_size,data_x.shape[2]))
    y_train=np.zeros((0,1))
    x_test=np.zeros((0,spatial_size,spatial_size,data_x.shape[2]))
    y_test=np.zeros((0,1))

    for k in range (1,num_class+1):
        np.random.seed(k)
        #提取每类数据
        y_temp=np.array(y_data_all[:,0]==k).astype(int).reshape(-1,1)
        y_temp=np.multiply(y_temp,y_data_all)
        y=y_temp[np.flatnonzero(y_temp),:]
        x=x_data_all[np.flatnonzero(y_temp),:]
        #print(y.shape)
        #print(x[:,1,1,1])

        #打乱数据集
        permutation = np.random.permutation(y.shape[0])
        x = x[permutation, :]
        #print(x[:,1,1,1])

        #构造训练集和测试集
        train_set_number=int(y.shape[0]*train_scale)
        x_train_batch=x[0:train_set_number,:]
        x_test_batch=x[train_set_number:,:]
        y_train_batch=y[0:train_set_number,:]
        y_test_batch=y[train_set_number:,:]

        x_train=np.concatenate((x_train,x_train_batch), axis=0)
        y_train=np.concatenate((y_train,y_train_batch), axis=0)
        x_test=np.concatenate((x_test,x_test_batch), axis=0)
        y_test=np.concatenate((y_test,y_test_batch), axis=0)  

    #打乱训练集
    np.random.seed(1)
    per_train_all = np.random.permutation(y_train.shape[0])
    x_train = x_train[per_train_all, :]
    y_train = y_train[per_train_all, :]

    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)

    return x_train,y_train,x_test,y_test

# 获得原始数据
def get_rawdata(dataset):
    if dataset=='indian_pines':
        data_corrected=loadmat('dataset_indian_pines/Indian_pines_corrected.mat')
        data_gt=loadmat('dataset_indian_pines/Indian_pines_gt.mat')
        data_x=data_corrected['indian_pines_corrected']
        data_y=data_gt['indian_pines_gt']
    elif dataset=='pavia_university':
        data_corrected=loadmat('dataset_pavia_university/PaviaU.mat')
        data_gt=loadmat('dataset_pavia_university/PaviaU_gt.mat')
        data_x=data_corrected['paviaU']
        data_y=data_gt['paviaU_gt']
    elif dataset=='salinas':
        data_corrected=loadmat('dataset_salinas/Salinas_corrected.mat')
        data_gt=loadmat('dataset_salinas/Salinas_gt.mat')
        data_x=data_corrected['salinas_corrected']
        data_y=data_gt['salinas_gt']
    
    return data_x,data_y