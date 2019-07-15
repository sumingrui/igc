#测试keras reshape函数输出
import keras
from keras.layers import Reshape,Input,Permute,Lambda,concatenate
from keras.models import Model
from keras.utils import plot_model
import numpy as np
import keras.backend as K

def reshape_model(input_shape):
    x_input=Input(input_shape)
    x = Reshape((2,3,2))(x_input)
    x = Permute((1,3,2))(x)
    x1=[]
    x1.append(Lambda(lambda z:z[:,:,:,0])(x))
    x1.append(Lambda(lambda z:z[:,:,:,1])(x))
    x1.append(Lambda(lambda z:z[:,:,:,2])(x))
    
    x=concatenate(x1)
    

    model=Model(inputs=x_input,outputs=x)
    #model.summary()
    plot_model(model,to_file='reshape.png',show_shapes=True)
    return model

if __name__ == '__main__':
    a=np.array([1,2,3,4,5,6,7,8,9,10,11,12])
    a=a.reshape((2,2,3))
    print(a)
    a=np.expand_dims(a,axis=0)
    model=reshape_model(a[0].shape)
    
    get_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                 [model.layers[6].output])
    # output in test mode = 0
    # layer_output = get_layer_output([a, 0])[0]
    # output in train mode = 1
    layer_output = get_layer_output([a, 1])[0]
    print(layer_output)