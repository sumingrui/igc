from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import plot_model
import tensorflow as tf
import keras.backend as K

# 计算flops
def get_flops():
    run_meta = tf.RunMetadata()

    opts = tf.profiler.ProfileOptionBuilder.float_operation()
    flops = tf.profiler.profile(graph=K.get_session().graph,run_meta=run_meta, cmd='op', options=opts)

    opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()    
    params = tf.profiler.profile(graph=K.get_session().graph,run_meta=run_meta, cmd='op', options=opts)
    
    #return flops.total_float_ops  # Prints the "flops" of the model.
    print('flops',flops.total_float_ops)
    print('params',params.total_parameters)




# 这部分返回一个张量
inputs = Input(shape=(784,))

# 层的实例是可调用的，它以张量为参数，并且返回一个张量
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# 这部分创建了一个包含输入层和三个全连接层的模型
model = Model(inputs=inputs, outputs=predictions)
# model.compile(optimizer='rmsprop',
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
# model.fit(data, labels)  # 开始训练

plot_model(model, to_file='model.png', show_shapes=True)
model.summary()
get_flops()