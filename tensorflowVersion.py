#!pip install matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import math,os

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')

# 
cfv_dim = None # control flow variable set dimension
i_dim = None  # input data flow variable set dimension
o_dim = None # output data flow variable set dimension

ivt_dim = None # NN input layer dim
rel_dim = None # NN output layer dim

X_train = None
Y_train = None

train_data_num = 10000

def GenSample1Data():
    '''
    external int a, b, c
    function foo(int x)
    if(x > 0)
    c = a
    else
    c = b
    return x

    I = {x,a,b}
    O = {c,ret}

    '''
    global cfv_dim, i_dim, o_dim, ivt_dim, rel_dim, X_train, Y_train
    cfv_dim = 3
    i_dim = 3
    o_dim = 2
    ivt_dim = cfv_dim
    rel_dim = i_dim * o_dim
    X_train = np.zeros((train_data_num,ivt_dim), dtype='int32')
    Y_train = np.zeros((train_data_num,rel_dim), dtype='int32')
    for i in range(train_data_num):
        x = np.random.randint(low=-10,high=10)
        a = np.random.randint(low=-10,high=10)
        b = np.random.randint(low=-10,high=10)
        X_train[i] = np.array([x,a,b], dtype='int32')
        x_to_c = False
        x_to_ret = True
        a_to_c = (x>0)
        a_to_ret = False
        b_to_c = (x<=0)
        b_to_ret = False
        Y_train[i] = np.array([x_to_c, x_to_ret, a_to_c, a_to_ret, b_to_c, b_to_ret], dtype='bool')
        print(X_train[i])
        print(Y_train[i])
    return 

def GenSample2Data(array_len):
    '''
    external int array_source[100]
    external int array_sink[100]
    function foo(int base, times, interval)
        count = 0
        for(int i=base; i>=0 && i<100 && count<times; i=i+interval)
            array_sink[i] = array_source[100-i-1]
            count = count + 1
        return    

    I = {base, times, interval, array_source[100]}
    O = {array_sink[100]}
    '''
    global cfv_dim, i_dim, o_dim, ivt_dim, rel_dim, X_train, Y_train
    cfv_dim = 3
    i_dim = array_len
    o_dim = array_len
    #ivt_dim = cfv_dim + array_len
    ivt_dim = cfv_dim
    #rel_dim = (cfv_dim + i_dim) * o_dim
    rel_dim = (i_dim) * o_dim
    X_train = np.zeros((train_data_num,ivt_dim), dtype='int32')
    Y_train = np.zeros((train_data_num,rel_dim), dtype='int32')
    for i in range(train_data_num):
        base = np.random.randint(low=int(array_len*0.4),high=int(array_len*0.6)+1)    # 0.4~0.6 array_len
        times = np.random.randint(low=1,high=11)     # 1~10
        intervalVals = [-3,-2,-1,1,2,3]
        interval = intervalVals[np.random.randint(low=0,high=len(intervalVals))]   # -3~3
        array_source = np.random.randint(low=-100,high=100, size=array_len, dtype='int32')
        X_train[i] = np.array([base,times,interval], dtype='int32')
        #X_train[i] = np.concatenate((np.array([base,times,interval], dtype='int32'), array_source))
        #base_to_array_sink = np.zeros((array_len,), dtype='bool')
        #times_to_array_sink = np.zeros((array_len,), dtype='bool')
        #interval_to_array_sink = np.zeros((array_len,), dtype='bool')
        array_source_to_array_sink = np.zeros((array_len,array_len), dtype='bool')
        for index in range(array_len): 
            if (index-base) % interval == 0 and (index-base) / interval >=0 and (index-base) / interval < times :
                array_source_to_array_sink[array_len-index-1][index] = True
        #Y_train[i] = np.concatenate((base_to_array_sink, times_to_array_sink, interval_to_array_sink, array_source_to_array_sink.flatten()))
        Y_train[i] = array_source_to_array_sink.flatten()
        print(X_train[i])
        print(Y_train[i])
    return 

def GenSample2DataRefined(array_len):
    '''
    external int array_source[100]
    external int array_sink[100]
    function foo(int base, times, interval)
        count = 0
        for(int i=base; i>=0 && i<100 && count<times; i=i+interval)
            array_sink[i] = array_source[100-i-1]
            count = count + 1
        return    

    I = {base, times, interval, array_source[100]}
    O = {array_sink[100]}
    '''
    global cfv_dim, i_dim, o_dim, ivt_dim, rel_dim, X_train, Y_train
    cfv_dim = 3
    i_dim = array_len
    o_dim = array_len
    ivt_dim = cfv_dim
    position_dim = int(math.ceil(math.log2(o_dim)))
    rel_dim = array_len * position_dim
    X_train = np.zeros((train_data_num,ivt_dim), dtype='int32')
    Y_train = np.zeros((train_data_num,rel_dim), dtype='int32')
    for i in range(train_data_num):
        base = np.random.randint(low=int(array_len*0.4),high=int(array_len*0.6)+1)    # 0.4~0.6 array_len
        times = np.random.randint(low=1,high=11)     # 1~10
        intervalVals = [-3,-2,-1,1,2,3]
        interval = intervalVals[np.random.randint(low=0,high=6)]   # -3~3
        X_train[i] = np.array([base,times,interval], dtype='int32')
        #array_source_to_array_sink = np.zeros((array_len,array_len), dtype='bool')
        array_sink_from_array_source = np.zeros((array_len,position_dim), dtype='bool')
        for index in range(array_len): 
            if (index-base) % interval == 0 and (index-base) / interval >=0 and (index-base) / interval < times :
                #array_source_to_array_sink[array_len-index-1][index] = True
                #print(f"index: {index}")
                #print(f"origin: {array_len-index-1}")
                binRep = bin(array_len-index-1)[2:].zfill(position_dim)[-position_dim:]  # '0000'
                #print(f"encode: {binRep}")
                binRepInInt = [int(x) for x in list(binRep)]
                array_sink_from_array_source[index] = np.array(binRepInInt, dtype='bool')
        Y_train[i] = array_sink_from_array_source.flatten()
        print(X_train[i])
        print(Y_train[i])
    return 

#GenSample1Data()
#GenSample2Data(100)
GenSample2DataRefined(100)

# 定義神經網路的參數
#ivt_dim = ...
#rel_dim = ...
print(f'ivf_dim = {ivt_dim}')
print(f'rel_dim = {rel_dim}')
hidden_layer_num = 2
hidden_layer_dims = []
for i in range(hidden_layer_num):
    #hidden_layer_dims.append(int(ivt_dim*math.pow(rel_dim/float(ivt_dim), ( float(i+1) / (hidden_layer_num+1)))))
    #hidden_layer_dims.append(int((train_data_num)/(2.0*(ivt_dim+rel_dim))))
    hidden_layer_dims.append(rel_dim)

print(f'{ivt_dim} {hidden_layer_dims} {rel_dim}')

# 建立神經網路的模型
model = tf.keras.models.Sequential()

# 加入輸入層
model.add(tf.keras.layers.Input(shape=(ivt_dim,),dtype='int32'))

# 加入多個隱藏層
for hidden_layer_dim in hidden_layer_dims:
    model.add(tf.keras.layers.Dense(hidden_layer_dim, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.1))

# 加入輸出層
model.add(tf.keras.layers.Dense(rel_dim, activation='sigmoid'))

# metrics
def one_hitRate_all(y_true, y_pred):  #  (# hit 1) / (# true 1)
    y_pred_int = tf.cast(tf.where(y_pred >= 0.5, tf.ones_like(y_pred), tf.zeros_like(y_pred)), dtype=tf.int32)
    y_true_int = tf.cast(y_true, dtype=tf.int32)

    k = y_true_int & y_pred_int
    return tf.math.count_nonzero(k) / tf.math.count_nonzero(y_true_int)

def one_hitRate_avg(y_true, y_pred):  #  avg of each row (# hit 1) / (# true 1)
    y_pred_int = tf.cast(tf.where(y_pred >= 0.5, tf.ones_like(y_pred), tf.zeros_like(y_pred)), dtype=tf.int32)
    y_true_int = tf.cast(y_true, dtype=tf.int32)
    y_hit_int = y_true_int & y_pred_int
    y_hit_int_sum = tf.reduce_sum(y_hit_int, axis=0)
    y_true_int_sum = tf.reduce_sum(y_true_int, axis=0)
    rate_each_row =  y_hit_int_sum / y_true_int_sum
    rate_each_row_filtered = tf.boolean_mask(rate_each_row, tf.math.logical_not(tf.math.is_nan(rate_each_row)))
    return tf.reduce_mean(rate_each_row_filtered)

def zero_hitRate_avg(y_true, y_pred):  # (# hit 0) / (# true 0)
    y_pred_int = tf.cast(tf.where(y_pred < 0.5, tf.ones_like(y_pred), tf.zeros_like(y_pred)), dtype=tf.int32)
    y_true_int = tf.cast(tf.where(y_true == 0.0, tf.ones_like(y_true), tf.zeros_like(y_true)), dtype=tf.int32)
    y_hit_int = y_true_int & y_pred_int
    y_hit_int_sum = tf.reduce_sum(y_hit_int, axis=0)
    y_true_int_sum = tf.reduce_sum(y_true_int, axis=0)
    rate_each_row =  y_hit_int_sum / y_true_int_sum
    rate_each_row_filtered = tf.boolean_mask(rate_each_row, tf.math.logical_not(tf.math.is_nan(rate_each_row)))
    return tf.reduce_mean(rate_each_row_filtered)   

bAcc = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
TN = tf.keras.metrics.TrueNegatives()
TP = tf.keras.metrics.TruePositives()
FN = tf.keras.metrics.FalseNegatives()
FP = tf.keras.metrics.FalsePositives()

def weighted_bincrossentropy(true, pred, weight_zero = 0.01, weight_one = 1.0):
    """
    Calculates weighted binary cross entropy. The weights are fixed.
        
    This can be useful for unbalanced catagories.
    
    Adjust the weights here depending on what is required.
    
    For example if there are 10x as many positive classes as negative classes,
        if you adjust weight_zero = 1.0, weight_one = 0.1, then false positives 
        will be penalize 10 times as much as false negatives.

    """
    # calculate the binary cross entropy
    true = tf.cast(true, dtype=tf.float32)
    bin_crossentropy = tf.keras.backend.binary_crossentropy(true, pred)
    # apply the weights
    weights = true * weight_one + (1 - true) * weight_zero
    weighted_bin_crossentropy = weights * bin_crossentropy 

    return tf.keras.backend.mean(weighted_bin_crossentropy)


def plotLearningCurves(history):
    df = pd.DataFrame(history.history)
    df.plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

# 編譯模型

#model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['binary_accuracy', one_hitRate_avg, zero_hitRate_avg, TP, FP, TN, FN])
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss=weighted_bincrossentropy, metrics=['binary_accuracy', one_hitRate_avg, zero_hitRate_avg])
model.summary()

logdir = "./results"
if not os.path.exists(logdir):
    os.mkdir(logdir)

output_model_file = os.path.join(logdir,"model.h5")

callbacks = [
    tf.keras.callbacks.TensorBoard(logdir),
    tf.keras.callbacks.ModelCheckpoint(output_model_file,save_best_only=True),
    tf.keras.callbacks.EarlyStopping(patience=100)
]

history = model.fit(X_train, Y_train, epochs=100, \
          #batch_size=X_train.shape[0], \
          batch_size=500, \
          #validation_split=0.1, validation_freq=1, \
          #callbacks = callbacks, \
          verbose = 1)

#model.fit(generator, steps_per_epoch=steps_per_epoch, epochs=10)

plotLearningCurves(history)
#X_test = np.array([[-2, -6, -3]])
#Y_test = np.array([[False, True, False, False, True, False]])
#model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))