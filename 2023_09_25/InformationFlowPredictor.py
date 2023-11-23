import math,os,logging
import DatasetUtility as du
from abc import ABC, abstractmethod
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import dill

class InformationFlowPredictor:
    
    # metrics
    @staticmethod
    def one_hitRate_avg(y_true, y_pred):  #  avg of each row (# hit 1) / (# true 1)
        y_pred_int = tf.cast(tf.where(y_pred >= 0.5, tf.ones_like(y_pred), tf.zeros_like(y_pred)), dtype=tf.int32)
        y_true_int = tf.cast(y_true, dtype=tf.int32)
        y_hit_int = y_true_int & y_pred_int
        y_hit_int_sum = tf.reduce_sum(y_hit_int, axis=0)
        y_true_int_sum = tf.reduce_sum(y_true_int, axis=0)
        rate_each_row =  y_hit_int_sum / y_true_int_sum
        rate_each_row_filtered = tf.boolean_mask(rate_each_row, tf.math.logical_not(tf.math.is_nan(rate_each_row)))
        return tf.reduce_mean(rate_each_row_filtered)

    @staticmethod
    def zero_hitRate_avg(y_true, y_pred):  # (# hit 0) / (# true 0)
        y_pred_int = tf.cast(tf.where(y_pred < 0.5, tf.ones_like(y_pred), tf.zeros_like(y_pred)), dtype=tf.int32)
        y_true_int = tf.cast(tf.where(y_true == 0.0, tf.ones_like(y_true), tf.zeros_like(y_true)), dtype=tf.int32)
        y_hit_int = y_true_int & y_pred_int
        y_hit_int_sum = tf.reduce_sum(y_hit_int, axis=0)
        y_true_int_sum = tf.reduce_sum(y_true_int, axis=0)
        rate_each_row =  y_hit_int_sum / y_true_int_sum
        rate_each_row_filtered = tf.boolean_mask(rate_each_row, tf.math.logical_not(tf.math.is_nan(rate_each_row)))
        return tf.reduce_mean(rate_each_row_filtered)
    
    @staticmethod
    def prob_abs_err(y_true, y_pred):
        absolute_errors = tf.abs(y_true - y_pred)
        return absolute_errors

    # Attention!! this function is incorrect, wait for fixed. 
    @staticmethod
    def index_acc(y_true, y_pred):
        # 將 y_pred 的 index 做四捨五入成接近的整數
        y_pred_index = tf.round(y_pred[:, 0])
        # 將 y_pred_index 與 y_true 相比較，相等為 1，否則為 0
        equality = tf.cast(tf.equal(y_pred_index, y_true[:, 0]), dtype=tf.float32)
        # 計算平均準確度
        acc = tf.reduce_mean(equality)
        return acc
    
    '''
    @staticmethod
    def prob_abs_err_on_correct_predictions(y_true, y_pred):
        
        
        # 將 y_pred 的 index 做四捨五入成接近的整數
        y_pred_index = tf.round(y_pred[:, 0])
        # 找出預測正確的部分（index 相等）as boolean mask
        correct_predictions = tf.equal(y_true[:, 0], y_pred_index)  
        # 計算 prob 的絕對誤差, 並取出正確預測的部分
        absolute_errors = tf.abs(y_true[:, 1] - y_pred[:, 1])
        absolute_errors_correct = tf.boolean_mask(absolute_errors, correct_predictions)
        # 計算平均 prob 絕對誤差
        return tf.reduce_mean(absolute_errors_correct)
    '''

    # loss function
    @staticmethod
    def weighted_bincrossentropy(weight_zero = 0.1, weight_one = 1.0):
        """
        Calculates weighted binary cross entropy. The weights are fixed.
            
        This can be useful for unbalanced catagories.
        
        Adjust the weights here depending on what is required.
        
        For example if there are 10x as many positive classes as negative classes,
            if you adjust weight_zero = 1.0, weight_one = 0.1, then false positives 
            will be penalize 10 times as much as false negatives.

        """
        def custom_weighted_bincrossentropy(true, pred):
            # calculate the binary cross entropy
            true = tf.cast(true, dtype=tf.float32)
            bin_crossentropy = tf.keras.backend.binary_crossentropy(true, pred)
            # apply the weights
            weights = true * weight_one + (1 - true) * weight_zero
            weighted_bin_crossentropy = weights * bin_crossentropy 
            return tf.keras.backend.mean(weighted_bin_crossentropy)
        return custom_weighted_bincrossentropy

    def __init__(self, encoder:du.Encoder.AbstractEncoder,
                 hiddenLayerNums = 2, hiddenLayerDim = None,
                 loadModelDirPath = None):
        self.__encoder = encoder
        inputDim = encoder.getInputDim()
        relDim = encoder.getRelDim()
        if hiddenLayerDim is None:
            hiddenLayerDim = relDim
        self.__encoder = encoder
        self.__maxXForNormalization = None
        if loadModelDirPath is not None:  # load a old model
            with open(os.path.join(loadModelDirPath,"custom_weighted_bincrossentropy.pkl"), "rb") as loadModelFuncFile:
                custom_objects = {
                    "custom_weighted_bincrossentropy": dill.load(loadModelFuncFile),
                    "one_hitRate_avg" : InformationFlowPredictor.one_hitRate_avg,
                    "zero_hitRate_avg" : InformationFlowPredictor.zero_hitRate_avg,
                    "prob_abs_err" : InformationFlowPredictor.prob_abs_err,
                    "index_acc" : InformationFlowPredictor.index_acc
                    #"prob_abs_err_on_correct_predictions" : InformationFlowPredictor.prob_abs_err
                }
                self.__model = tf.keras.models.load_model(os.path.join(loadModelDirPath,"model.h5"), custom_objects = custom_objects)
            with open(os.path.join(loadModelDirPath,"maxXForNormalization.pkl"), "rb") as maxXForNormalizationFile:
                self.__maxXForNormalization = dill.load(maxXForNormalizationFile)
            
            # recompile the metric because of tf bug. Ref: https://stackoverflow.com/questions/65549053/typeerror-not-supported-between-instances-of-function-and-str
            bAcc = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
            metrics = {}
            if isinstance(self.__encoder, du.Encoder.BinaryEntryEncoder):
                metrics['prob'] = [bAcc, InformationFlowPredictor.one_hitRate_avg, InformationFlowPredictor.zero_hitRate_avg, InformationFlowPredictor.prob_abs_err]
            elif isinstance(self.__encoder, du.Encoder.SparseEntryEncoder):
                metrics['index'] = [InformationFlowPredictor.index_acc]
                metrics['prob'] = [InformationFlowPredictor.prob_abs_err]
            self.__model.compile(loss=self.__model.loss, optimizer=self.__model.optimizer, metrics=metrics)
        else:  # create a new model
            inputs = tf.keras.Input(shape=(inputDim,),dtype='float32')
            layer = inputs
            for _ in range(hiddenLayerNums):
                layer = tf.keras.layers.Dense(hiddenLayerDim, activation='relu')(layer)
                layer = tf.keras.layers.Dropout(0.1)(layer)
            outputs = []
            if isinstance(self.__encoder, du.Encoder.BinaryEntryEncoder):
                outputs.append(tf.keras.layers.Dense(relDim, activation='sigmoid', name='prob')(layer))
            elif isinstance(self.__encoder, du.Encoder.SparseEntryEncoder):
                indexLayer = tf.keras.layers.Dense(relDim, activation='linear', name='index')(layer)
                concatenatedLayer = tf.keras.layers.Concatenate()([indexLayer,layer])
                probLayer = tf.keras.layers.Dense(relDim, activation='sigmoid', name='prob')(concatenatedLayer)
                outputs.extend([indexLayer, probLayer])
            self.__model = tf.keras.Model(inputs=inputs, outputs=outputs)

            bAcc = tf.keras.metrics.BinaryAccuracy(threshold=0.5)

            # setting loss & metirc function
            loss = {}
            metrics = {}
            if isinstance(self.__encoder, du.Encoder.BinaryEntryEncoder):
                # according to 1/0 distribution in training data
                self.__weighted_bincrossentropy = InformationFlowPredictor.weighted_bincrossentropy(0.1, 1.0)
                loss['prob'] = self.__weighted_bincrossentropy
                metrics['prob'] = [bAcc, InformationFlowPredictor.one_hitRate_avg, InformationFlowPredictor.zero_hitRate_avg, InformationFlowPredictor.prob_abs_err]
            elif isinstance(self.__encoder, du.Encoder.SparseEntryEncoder):
                self.__weighted_bincrossentropy = InformationFlowPredictor.weighted_bincrossentropy(1.0, 1.0)
                loss['index'] = tf.keras.losses.mean_squared_error
                metrics['index'] = [InformationFlowPredictor.index_acc]
                loss['prob'] = self.__weighted_bincrossentropy
                metrics['prob'] = [InformationFlowPredictor.prob_abs_err]
            self.__model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001), loss=loss, metrics=metrics)
        
        self.__model.summary()
        return
    
    def train(self, datasetFilePath:str, epochs:int=100, batch_size:int=200, readSize:int=-1,
              saveModelDirPath:str = None, debugDirPath:str = None):
        callbacks = []
        if saveModelDirPath is not None:
            if not os.path.exists(saveModelDirPath):
                os.mkdir(saveModelDirPath)
            callbacks.extend([
                tf.keras.callbacks.ModelCheckpoint(os.path.join(saveModelDirPath,"model.h5"), save_best_only=True),
                tf.keras.callbacks.EarlyStopping(patience=5, monitor = 'val_loss', mode = 'min')
            ])
            with open(os.path.join(saveModelDirPath,"custom_weighted_bincrossentropy.pkl"), "wb") as saveModelFuncFile:
                dill.dump(self.__weighted_bincrossentropy, saveModelFuncFile)

        if debugDirPath is not None:
            if not os.path.exists(debugDirPath):
                os.mkdir(debugDirPath)
            callbacks.append(tf.keras.callbacks.TensorBoard(debugDirPath))

        with open(datasetFilePath,"r") as datasetFile:
            inputSetSize = self.__encoder.getInputSetSize()
            outputSetSize = self.__encoder.getOutputSetSize()
            trainDatasetX = []
            trainDatasetY = []
            for line in datasetFile: 
                entry = du.DatasetUtility.Entry.createFromTextFormat(line, inputSetSize, outputSetSize)
                x,y = self.__encoder.encode(entry)
                trainDatasetX.append(x)
                trainDatasetY.append(y)
            if readSize > 0:
                trainDatasetX = trainDatasetX[:readSize]
                trainDatasetY = trainDatasetY[:readSize]

            trainDatasetX = np.array(trainDatasetX)
            trainDatasetY = [np.vstack(item) for item in zip(*trainDatasetY)]
            if self.__maxXForNormalization is None:
                self.__maxXForNormalization = np.max(trainDatasetX, axis=0)
            trainDatasetX = self.__normalizeX(trainDatasetX)

        if saveModelDirPath is not None:
            with open(os.path.join(saveModelDirPath,"maxXForNormalization.pkl"), "wb") as maxXForNormalizationFile:
                dill.dump(self.__maxXForNormalization, maxXForNormalizationFile)

        self.__history = self.__model.fit(trainDatasetX, trainDatasetY, epochs=epochs,
                                   #batch_size=trainDatasetX.shape[0],
                                   batch_size=batch_size,
                                   validation_split=0.1, validation_freq=1,
                                   callbacks = callbacks,
                                   verbose = 1)
        if debugDirPath is not None:
            self.__plotLearningCurves(os.path.join(debugDirPath,"history.png"))
        return

    def evaluate(self, datasetFilePath:str, batch_size:int=200, readSize:int=-1,
                outputDirPath:str = None):

        with open(datasetFilePath,"r") as datasetFile:
            inputSetSize = self.__encoder.getInputSetSize()
            outputSetSize = self.__encoder.getOutputSetSize()
            testDatasetX = []
            testDatasetY = []
            for line in datasetFile: 
                entry = du.DatasetUtility.Entry.createFromTextFormat(line, inputSetSize, outputSetSize)
                x,y = self.__encoder.encode(entry)
                testDatasetX.append(x)
                testDatasetY.append(y)
            if readSize > 0:
                testDatasetX = testDatasetX[:readSize]
                testDatasetY = testDatasetY[:readSize]
            testDatasetX = np.array(testDatasetX)
            testDatasetY = [np.vstack(item) for item in zip(*testDatasetY)]
            testDatasetX = self.__normalizeX(testDatasetX)
            results = self.__model.evaluate(testDatasetX, testDatasetY, batch_size=200, return_dict = True, verbose = 2)
            if outputDirPath is not None:
                if not os.path.exists(outputDirPath):
                    os.mkdir(outputDirPath)
                with open(os.path.join(outputDirPath,"evaluateResult.txt"), "w") as evaluateResultFile:
                    evaluateResultFile.write(f"{results}\n")
        return

    def __normalizeX(self, datasetX):
        return datasetX / self.__maxXForNormalization

    def __plotLearningCurves(self, figureFilePath="history.png"):
        figure, axis = plt.subplots(1, 2)
        df = pd.DataFrame(self.__history.history)
        #print(df.columns)
        loss_df = df.filter(regex=".*loss")
        other_df = df.drop(loss_df.columns, axis=1)
        for columnName in loss_df.columns:
            axis[0].plot(loss_df.index, loss_df[columnName], label=columnName)
        axis[0].legend()
        for columnName in other_df.columns:
            axis[1].plot(other_df.index, other_df[columnName], label=columnName)
        axis[1].legend()
        plt.savefig(figureFilePath)
        #plt.show()
        

def __testing():
    
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
    

    #datasetFilePath = "./sample1.dataset"
    #inputSetSize,outputSetSize = 3,2

    #datasetFilePath = "./sample2.dataset"
    #inputSetSize,outputSetSize = 103,100

    #datasetFilePath = "./dataset/sample2/train_intRange_arrayLen10_interval3.dataset"
    #inputSetSize,outputSetSize = 13,10
    
    datasetFilePath = "./dataset/sample2/train_intRange_arrayLen100_interval3.dataset"
    inputSetSize,outputSetSize = 103,100

    encoder = du.Encoder.BinaryEntryEncoder(inputSetSize, outputSetSize)
    #encoder = du.Encoder.SparseEntryEncoder(inputSetSize, outputSetSize, maxInputNum = 3)

    #predictor = InformationFlowPredictor(encoder, hiddenLayerDim = 10000)
    predictor = InformationFlowPredictor(encoder)
    predictor.train(datasetFilePath, epochs=500, readSize=10000, saveModelDirPath = "output")
    
    datasetFilePath = "./dataset/sample2/test_intRange_arrayLen100_interval3.dataset"
    predictor = InformationFlowPredictor(encoder, loadModelDirPath = "output")
    predictor.evaluate(datasetFilePath, outputDirPath = "output")
    
    return

def __testing2():
    '''
    f = InformationFlowPredictor.weighted_bincrossentropy(1.0, 1.0)
    with open('myCustom2.pkl', 'wb') as file:
        dill.dump(f, file)

    '''
    f1 = None
    f2 = None
    true = [0.0, 0.0, 1.0]
    pred = [1.0, 1.0, 0.0]
    with open('myCustom1.pkl', 'rb') as file:
        f1 = dill.load(file)
        print(f1.__name__)
        print(f1(true, pred))
    with open('myCustom2.pkl', 'rb') as file:
        f2 = dill.load(file)
        print(f2.__name__)
        print(f2(true, pred))
    
    print(f'sss = {f1 == f2}')
    print(f'sss = {f1(true, pred)}, {f2(true, pred)}')
    return
#__testing2()

def __testingMetricAndLoss():
    true = tf.constant([[1, 0.5], [2, 0.3], [3, 0.8]])
    pred = tf.constant([[1.2, 0.4], [1.8, 0.6], [1.1, 0.7]])
    
    tf.print(InformationFlowPredictor.index_acc(true, pred))
    tf.print(InformationFlowPredictor.prob_abs_err_on_correct_predictions(true, pred))
    tf.print(tf.keras.losses.mean_squared_error(true[:,0], pred[:,0]))
    return

#__testing()
#__testing2()
#__testingMetricAndLoss()