import os
import InformationFlowPredictor as IFP
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')

# 定義測試的設定值集合
read_size_set = {100, 1000, 10000}
max_input_num_set = {1, 2, 3}
range_set = {'intRange', 'smallRange'}
array_len_set = {('arrayLen10', 13, 10), ('arrayLen100', 103, 100)}
interval_set = {'interval1', 'interval2', 'interval3'}

#
epochs = 500

# 設定資料集和模型儲存的根目錄
root_dir = './exp2'
dataset_dir = './dataset/sample2'
# 執行測試實驗
for read_size in read_size_set:
    for max_input_num in max_input_num_set:
        for range_val in range_set:
            for array_len,inputSetSize,outputSetSize in array_len_set:
                for interval in interval_set:
                    # 創建資料夾路徑
                    experiment_dir = os.path.join(root_dir, f'{range_val}_{array_len}_{interval}_readSize{read_size}_maxInput{max_input_num}')
                    #save_model_dir = os.path.join(experiment_dir, 'save_model_dir')
                    debug_dir = os.path.join(experiment_dir, 'debug_dir')
                    trainDatasetFilePath = os.path.join(dataset_dir, f'train_{range_val}_{array_len}_{interval}.dataset')
                    testDatasetFilePath = os.path.join(dataset_dir, f'train_{range_val}_{array_len}_{interval}.dataset')

                    # 建立資料夾
                    #os.makedirs(experiment_dir, exist_ok=True)
                    #os.makedirs(save_model_dir, exist_ok=True)
                    os.makedirs(debug_dir, exist_ok=True)

                    # 創建 InformationFlowPredictor 物件
                    encoder = IFP.SparseEntryEncoder(inputSetSize, outputSetSize, maxInputNum=max_input_num)
                    predictor = IFP.InformationFlowPredictor(encoder)

                    # 訓練模型
                    predictor.train(trainDatasetFilePath, epochs=epochs, readSize=read_size, saveModelDirPath=experiment_dir, debugDirPath=debug_dir)

                    # 評估模型
                    predictor = IFP.InformationFlowPredictor(encoder, loadModelDirPath=experiment_dir)
                    predictor.evaluate(testDatasetFilePath, outputDirPath=experiment_dir)