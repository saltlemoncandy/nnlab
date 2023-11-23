###  10/27
1. create a new class binarymodel and remove a sequential
2. Change every sparse->binary
3. create a new class binarycrossentropy
4. 


### todo
1. finish models and the forward in binarymodel
2. implement cross entropy
3. fix validation() and mapping
4. find out what else need fixing

# Description
### 動機與目的：
https://senselabs.atlassian.net/wiki/spaces/SL/pages/170950657

### 目前進度：
1. tensorflow version sparse/binary encoding model (but metric for sparse is wrong)
2. pytorch version sparse encoding model
3. the datasets of the three samples [參考](https://senselabs.atlassian.net/wiki/spaces/SL/pages/218497027)

### 預計規劃：
1. pytorch version binary encoding model
2. 處理非固定長度輸入(加入embedding和lstm層)
3. 整合程式碼，類似於`InformationFlowPredictor.py`(https://senselabs.atlassian.net/wiki/spaces/SL/pages/181108737)
4. 實際資料測試

# Structure
```
Shared
./
├── DatasetUtility.py # 生成dataset和encoder class
├── Dataset/ # 含有dataset資料，[資料格式](https://https://senselabs.atlassian.net/wiki/spaces/SL/pages/181108737)
├──├── sample1/
├──├── sample2/
├──├── sample3/

pytorch version
./
├── pytorchVersion(Test).py/  # 模型半成品檔案

tensorflow version (not testing)
./
├── InformationFlowPredictor.py/  # 整合過的模型檔案，只有Binary是正確的，Sparse的index accuracy metric運算有誤
├── tensorflowVersion(Test).py  # 模型半成品檔案
├── SampleExp.py     # for sample2 experiments
└── output/     # 訓練過的model暫存處

```

# Requirement
建議pytorch和tensorflow不要裝在同一個環境，容易有問題
### pytorch version
Python 3.9.10
pytorch 2.0.1
numpy 1.25.2
pandas 2.0.3

### tensorflow version (未被確認是否能運行)
Python 3.9.10
tensorflow 2.10.0
matplotlib 3.7.2
numpy 1.24.3
pandas 2.0.3
dill 0.3.6

# Setup & Execution Steps
### dataset generation (若已有dataset可跳過)
1. 修改DatasetUtility.py內的__testing()函數
```
2. python 
3. import DatasetUtility
4. DatasetUtility.testing()
```  


### pytorch version
1. 修改pytorchVersion(Test).py內的__test()函數內的config變數，選擇使用哪個sample資料集
`config = conifgSample3  # conifgSample1 / conifgSample2 / conifgSample3`
2. `python pytorchVersion(Test).py`

### tensorflow version (未被確認是否能運行)
1. 參考SampleExp.py中的最內層predictor的使用方式
2. 因沒有這個版本已久沒測試，不確定能不能運行


