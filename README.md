###  10/27
1. create a new class binarymodel and remove a sequential
2. Change every sparse->binary
3. create a new class binarycrossentropy

###  11/24
1. finish binary config3 embedding

# Description
### 動機與目的：
https://senselabs.atlassian.net/wiki/spaces/SL/pages/170950657

### 目前進度：
2023-09-26
+ tensorflow version sparse/binary encoding model (but metric for sparse is wrong)
+ pytorch version sparse encoding model ([架構參考](https://senselabs.atlassian.net/wiki/spaces/SL/pages/170950657/NN#NN-Model))
+ the datasets of the three samples ([參考](https://senselabs.atlassian.net/wiki/spaces/SL/pages/218497027))

2023-11-13
+ 修訂DatasetUtility.py內的encoder
  * 修改encoder，初始化加入nonFloatIndices變數來告知輸入資料哪些不屬於數值類
  * 修改BinaryEncoder/BinaryEncoder.encode，不屬於數值類的輸入將會套用byte encoding方式處理，並調整最後輸出格式
+ 修訂pytorch version，調整模型架構(add simple character embedding)使其能夠處理不定常輸入
  * 模型內的DFNN inputDim必須重新設定(因取決於embedding layer維度)

### 預計規劃：
+ [ ] pytorch version binary encoding model with simple character embedding for variable inputs
+ [ ] 整合程式碼，類似於`InformationFlowPredictor.py`(https://senselabs.atlassian.net/wiki/spaces/SL/pages/181108737)
  * 會需要實做decode來解釋模型輸出結果
+ [ ] 實際資料測試
+ [ ] 模型優化
  * patience設計 (訓練到一定程度自動停止)
  * batch設計 (還不確定需要)
+ [ ] 超參數相關實驗探討
  * DFNN hidden layer dim.
  * Embedding layer dim.

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

### tensorflow version (已確認無法運行)
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

### tensorflow version (已確認無法運行)
1. 參考SampleExp.py中的最內層predictor的使用方式
2. 由於encoder修改，確定不能運行


