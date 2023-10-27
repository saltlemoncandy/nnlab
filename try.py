import logging
import DatasetUtility as du
from abc import ABC, abstractmethod
import dill
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import timeit
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"
torch.set_printoptions(threshold=float('inf'))

class SparseModel(nn.Module):
    def __init__(self, encoder:du.Encoder.AbstractEncoder,
                 hiddenDim): 
        super(SparseModel, self).__init__()
        self.encoder = encoder
        inputDim = encoder.getInputDim()
        relDim = encoder.getRelDim()
        # network layer
        self.dfnn_hidden = nn.Sequential(
            nn.Linear(inputDim, hiddenDim),
            nn.ReLU(),
            nn.Linear(hiddenDim, hiddenDim),
            nn.ReLU()
        )
        self.dfnn_index = nn.Sequential(
            nn.Linear(hiddenDim, relDim),
            nn.LeakyReLU(),
        )
        self.dfnn_prob = nn.Sequential(
            nn.Linear(hiddenDim + relDim, hiddenDim),
            nn.ReLU(),
            nn.Linear(hiddenDim, relDim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # forward propagation
        hidden = self.dfnn_hidden(x)
        index = self.dfnn_index(hidden)
        prob = self.dfnn_prob(torch.cat((hidden, index),dim=1))

        #return torch.stack((index,prob),dim=0)
        return [index, prob]
    
class BinaryModel(nn.Module):
    def __init__(self, encoder:du.Encoder.AbstractEncoder,
                 hiddenDim): 
        super(BinaryModel, self).__init__()
        self.encoder = encoder
        inputDim = encoder.getInputDim()
        relDim = encoder.getRelDim()
        # network layer
        self.dfnn_hidden = nn.Sequential(
            nn.Linear(inputDim, hiddenDim),
            nn.ReLU(),
            nn.Linear(hiddenDim, hiddenDim),
            nn.ReLU()
        )
        self.dfnn_prob = nn.Sequential(
            nn.Linear(hiddenDim + relDim, hiddenDim),
            nn.ReLU(),
            nn.Linear(hiddenDim, relDim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # forward propagation
        hidden = self.dfnn_hidden(x)
        prob = self.dfnn_prob(hidden)

        #return torch.stack((index,prob),dim=0)
        return prob

    
def validation(model, encoder:du.Encoder.AbstractEncoder, validDatasetX, validDatasetY):
    '''
    this only implement index metric, and is evaluated as inefficiency 
    '''
    model.eval()
    index_acc, prob_abs_err = 0.0, 0.0
    FP_sum,FN_sum = 0.0, 0.0
    TP_sum,TN_sum = 0.0, 0.0
    batch_size = validDatasetX.shape[0]
    #print(f"batchsize = {validDatasetX.shape[0]}")
    #print(f"batchsize = {len(validDatasetY)}")
    
    inputSetSize = encoder.getInputSetSize()
    outputSetSize = encoder.getOutputSetSize()
    maxInputNum = encoder.getMaxInputNum()
    with torch.no_grad():
        # Get model predictions
        batch_pred = model(validDatasetX.to(device=DEVICE, dtype=torch.float))
        # batch_pred shape = [batch_size, encoder.getRelDim()] * 2
        batch_pred_index = torch.round(batch_pred[0]).view(-1, outputSetSize, maxInputNum)
        batch_pred_index = torch.where(torch.logical_or(batch_pred_index < 0, batch_pred_index >= inputSetSize), du.Encoder.SparseEntryEncoder.NULL_INDEX, batch_pred_index)
        batch_pred_prob = batch_pred[1].view(-1, outputSetSize, maxInputNum)
        # batch_pred_index shape = [batch_size, outputSetSize, maxInputNum]
        
        # validDatasetY shape = [batch_size, encoder.getRelDim()] * 2
        batch_true_index = validDatasetY[0].view(-1, outputSetSize, maxInputNum)
        batch_true_index = torch.where(torch.logical_or(batch_true_index < 0, batch_true_index >= inputSetSize), du.Encoder.SparseEntryEncoder.NULL_INDEX, batch_true_index)
        batch_true_prob = validDatasetY[1].view(-1, outputSetSize, maxInputNum)
        # batch_true_index shape = [batch_size, outputSetSize, maxInputNum]
        '''
        Formulas: 
        # true = [-4., -1.]
        # pred = [-1, 3.]
        # all = inputSetSize = 10
        # TP = (true & pred) - {-1}
        # FP = pred - {-1} - TP
        # FN = true - {-1} - TP
        # FP+FN = (true xor pred) - {-1}
        # TN = all - TP - FP - FN
        # acc = (TP+TN)/(TP+TN+FP+FN) = (all-(FP+FN))/all
        '''
        start = timeit.default_timer()
        for i in range(batch_true_index.shape[0]):
            for outputIndex in range(batch_true_index.shape[1]):
                true_indices = batch_true_index[i][outputIndex].numpy(force=True)
                pred_indices = batch_pred_index[i][outputIndex].numpy(force=True)
                # np version
                xor = np.setxor1d(true_indices, pred_indices)
                TPTN = encoder.getInputSetSize() - xor.shape[0] + (1. if (du.Encoder.SparseEntryEncoder.NULL_INDEX in xor) else 0.)
                intersect = np.intersect1d(true_indices,pred_indices)
                TP = intersect.shape[0] + (-1. if (du.Encoder.SparseEntryEncoder.NULL_INDEX in intersect) else 0.)
                TN = TPTN - TP
                FP_set = np.setdiff1d(pred_indices, intersect)
                FP = FP_set.shape[0] + (-1. if (du.Encoder.SparseEntryEncoder.NULL_INDEX in FP_set) else 0.)
                FN_set = np.setdiff1d(true_indices, intersect)
                FN = FN_set.shape[0] + (-1. if (du.Encoder.SparseEntryEncoder.NULL_INDEX in FN_set) else 0.)
                # python version but nealy equal to np version
                #xor = set(true_indices) ^ set(pred_indices)
                #offsetForNullIndex = 1. if (SparseEntryEncoder.NULL_INDEX in xor) else 0.
                #acc = (encoder.getInputSetSize() - len(xor) + offsetForNullIndex)
                
                index_acc += TPTN
                FP_sum += FP
                FN_sum += FN
                TP_sum += TP
                TN_sum += TN
                
        index_acc = index_acc / inputSetSize / outputSetSize / batch_size 
        FP_sum = FP_sum / inputSetSize / outputSetSize / batch_size
        FN_sum = FN_sum / inputSetSize / outputSetSize / batch_size
        TP_sum = TP_sum / inputSetSize / outputSetSize / batch_size
        TN_sum = TN_sum / inputSetSize / outputSetSize / batch_size
        end = timeit.default_timer()
        print(f"Time taken is {end - start}s")
    metrics = {"index_acc": index_acc, "prob_abs_err": prob_abs_err,
                "FP": FP_sum, "FN": FN_sum, "TP": TP_sum, "TN": TN_sum}
    return metrics

def validation_mapping(model, encoder:du.Encoder.AbstractEncoder, validDatasetX, validDatasetY,
                       flowThreshold:float = 0.01, flowWeight:float = 1.0):
    '''
    \n faster implementation
    \n batch mode has not been implemented yet
    '''
    model.eval()
    prob_abs_err = 0.0
    FP_sum,FN_sum = 0.0, 0.0
    TP_sum,TN_sum = 0.0, 0.0
    batch_size = validDatasetX.shape[0]
    #print(f"batchsize = {validDatasetX.shape[0]}")
    #print(f"batchsize = {len(validDatasetY)}")
    
    inputSetSize = encoder.getInputSetSize()
    outputSetSize = encoder.getOutputSetSize()
    # maxInputNum = encoder.getMaxInputNum()
    with torch.no_grad():
        # Get model predictions
        batch_pred = model(validDatasetX.to(device=DEVICE, dtype=torch.float))
        # batch_pred shape = [batch_size, encoder.getRelDim()] * 2
        batch_pred_index = torch.round(batch_pred[0]).view(-1, outputSetSize, maxInputNum)
        batch_pred_prob = batch_pred[1].view(-1, outputSetSize, maxInputNum)
        # batch_pred_index shape = [batch_size, outputSetSize, maxInputNum]
        
        # validDatasetY shape = [batch_size, encoder.getRelDim()] * 2
        batch_true_index = validDatasetY[0].view(-1, outputSetSize, maxInputNum)
        batch_true_prob = validDatasetY[1].view(-1, outputSetSize, maxInputNum)
        # batch_true_index shape = [batch_size, outputSetSize, maxInputNum]
        
        #start = timeit.default_timer()
        new_tensor = torch.where(torch.logical_or(batch_pred_index < 0, batch_pred_index >= inputSetSize), inputSetSize, batch_pred_index).to(device=DEVICE, dtype=torch.int64)
        batch_pred_relation = torch.zeros((batch_size, outputSetSize, inputSetSize+1), dtype=torch.float).to(device=DEVICE, dtype=torch.float).scatter_(2, new_tensor, batch_pred_prob)[:,:, :-1]
        
        new_tensor = torch.where(torch.logical_or(batch_true_index < 0, batch_true_index >= inputSetSize), inputSetSize, batch_true_index).to(device=DEVICE, dtype=torch.int64)
        batch_true_relation = torch.zeros((batch_size, outputSetSize, inputSetSize+1), dtype=torch.float).to(device=DEVICE, dtype=torch.float).scatter_(2, new_tensor, batch_true_prob)[:,:, :-1]
        
        TP_sum = torch.sum((batch_true_relation > flowThreshold) & (batch_pred_relation > flowThreshold)).item() / inputSetSize / outputSetSize / batch_size
        TN_sum = torch.sum((batch_true_relation <= flowThreshold) & (batch_pred_relation <= flowThreshold)).item() / inputSetSize / outputSetSize / batch_size
        FP_sum = torch.sum((batch_true_relation <= flowThreshold) & (batch_pred_relation > flowThreshold)).item() / inputSetSize / outputSetSize / batch_size
        FN_sum = torch.sum((batch_true_relation > flowThreshold) & (batch_pred_relation <= flowThreshold)).item() / inputSetSize / outputSetSize / batch_size
        
        
        prob_abs_err = torch.mean(torch.abs(batch_true_relation - batch_pred_relation))
        weights = torch.where(batch_true_relation > flowThreshold, flowWeight, 1.0)
        weighted_prob_abs_err = torch.sum(torch.abs(batch_true_relation - batch_pred_relation) * weights) / torch.sum(weights)
        
        index_acc = TP_sum + TN_sum
        #end = timeit.default_timer()
        #print(f"Time taken is {end - start}s")
        metrics = {"index_acc": index_acc, "prob_abs_err": prob_abs_err, "weighted_prob_abs_err": weighted_prob_abs_err,
                "FP": FP_sum, "FN": FN_sum, "TP": TP_sum, "TN": TN_sum}
    return metrics


def train(model, encoder:du.Encoder.AbstractEncoder, trainDatasetX, trainDatasetY, testDatasetX, testDatasetY):
    # loss function and optimizer
    #null_index_count = torch.sum(torch.eq(trainDatasetY[0], -1)).item()
    #normal_index_count = trainDatasetY[0].numel() - null_index_count
    null_index_count = 90000
    normal_index_count = 30000
    print(f'null_index_count = {null_index_count}')
    print(f'normal_index_count = {normal_index_count}')
    weightNormalIndex = float(null_index_count) / normal_index_count * 10
    #index_loss_fn = nn.MSELoss()
    # index_loss_fn = WeightedMSELoss(du.Encoder.BinaryEntryEncoder.NULL_INDEX, weightNormalIndex)
    #prob_loss_fn = nn.BCELoss()
    prob_loss_fn = WeightedBCELoss(weightNormalIndex)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    flowThreshold = 0.01
    flowWeight = weightNormalIndex
    eps = 0.1
    
    epochs = 5000
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad()
        # forward
        outputs = model(trainDatasetX)
        #loss_index = index_loss_fn(process_output(outputs[0],eps), trainDatasetY[0])
        # loss_index = index_loss_fn(outputs[0], trainDatasetY[0])
        loss_prob = prob_loss_fn(outputs[1], trainDatasetY[1])
        loss = loss_prob
        # backward
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.3f}, loss_prob: {loss_prob.item():.3f}')
        
        if epoch % epochs == 0:
            # validation_mapping
            metrics  = validation_mapping(model, encoder, testDatasetX, testDatasetY, flowThreshold=flowThreshold, flowWeight=flowWeight)
            print(f"index_acc: {metrics['index_acc']:.6f}")
            print(f"prob_abs_err: {metrics['prob_abs_err']:.6f}")
            print(f"weighted_prob_abs_err: {metrics['weighted_prob_abs_err']:.6f}")
            print(f"TP: {metrics['TP']:.6f}")
            print(f"TN: {metrics['TN']:.6f}")
            print(f"FP: {metrics['FP']:.6f}")
            print(f"FN: {metrics['FN']:.6f}")
            
            '''
            # validation
            metrics  = validation(encoder, trainDatasetX, trainDatasetY)
            print(f"index_acc: {metrics['index_acc']:.6f}")
            print(f"prob_abs_err: {metrics['prob_abs_err']:.6f}")
            print(f"TP: {metrics['TP']:.6f}")
            print(f"TN: {metrics['TN']:.6f}")
            print(f"FP: {metrics['FP']:.6f}")
            print(f"FN: {metrics['FN']:.6f}")
            '''
"""
# with dataLoader (still under testing)
def train(self):
    # loss function and optimizer
    #null_index_count = torch.sum(torch.eq(trainDatasetY[0], -1)).item()
    #normal_index_count = trainDatasetY[0].numel() - null_index_count
    null_index_count = 90000
    normal_index_count = 30000
    print(f'null_index_count = {null_index_count}')
    print(f'normal_index_count = {normal_index_count}')
    weightNormalIndex = float(null_index_count) / normal_index_count * 10
    #index_loss_fn = nn.MSELoss()
    index_loss_fn = WeightedMSELoss(weightNormalIndex)
    #prob_loss_fn = nn.BCELoss()
    prob_loss_fn = WeightedBCELoss(weightNormalIndex)
    optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
    flowThreshold = 0.01
    flowWeight = weightNormalIndex
    eps = 0.1
    
    epochs = 5000
    for epoch in range(1, epochs+1):
        self.train()
        for batchIndex, (batchX, batchY) in enumerate(trainDataLoader):
            optimizer.zero_grad()
            # forward
            outputs = self(batchX)
            #loss_index = index_loss_fn(process_output(outputs[0],eps), trainDatasetY[0])
            loss_index = index_loss_fn(outputs[0], batchY[0])
            loss_prob = prob_loss_fn(outputs[1], batchY[1])
            loss = loss_index + loss_prob
            # backward
            loss.backward()
            optimizer.step()

        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.3f}, loss_index: {loss_index.item():.3f}, loss_prob: {loss_prob.item():.3f}')
        
        if epoch % epochs == 0:
            # validation_mapping
            metrics  = validation_mapping(testDatasetX, testDatasetY, flowThreshold=flowThreshold, flowWeight=flowWeight)
            print(f"index_acc: {metrics['index_acc']:.6f}")
            print(f"prob_abs_err: {metrics['prob_abs_err']:.6f}")
            print(f"weighted_prob_abs_err: {metrics['weighted_prob_abs_err']:.6f}")
            print(f"TP: {metrics['TP']:.6f}")
            print(f"TN: {metrics['TN']:.6f}")
            print(f"FP: {metrics['FP']:.6f}")
            print(f"FN: {metrics['FN']:.6f}")
            
"""
        
class WeightedMSELoss(nn.Module):
    '''
    \n WeightedMSELoss = Total sum of `weightNormalIndex * (y_true - y_pred)^2` / TotalWeight
    '''
    def __init__(self, NULL_INDEX, weightNormalIndex):
        super(WeightedMSELoss, self).__init__()
        self.weightNormalIndex = weightNormalIndex
        self.NULL_INDEX = NULL_INDEX
    def forward(self, y_pred, y_true):
        weights = torch.where(y_true == self.NULL_INDEX, 1.0, self.weightNormalIndex)
        squaredErrors = torch.square(y_pred - y_true)
        weightedSquaredErrors = squaredErrors * weights
        loss = torch.sum(weightedSquaredErrors) / torch.sum(weights)
        return loss

class WeightedBCELoss(nn.Module):
    '''
    \n WeightedBCELoss = (- weightNormalIndex * y_true * log(y_pred) - (1 - y_true) * log(1 - y_pred)) / TotalWeight
    '''
    def __init__(self, weightNormalIndex, eps = 1e-12):
        super(WeightedBCELoss, self).__init__()
        self.weightNormalIndex = weightNormalIndex
        self.eps = eps
    
    def forward(self, y_pred_prob, y_true_prob):
        '''
        \n Optiontional improvement: 
        \n 1. `y_true_prob == 1.0` can be replaced with `y_true_index` for more flexibility on having a flow relationship
        '''
        weightedBCELoss = -(self.weightNormalIndex * y_true_prob * torch.log(torch.clamp(y_pred_prob, min=self.eps, max=1-self.eps)) + (1 - y_true_prob) * torch.log(torch.clamp(1 - y_pred_prob, min=self.eps, max=1-self.eps)))
        totalWeight = torch.sum(torch.where(y_true_prob == 1.0, self.weightNormalIndex, 1.0))
        loss = torch.sum(weightedBCELoss) / totalWeight
        return loss

class WeightedBinBrossEntropy(nn.Module):
    pass


class InformationFlowDataset(torch.utils.data.Dataset):
    ''' for dataLoader '''
    def __init__(self, encoder:du.Encoder.AbstractEncoder, datasetFilePath:str, device,
                readSize:int = -1, maxXForNormalization:float = None):
        # 定義初始化參數
        # 讀取資料集路徑
        self.encoder = encoder
        self.datasetX = [] # inputs
        self.datasetY = [] # outputs
        with open(datasetFilePath,"r") as datasetFile:
            inputSetSize = encoder.getInputSetSize()
            outputSetSize = encoder.getOutputSetSize()
            for line in datasetFile: 
                entry = du.DatasetUtility.Entry.createFromTextFormat(line, inputSetSize, outputSetSize)
                x,y = encoder.encode(entry)
                self.datasetX.append(x)
                self.datasetY.append(y)
            if readSize > 0:
                self.datasetX = self.datasetX[:readSize]
                self.datasetY = self.datasetY[:readSize]

            self.datasetX = np.array(self.datasetX)
            self.datasetY = [np.vstack(item) for item in zip(*self.datasetY)]
            if maxXForNormalization is None:
                self.maxXForNormalization = np.max(self.datasetX, axis=0)
            else:
                self.maxXForNormalization = maxXForNormalization
            
            self.datasetX = self.datasetX / self.maxXForNormalization
        # transform to tensor
        self.datasetX = torch.from_numpy(self.datasetX).to(device=device, dtype=torch.float)
        # trainDatasetX shape = [readSize, encoder.getInputDim()]
        self.datasetY = [torch.from_numpy(item).to(device=device, dtype=torch.float) for item in self.datasetY]
        # trainDatasetY shape = [readSize, encoder.getOutputDim()] * 2
        assert self.datasetX.shape[0] == self.datasetY[0].shape[0] == self.datasetY[1].shape[0]
    def __getitem__(self, index):
        # 讀取每次迭代的資料集中第 idx  資料
        # 進行前處理 (torchvision.Transform 等)
        return self.datasetX[index], [item[index] for item in self.datasetY] 
    
    def __len__(self):
        # 計算資料集總共數量
        return self.datasetX.shape[0]

    def getMaxXForNormalization(self):
        return self.maxXForNormalization

'''
def process_output(y_pred, eps):
    rounded_y_pred = torch.round(y_pred)
    diff = y_pred - rounded_y_pred
    null_index_y_pred = torch.where(diff > 0, encoder.NULL_INDEX + diff - 0.5, encoder.NULL_INDEX + diff + 0.5)
    processed_y_pred = torch.where(torch.logical_or(y_pred<0, torch.logical_or(y_pred>=encoder.getInputSetSize(), torch.abs(diff)<eps)), y_pred, null_index_y_pred)
    #print(y_pred)
    #print(processed_y_pred)
    return processed_y_pred
    

eps = 0.25
y_pred = torch.tensor([[1.2, 2.7],[3.5, 4.8]])
print(process_output(y_pred,eps))
y_pred2 = torch.tensor([[-100, 2.0],[3.66, 4.8]])
print(process_output(y_pred2,eps))


y_true = torch.tensor([[1.0, 0.0],[1.0, 0.0]])
y_pred = torch.tensor([[0.5, 0.0],[1.0, 0.0]])
print(nn.BCELoss()(y_pred,y_true))
print(WeightedBCELoss(10.0,1.0)(y_pred, y_true))
exit()
'''


def __testing():
    
    ''' dataset path '''
    # sample 1
    conifgSample1 = {
        'trainDatasetFilePath': "dataset/sample1/train_intRange.dataset",
        'testDatasetFilePath': "dataset/sample1/test_intRange.dataset",
        #'trainDatasetFilePath': "dataset/sample1/train_smallRange.dataset",
        #'testDatasetFilePath': "dataset/sample1/test_smallRange.dataset",
        'inputSetSize' : 3,
        'outputSetSize' : 3,
        'maxInputNum' : 3,
    }
    
    # sample 2
    arrayLen = 100   # 10 / 100
    interval = 3    # 1 / 2 / 3
    conifgSample2 = {
        'trainDatasetFilePath': f"dataset/sample2/train_intRange_arrayLen{arrayLen}_interval{interval}.dataset",
        'testDatasetFilePath': f"dataset/sample2/test_intRange_arrayLen{arrayLen}_interval{interval}.dataset",
        'inputSetSize' : arrayLen + 3,
        'outputSetSize' : arrayLen,
        'maxInputNum' : 3,
    }
    
    # sample 3
    seed = 0
    emb = 'IMPERFECT'  # PERFECT / IMPERFECT / RANDOM
    conifgSample3 = {
        'trainDatasetFilePath': f"dataset/sample3/train_seed{seed}_{emb}.dataset",
        'testDatasetFilePath': f"dataset/sample3/train_seed{seed}_{emb}.dataset",
        'inputSetSize' : 5,
        'outputSetSize' : 3,
        'maxInputNum' : 4,
    }
    
    # choose config
    config = conifgSample3  # conifgSample1 / conifgSample2 / conifgSample3
    readSize = 10000 # use how many data in training and testing. integer, -1~10000
    trainDatasetFilePath = config['trainDatasetFilePath']
    testDatasetFilePath = config['testDatasetFilePath']
    inputSetSize, outputSetSize = config['inputSetSize'], config['outputSetSize']
    maxInputNum = config['maxInputNum']
    
    ''' dataLoader Start (still under testing) '''
    '''
    trainDataset = InformationFlowDataset(encoder, trainDatasetFilePath, DEVICE, readSize=readSize, maxXForNormalization=maxXForNormalization)
    maxXForNormalization = trainDataset.getMaxXForNormalization()
    testDataset = InformationFlowDataset(encoder, testDatasetFilePath, DEVICE, readSize=readSize, maxXForNormalization=maxXForNormalization)

    batch_size = 10000
    shuffle = False
    trainDataLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=shuffle)
    '''
    ''' dataLoader End '''
    
    ''' Directly Dataset Read Start '''
    encoder = du.Encoder.BinaryEntryEncoder(inputSetSize,outputSetSize)
    maxXForNormalization = None
    with open(trainDatasetFilePath,"r") as trainDatasetFile:
        inputSetSize = encoder.getInputSetSize()
        outputSetSize = encoder.getOutputSetSize()
        trainDatasetX = []
        trainDatasetY = []
        for line in trainDatasetFile: 
            entry = du.DatasetUtility.Entry.createFromTextFormat(line, inputSetSize, outputSetSize)
            x,y = encoder.encode(entry)
            trainDatasetX.append(x)
            trainDatasetY.append(y)
        if readSize > 0:
            trainDatasetX = trainDatasetX[:readSize]
            trainDatasetY = trainDatasetY[:readSize]

        trainDatasetX = np.array(trainDatasetX)
        trainDatasetY = [np.vstack(item) for item in zip(*trainDatasetY)]
        if maxXForNormalization is None:
            maxXForNormalization = np.max(trainDatasetX, axis=0)
    
    trainDatasetX = trainDatasetX / maxXForNormalization

    with open(testDatasetFilePath,"r") as testDatasetFile:
        inputSetSize = encoder.getInputSetSize()
        outputSetSize = encoder.getOutputSetSize()
        testDatasetX = []
        testDatasetY = []
        for line in testDatasetFile: 
            entry = du.DatasetUtility.Entry.createFromTextFormat(line, inputSetSize, outputSetSize)
            x,y = encoder.encode(entry)
            testDatasetX.append(x)
            testDatasetY.append(y)
        if readSize > 0:
            testDatasetX = testDatasetX[:readSize]
            testDatasetY = testDatasetY[:readSize]

        testDatasetX = np.array(testDatasetX)
        testDatasetY = [np.vstack(item) for item in zip(*testDatasetY)]
        testDatasetX = testDatasetX / maxXForNormalization

    # transform data to tensor
    trainDatasetX = torch.from_numpy(trainDatasetX).to(device=DEVICE, dtype=torch.float)
    testDatasetX = torch.from_numpy(testDatasetX).to(device=DEVICE, dtype=torch.float)
    print(trainDatasetX[0])
    #print(trainDatasetX.shape)
    # trainDatasetX shape = [readSize, encoder.getInputDim()]
    trainDatasetY = [torch.from_numpy(item).to(device=DEVICE, dtype=torch.float) for item in trainDatasetY]
    testDatasetY = [torch.from_numpy(item).to(device=DEVICE, dtype=torch.float) for item in testDatasetY]
    #print(trainDatasetY[0].shape)
    # trainDatasetY shape = [readSize, encoder.getOutputDim()] * 2
    ''' Directly Dataset Read End '''

    # create NN model
    model = BinaryModel(encoder, encoder.getRelDim())
    model.to(device=DEVICE)
    
    # training
    train(model, encoder, trainDatasetX, trainDatasetY, testDatasetX, testDatasetY)

    # testing
    model.eval()
    with torch.no_grad():
        test_outputs = model(testDatasetX)
        #eps = 0.25
        #rounded_test_outputs = torch.round(process_output(test_outputs[0][0],eps))
        ''' show some case '''
        caseIndices = [0]  # [...]
        for caseIndex in caseIndices:
            print(f'case {caseIndex}: ')
            print(torch.round(test_outputs[0][caseIndex]).view(-1, outputSetSize, maxInputNum))
            print(testDatasetY[0][caseIndex].view(-1, outputSetSize, maxInputNum))
            print(test_outputs[1][caseIndex].view(-1, outputSetSize, maxInputNum))
            print(testDatasetY[1][caseIndex].view(-1, outputSetSize, maxInputNum))
        
__testing()
