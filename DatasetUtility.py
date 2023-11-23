import logging
from abc import ABC, abstractmethod
import numpy as np
import io, math, random, string, re
from enum import Enum

class DatasetUtility:
    RandomIntMin = -2147483648
    RandomIntMax = 2147483647
    '''
    data format:
    {"inputSet":[], "outputSet":[]}
    ivt1 ivt2 ... ivtn : relOut1Src1@relOut1Src1Prob relOut1Src2@relOut1Src2Prob ... relOut1Srck1@relOut1SrckProb : relOut2Src1@relOut2Src1Prob relOut2Src2@relOut2Src2Prob ... relOut2Srck2@relOut1Srck2Prob : ... : \n

    '''
    class Entry:
        def __init__(self, inputSetSize:int, outputSetSize:int):
            self.__inputValueSet = [0.0] * inputSetSize
            self.__relSet = []
            for i in range(outputSetSize):
                self.__relSet.append([])

        def putInputValue(self, index:int, value):
            self.__inputValueSet[index] = value
            return
        
        def putRelation(self, outputIndex:int, inputIndex:int, prob=1.0):
            self.__relSet[outputIndex].append((inputIndex, prob))
            return 

        # still not be tested
        # Warning: nonFloat inputValue cannot contain ':'
        @staticmethod
        def createFromTextFormat(text:str, inputSetSize:int, outputSetSize:int, nonFloatIndices:list[int]=[]):
            entry = DatasetUtility.Entry(inputSetSize, outputSetSize)
            textSplit = text.split(":")
            for inputIndex, inputValue in enumerate(filter(None,textSplit[0].split(" "))):
                if inputIndex in nonFloatIndices:
                    entry.putInputValue(inputIndex, inputValue)
                else:
                    entry.putInputValue(inputIndex, float(inputValue))
                    
            for i in range(1, len(textSplit)-1):
                outputIndex = i-1
                for relStr in filter(None,textSplit[i].split(" ")):
                    relStrSplit = relStr.split("@")
                    entry.putRelation(outputIndex, int(relStrSplit[0]), float(relStrSplit[1]))
            return entry

        def toTextFormat(self):
            stringBuffer = []
            for inputValue in self.__inputValueSet:
                stringBuffer.append(f"{inputValue}")
            stringBuffer.append(":")
            for outputIndex in range(len(self.__relSet)):
                for rel in self.__relSet[outputIndex]:
                    stringBuffer.append(f"{rel[0]}@{rel[1]}")
                stringBuffer.append(":")
            return ' '.join(stringBuffer)

    @staticmethod
    def GenSample1Data(outputFilePath:str, trainingDataNum:int,
                       seed = None):
        '''
        Sample 1

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
        if seed is not None:
            random.seed(seed)
        inputSetDict = {"x":0,"a":1,"b":2}
        outputSetDict = {"c":0,"ret":1}
        outputFile = None
        try:
            outputFile = open(outputFilePath,"w")
            for i in range(trainingDataNum):
                entry = DatasetUtility.Entry(len(inputSetDict), len(outputSetDict))
                x = random.randrange(DatasetUtility.RandomIntMin, DatasetUtility.RandomIntMax)
                a = random.randrange(DatasetUtility.RandomIntMin, DatasetUtility.RandomIntMax)
                b = random.randrange(DatasetUtility.RandomIntMin, DatasetUtility.RandomIntMax)
                entry.putInputValue(inputSetDict.get("x"), float(x))
                entry.putInputValue(inputSetDict.get("a"), float(a))
                entry.putInputValue(inputSetDict.get("b"), float(b))
                entry.putRelation(outputSetDict.get("c"), inputSetDict.get("x"), 0.0) # x_to_c = False
                entry.putRelation(outputSetDict.get("ret"), inputSetDict.get("x"), 1.0) # x_to_ret = True
                entry.putRelation(outputSetDict.get("c"), inputSetDict.get("a"), 1.0 if (x>0) else 0.0) # a_to_c = (x>0)
                entry.putRelation(outputSetDict.get("ret"), inputSetDict.get("a"), 0.0) # a_to_ret = False
                entry.putRelation(outputSetDict.get("c"), inputSetDict.get("b"), 1.0 if (x<=0) else 0.0) # b_to_c = (x<=0)
                entry.putRelation(outputSetDict.get("ret"), inputSetDict.get("b"), 0.0) # b_to_ret = False
                #print(f"{entry.toTextFormat()}")
                outputFile.write(f"{entry.toTextFormat()}\n")
        finally:
            if outputFile is not None:
                outputFile.close
        return 

    @staticmethod
    def GenSample2Data(outputFilePath:str, trainingDataNum:int,
                        arrayLen:int, seed = None):
        '''
        Sample 2

        external int array_source[arrayLen]
        external int array_sink[arrayLen]
        function foo(int base, times, interval)
            count = 0
            for(int i=base; i>=0 && i<arrayLen && count<times; i=i+interval)
                array_sink[i] = array_source[arrayLen-1-i]
                count = count + 1
            return    

        I = {base, times, interval, array_source[arrayLen]}
        O = {array_sink[arrayLen]}
        '''
        if seed is not None:
            random.seed(seed)
        inputSetDict = {"base":0,"times":1,"interval":2}
        outputSetDict = {}
        for i in range(arrayLen):
            inputSetDict[f"array_source_{i}"] = i+3
            outputSetDict[f"array_sink_{i}"] = i
        
        outputFile = None
        try:
            outputFile = open(outputFilePath,"w")
            for i in range(trainingDataNum):
                entry = DatasetUtility.Entry(len(inputSetDict), len(outputSetDict))
                base = random.randrange(int(arrayLen*0.4),int(arrayLen*0.6)) # 0.4~0.6 array_len
                times = random.randrange(1,10) # 1~10
                #intervalVals = [-3,-2,-1,1,2,3]
                #intervalVals = [-2,-1,1,2]
                intervalVals = [-1,1]
                interval = intervalVals[random.randrange(0,len(intervalVals))]  # -3~3
                array_source = [random.randrange(DatasetUtility.RandomIntMin, DatasetUtility.RandomIntMax) for _ in range(arrayLen)]
                entry.putInputValue(inputSetDict.get("base"), float(base))
                entry.putInputValue(inputSetDict.get("times"), float(times))
                entry.putInputValue(inputSetDict.get("interval"), float(interval))
                for i,source in enumerate(array_source):
                    entry.putInputValue(inputSetDict.get(f"array_source_{i}"), float(source))
                for index in range(arrayLen): 
                    if (index-base) % interval == 0 and (index-base) / interval >=0 and (index-base) / interval < times:
                        entry.putRelation(outputSetDict.get(f"array_sink_{index}"), inputSetDict.get(f"array_source_{arrayLen-1-index}"), 1.0) # array_source[arrayLen-1-index]_to_array_sink[index] = True
                outputFile.write(f"{entry.toTextFormat()}\n")
        finally:
            if outputFile is not None:
                outputFile.close
        return 
    
    class EmbText(Enum):
        NONE = 0
        RANDOM = 1
        IMPERFECT = 2
        PERFECT = 3
        IMPERFECT_TEXT = 4
        PERFECT_TEXT = 5
    
    @staticmethod
    def GenSample3Data(outputFilePath:str, trainingDataNum:int,
                       embText:EmbText = EmbText.NONE,
                       seed = None):
        '''
        Sample 3

        external int src1, src2, src3
        external int sink1, sink2, sink3
        function foo(int key, char[] text)
            # text embedding for simulating text class
            emb = 0
            for c in text
                emb = (emb*key + int(c)) % 8

            if emb in [0,1,2,3]:
                sink1 = src1
            else:
                sink2 = src1

            if emb in [0,2,4,6]:
                sink2 = src2
            else:
                sink3 = src2
            
            if emb in [0,1,4,5]:
                sink3 = src3
            else:
                sink1 = src3
            
            # relation for different emb
            #0: sink1:[src1], sink2:[src2], sink3[src3]
            #1: sink1:[src1], sink2:[], sink3[src2,src3]
            #2: sink1:[src1,src3], sink2:[src2], sink3[]
            #3: sink1:[src1,src3], sink2:[], sink3[src2]
            #4: sink1:[], sink2:[src1,src2], sink3[src3]
            #5: sink1:[], sink2:[src1], sink3[src2,src3]
            #6: sink1:[src3], sink2:[src1,src2], sink3[]
            #7: sink1:[src3], sink2:[src1], sink3[src2]
            return    

        I = {key, text, src1, src2, src3}
        O = {sink1, sink2, sink3}
        '''
        randomData = random.Random()
        randomSpecial = random.Random()  # for PERFECT_TEXT and IMPERFECT_TEXT
        if seed is not None:
            randomData.seed(seed)
            randomSpecial.seed(seed)
            
        inputSetDict = {"key":0,"text":1,"src1":2, "src2":3, "src3":4}
        outputSetDict = {"sink1":0,"sink2":1,"sink3":2}
        outputFile = None
        try:  
            outputFile = open(outputFilePath,"w")
            charactersSet = string.ascii_letters + string.digits
            textMinLen = 10
            textMaxLen = 30
            specialTextList = [''.join(randomSpecial.choice(charactersSet) for _ in range(randomSpecial.randrange(textMinLen, textMaxLen))) for _ in range(8)] # for PERFECT_TEXT and IMPERFECT_TEXT
            
            for i in range(trainingDataNum):
                entry = DatasetUtility.Entry(len(inputSetDict), len(outputSetDict))          
                key = randomData.randrange(DatasetUtility.RandomIntMin, DatasetUtility.RandomIntMax)
                
                textLen = randomData.randrange(textMinLen, textMaxLen)
                text = ''.join(randomData.choice(charactersSet) for _ in range(textLen))
                emb = 0
                for c in text:
                    emb = (emb*key + ord(c)) % 8
                
                match(embText):
                    case DatasetUtility.EmbText.NONE:
                        pass
                    case DatasetUtility.EmbText.RANDOM:
                        text = float(hash(text)) % 100
                    case DatasetUtility.EmbText.IMPERFECT:
                        text = randomData.gauss(emb, 1)
                    case DatasetUtility.EmbText.PERFECT:
                        text = float(emb)
                    case DatasetUtility.EmbText.IMPERFECT_TEXT:
                        # only change last character
                        text = specialTextList[emb][:-1] + randomSpecial.choice(charactersSet)
                    case DatasetUtility.EmbText.PERFECT_TEXT:
                        text = specialTextList[emb]
                    
                src1 = randomData.randrange(DatasetUtility.RandomIntMin, DatasetUtility.RandomIntMax)
                src2 = randomData.randrange(DatasetUtility.RandomIntMin, DatasetUtility.RandomIntMax)
                src3 = randomData.randrange(DatasetUtility.RandomIntMin, DatasetUtility.RandomIntMax)

                entry.putInputValue(inputSetDict.get("key"), float(key))
                entry.putInputValue(inputSetDict.get("text"), text)
                entry.putInputValue(inputSetDict.get("src1"), float(src1))
                entry.putInputValue(inputSetDict.get("src2"), float(src2))
                entry.putInputValue(inputSetDict.get("src3"), float(src3))

                # relation default is 0.0, so key and text do not need to be set
                #entry.putRelation(outputSetDict.get("sink1"), inputSetDict.get("key"), 0.0) 
                #entry.putRelation(outputSetDict.get("sink2"), inputSetDict.get("key"), 0.0)
                #entry.putRelation(outputSetDict.get("sink3"), inputSetDict.get("key"), 0.0) 

                #entry.putRelation(outputSetDict.get("sink1"), inputSetDict.get("text"), 0.0) 
                #entry.putRelation(outputSetDict.get("sink2"), inputSetDict.get("text"), 0.0)
                #entry.putRelation(outputSetDict.get("sink3"), inputSetDict.get("text"), 0.0) 

                entry.putRelation(outputSetDict.get("sink1"), inputSetDict.get("src1"), 1.0 if (emb==0 or emb==1 or emb==2 or emb==3 ) else 0.0) 
                entry.putRelation(outputSetDict.get("sink2"), inputSetDict.get("src1"), 1.0 if (emb==4 or emb==5 or emb==6 or emb==7 ) else 0.0)
                entry.putRelation(outputSetDict.get("sink3"), inputSetDict.get("src1"), 0.0) 

                entry.putRelation(outputSetDict.get("sink1"), inputSetDict.get("src2"), 0.0) 
                entry.putRelation(outputSetDict.get("sink2"), inputSetDict.get("src2"), 1.0 if (emb==0 or emb==2 or emb==4 or emb==6 ) else 0.0) 
                entry.putRelation(outputSetDict.get("sink3"), inputSetDict.get("src2"), 1.0 if (emb==1 or emb==3 or emb==5 or emb==7 ) else 0.0) 

                entry.putRelation(outputSetDict.get("sink1"), inputSetDict.get("src3"), 1.0 if (emb==2 or emb==3 or emb==6 or emb==7) else 0.0) 
                entry.putRelation(outputSetDict.get("sink2"), inputSetDict.get("src3"), 0.0) 
                entry.putRelation(outputSetDict.get("sink3"), inputSetDict.get("src3"), 1.0 if (emb==0 or emb==1 or emb==4 or emb==5) else 0.0) 
                outputFile.write(f"{entry.toTextFormat()}\n")
        finally:
            if outputFile is not None:
                outputFile.close
        return 
class Encoder:
    class AbstractEncoder(ABC):
        DEFAULT_PROB = 0.0
        def __init__(self, inputSetSize, outputSetSize, nonFloatIndices = None,
                    inputSetSelectedIndices = None, outputSetSelectedIndices = None
                    ) -> None:
            self._inputSetSize = inputSetSize
            self._outputSetSize = outputSetSize
            self._nonFloatIndices = nonFloatIndices if nonFloatIndices is not None else []
            self._inputSetSelectedIndices = inputSetSelectedIndices if inputSetSelectedIndices is not None else list(range(inputSetSize))
            self._outputSetSelectedIndices = outputSetSelectedIndices if outputSetSelectedIndices is not None else list(range(outputSetSize))

        def getInputSetSize(self)->int:
            return self._inputSetSize
        def getOutputSetSize(self)->int:
            return self._outputSetSize
        def getVariableLenInputNum(self)->int:
            return len(self._nonFloatIndices)
        
        @abstractmethod
        def encode(self, entry:DatasetUtility.Entry)->np.ndarray:
            pass
        
        @abstractmethod
        def decode(self, encoded_entry:np.ndarray)->DatasetUtility.Entry:
            pass
        
        @abstractmethod
        def getInputDim(self)->int:
            pass
        
        @abstractmethod
        def getRelDim(self)->int:
            pass
        
        

    class BinaryEntryEncoder(AbstractEncoder):
        def __init__(self, inputSetSize, outputSetSize, nonFloatIndices:list[int] = None, inputSetSelectedIndices = None, outputSetSelectedIndices = None) -> None:
            super().__init__(inputSetSize, outputSetSize, nonFloatIndices, inputSetSelectedIndices, outputSetSelectedIndices)
            return
        
        def encode(self, entry:DatasetUtility.Entry)->np.ndarray:
            fixedLenInputs = []
            variableLenInputs = []
            for inputIndex, inputValue in enumerate(entry._Entry__inputValueSet):
                if inputIndex in self._nonFloatIndices:
                    variableLenInputs.append(inputValue)
                else:
                    fixedLenInputs.append(inputValue)
            fixedLenInputs = np.array(fixedLenInputs, dtype='float32')
            for i in range(len(variableLenInputs)):
                byteArray = bytes(variableLenInputs[i], encoding='utf-8')
                variableLenInputs[i] = np.array([int(byte) for byte in byteArray], dtype='int32')
            #x = np.array(entry._Entry__inputValueSet, dtype='float32')
            y = np.zeros((self._outputSetSize, self._inputSetSize), dtype='float32')
            for outputIndex in range(self._outputSetSize):
                if outputIndex not in self._outputSetSelectedIndices:
                    continue
                relList = entry._Entry__relSet[outputIndex]
                for (inputIndex,prob) in relList:
                    if inputIndex not in self._inputSetSelectedIndices:
                        continue
                    y[outputIndex][inputIndex] = prob
            y = y.flatten()
            return ([fixedLenInputs,*variableLenInputs],[y])
        
        def decode(self, encoded_entry:np.ndarray)->DatasetUtility.Entry:
            # not implemented now
            pass
        
        def getInputDim(self)->int:
            return self._inputSetSize

        def getRelDim(self)->int:
            return self._inputSetSize * self._outputSetSize

    class SparseEntryEncoder(AbstractEncoder):
        NULL_INDEX = -1.0
        def __init__(self, inputSetSize:int, outputSetSize:int, nonFloatIndices:list[int] = None, inputSetSelectedIndices:list[int] = None, outputSetSelectedIndices:list[int] = None,
                    maxInputNum:int = 1) -> None:
            super().__init__(inputSetSize, outputSetSize, nonFloatIndices, inputSetSelectedIndices, outputSetSelectedIndices)
            self._maxInputNum = maxInputNum
            return
        
        def encode(self, entry:DatasetUtility.Entry)->np.ndarray:
            fixedLenInputs = []
            variableLenInputs = []
            for inputIndex, inputValue in enumerate(entry._Entry__inputValueSet):
                if inputIndex in self._nonFloatIndices:
                    variableLenInputs.append(inputValue)
                else:
                    fixedLenInputs.append(inputValue)
            fixedLenInputs = np.array(fixedLenInputs, dtype='float32')
            for i in range(len(variableLenInputs)):
                byteArray = bytes(variableLenInputs[i], encoding='utf-8')
                variableLenInputs[i] = np.array([int(byte) for byte in byteArray], dtype='int32')
            # x = np.array(entry._Entry__inputValueSet, dtype='float32')
            y_index = np.full((self._outputSetSize,self._maxInputNum), Encoder.SparseEntryEncoder.NULL_INDEX, dtype='float32')
            y_prob = np.full((self._outputSetSize,self._maxInputNum), Encoder.SparseEntryEncoder.DEFAULT_PROB, dtype='float32')
            for outputIndex in range(self._outputSetSize):
                if outputIndex not in self._outputSetSelectedIndices:
                    continue
                relList = entry._Entry__relSet[outputIndex]
                sortedRelList = sorted(relList, reverse=True, key=lambda elem: elem[1])[:self._maxInputNum]
                for index,(inputIndex,prob) in enumerate(sortedRelList):
                    if inputIndex not in self._inputSetSelectedIndices:
                        continue
                    if prob == 0.0:
                        break
                    if index >= self._maxInputNum:
                        logging.warning(f'len(rel[{outputIndex}]) is larger than maxInputNum = {self._maxInputNum}, ignore the excessive part')
                        break
                    y_index[outputIndex][index] = inputIndex
                    y_prob[outputIndex][index] = prob
            y_index = y_index.flatten()
            y_prob = y_prob.flatten()
            return ([fixedLenInputs,*variableLenInputs],[y_index,y_prob])
        
        def decode(self, encoded_entry:np.ndarray)->DatasetUtility.Entry:
            # encoded_entry = array of [index, prob] 
            # index: shape = (outputSetSize, maxInputNum)
            # prob: shape = (outputSetSize, maxInputNum)
            entry = DatasetUtility.Entry(self._inputSetSize, self._outputSetSize)
            for outputIndex, (inputIndexes, probs) in enumerate(zip(*encoded_entry)):
                for inputIndex, prob in zip(inputIndexes,probs):
                    if inputIndex == Encoder.SparseEntryEncoder.NULL_INDEX:
                        pass
                    else:
                        entry.putRelation(outputIndex, inputIndex, prob)
            return entry
        
        def getInputDim(self)->int:
            return self._inputSetSize

        def getRelDim(self)->int:
            return self._outputSetSize * self._maxInputNum
        
        def getMaxInputNum(self)->int:
            return self._maxInputNum
        

def __testing():
    datasetSize = 10000
    trainSeed = 0
    testSeed = 1
    '''
    DatasetUtility.GenSample1Data("./dataset/sample1/train_intRange.dataset", datasetSize)
    DatasetUtility.GenSample1Data("./dataset/sample1/test_intRange.dataset", datasetSize)
    DatasetUtility.RandomIntMin = -1000
    DatasetUtility.RandomIntMax = 1000
    DatasetUtility.GenSample1Data("./dataset/sample1/train_smallRange.dataset", datasetSize)
    DatasetUtility.GenSample1Data("./dataset/sample1/test_smallRange.dataset", datasetSize)
    '''
    '''
    arrayLen = 100
    DatasetUtility.GenSample2Data(f"./dataset/sample2/train_intRange_arrayLen{arrayLen}_interval3.dataset", datasetSize, arrayLen)
    DatasetUtility.GenSample2Data(f"./dataset/sample2/test_intRange_arrayLen{arrayLen}_interval3.dataset", datasetSize, arrayLen)
    DatasetUtility.RandomIntMin = -1000
    DatasetUtility.RandomIntMax = 1000
    DatasetUtility.GenSample2Data(f"./dataset/sample2/train_smallRange_arrayLen{arrayLen}_interval3.dataset", datasetSize, arrayLen)
    DatasetUtility.GenSample2Data(f"./dataset/sample2/test_smallRange_arrayLen{arrayLen}_interval3.dataset", datasetSize, arrayLen)
    '''
    
    for embText in DatasetUtility.EmbText:
        DatasetUtility.GenSample3Data(f"./dataset/sample3/train_seed{trainSeed}_{embText.name}.dataset", datasetSize, seed=trainSeed, embText=embText)
        DatasetUtility.GenSample3Data(f"./dataset/sample3/test_seed{testSeed}_{embText.name}.dataset", datasetSize, seed=testSeed, embText=embText)

__testing()