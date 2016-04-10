
import Queue
from collections import deque, Counter 
import os
import pyttsx
from scipy.signal import convolve2d
import serial
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import struct
from threading import Thread
import time

import cPickle as pickle
import itertools as it
import matplotlib.pyplot as plt
import numpy as np


pass
# =============================================================================
# CLASSES
# =============================================================================

class DataSet:
    def __init__(self, savePath):
        self.savePath = savePath
        self.data = {}
    
    @staticmethod
    def load(filePath):
        with open(filePath, "rb") as f:
            dataSet, dataTuple = pickle.load(f)
            dataSet.data = dict(dataTuple)
            return dataSet
    
    def save(self):
        with open(self.savePath, "wb") as f:
            everything = [self, self.data.items()]
            pickle.dump(everything, f)
    
    def add(self, classType, vector):
        if classType not in self.data:
            self.data[classType] = []
        self.data[classType].append(vector)        

    def combine(self, otherDataSet):
        for k, values in otherDataSet.data.items():
            if k not in self.data:
                self.data[k] = []
            self.data[k].extend(values)

    def getLabelsAndInstances(self, labels=None):
        """ Returns a list with each of the labes and a list with each
        of the instance groups in an array format.
        """
        if labels is None:
            labels = sorted(self.data.keys())
        instances = [np.array(self.data[l]) for l in labels]
        return labels, instances

    def getLabelsAndInstances2(self, specificLabels=None):
        """ Returns two arrays. One contains the labels and the other 
        contains all of the instances, whose labels can be found in the
        initial array.
        """
        # Format into two continious lists/arrays
        labels, instancesGrouppList = self.getLabelsAndInstances(specificLabels)
        
        labels2 = []
        instancesGrouppList2 = []
        for i in range(len(labels)):
            instT = instancesGrouppList[i]
            instancesGrouppList2.append(instT)       
            labels2.append([labels[i]]*instT.shape[0])
        
        return np.hstack(labels2), np.vstack(instancesGrouppList2)    

class SerialWrapper(Thread):
    HEADER = struct.pack("BBBB", 0xA1, 0xB2, 0xC3, 0xD4)
    FORMAT = "BBBB" + "hhhhhh" + "HHHHH" + "BB" + "H"
    
    def __init__(self, port):
        super(SerialWrapper, self).__init__()
        self.ser = serial.Serial(port)
        self.ser.flushInput()
#         self.ser.read(self.ser.inWaiting())
        self.packer = struct.Struct(self.FORMAT)
        self.packetLen = self.packer.size
        self.queue = Queue.Queue()
        self.running = False
        
        self.freq = {"freq":0, "lastT":time.time()}
        
    def getPacket(self):
        if self.queue.empty():
            return None
        else:
            return self.queue.get()
    
    def run(self):
        self.running = True
        
        headerSize = len(self.HEADER)
        
        unprocessedData = ""
        while self.running:
            try:
                unprocessedData += self.ser.read(1)
                unprocessedData += self.ser.read(self.ser.inWaiting())
            except:
                print "Serial was closed?"
                break
            
            # Extract all packets
            while True:
                hLoc = unprocessedData.find(self.HEADER)
                if hLoc == -1:
                    break
                
                unprocessedData = unprocessedData[hLoc:]
                if len(unprocessedData) < self.packetLen:
                    break
                
                packetStr = unprocessedData[:self.packetLen]
                unprocessedData = unprocessedData[self.packetLen:]
                
                if packetStr[1:].find(self.HEADER) != -1:
                    continue
                
                packetT = self.packer.unpack(packetStr)[headerSize:]
                
                # Checksum
                receivedSum = packetT[-1]
                calculatedSum = 0
                for i in range(4, len(packetStr)-2):
                    calculatedSum += ord(packetStr[i])
                if receivedSum == calculatedSum:
                    self.queue.put(packetT[:-1])
                    
#                 #########################
#                 print "Checksum..."
#                 print packetStr.encode("hex")
#                 print unprocessedData.encode("hex")
#                 print packetT
#                 print receivedSum, calculatedSum      
                
                # Frequency check
                t = float(time.time())
                self.freq["freq"] = 1/(t-self.freq["lastT"])
                self.freq["lastT"] = t     
                                
    def close(self):
        self.running = False
        self.ser.close()
        print "Closed serial port..."
    
    def clear(self):
        self.queue = Queue.Queue()
    
class DataGather(Thread):
    
    def __init__(self, port, dataSet, startDelay=0):
        super(DataGather, self).__init__()
        self.glove = SerialWrapper(port)
        self.dataSet = dataSet
        self.signToGather = ""
        self.startDelay = startDelay
        self.running = False
        
    def setSignToGather(self, sign):
        self.signToGather = sign
        
    def run(self):
        time.sleep(self.startDelay)
        
        self.running = True
        self.glove.start()
        while self.running:
            packet = self.glove.getPacket()
            if packet is not None:
                self.dataSet.add(self.signToGather, packet)
                print "Num of instances: ", len(self.dataSet.data[self.signToGather])
            else:
                time.sleep(.001)
                                
    def stop(self):
        self.running = False            
        self.glove.close()

pass
# =============================================================================
# EXECUTION FUNCTIONS
# =============================================================================

def gatherNSamples(serialObj, n):
    gathered = []
    while len(gathered) < n:
        inst = serialObj.getPacket()
        if inst is not None:
            gathered.append(inst)
            print len(gathered)
    return gathered

def gatherData(port, dataSet):
    message = "Type 's' to continue and add the new data to the data set"
    message += "\n"
    message += "Type 'd' to trash the new data"
    message += "\n"
    message += "Type 'delete full sign' to erase all data for this sign\n"
    
    # Request sign
    sign = raw_input("Which sign will you be recording?")    
    
    tempDataSet = DataSet("temp.pickle")
    
    dg = DataGather(port, tempDataSet, startDelay=2)
    dg.setSignToGather(sign)
    dg.start()
    
#     # Print frequency
#     fThread = Thread(target=_printFreq, args=(dg.glove,))
#     fThread.isDaemon() 
#     fThread.start()

    raw_input("Hit Enter to stop...")
    dg.stop()
    decided = False
    while not decided:
        result = raw_input(message)
        if result == "d":
            decided = True
        elif result == "s":
            dataSet.combine(tempDataSet)
            decided = True
        elif result == "delete full sign":
            print "Deleted {} instances".format(len(dataSet.data[sign]))
            dataSet.data[sign] = []
            decided = True            
        else:
            print "\nUnkown command...\n"
        
    if sign in dataSet.data:
        print "Total data for this sign: ", len(dataSet.data[sign])
    else:
        print "Sign was not stored in data set"
    dataSet.save() 
    
def gatherAlphabet(port, dataSet):
    
    totalPerLetter = 500
    totalPerIteration = 100
    
    #-------------------------------------------------------------------------- 
    
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    alphabet = [l for l in alphabet] + ["nothing", "relaxed"] 
    
    message = "Enter to accept and gather \n"
    message += "Type 'quit' to discard data \n\n"
    
    #-------------------------------------------------------------------------- 
    
    tempDataSet = DataSet("temp.pickle")
    glove = SerialWrapper(port)
    glove.start()

    # Start gathering data
    gatheredPerLetter = 0
    while gatheredPerLetter < totalPerLetter:
        np.random.shuffle(alphabet)
        for l in alphabet:
            
            # User input
            while True:
                messageT = message + "Recording {} samples of '{}'\n".format(
                                                        totalPerIteration, l)
                result = raw_input(messageT)
                if result == "":
                    # Record data
                    glove.clear()
                    instances = gatherNSamples(glove, totalPerIteration)
                    
                    if raw_input("Type 'd' to discard\n") == 'd':
                        continue
                    else:
                        [tempDataSet.add(l, inst) for inst in instances]
                        break                                    
                elif result == "quit":
                    # Destroy data
                    return
                else:
                    print "Unkown input...\n"
                    
        gatheredPerLetter += totalPerIteration
                    
    # Decide to store
    message = ">>>>>>>>>>>>>>>>>>>>>>>>>>\n"
    message += "Type 'save' or 'discard' \n"
    message += ">>>>>>>>>>>>>>>>>>>>>>>>>>\n"
    while True:
        result = raw_input(message)
        if result == "save":
            dataSet.combine(tempDataSet)
            dataSet.save()
            return
        elif result == "discard":
            return
        else:
            print "Unkown command"
    
def simpleTrainAndTest(dataSet):
    percentTest = .2
    
    # Format into two continious lists/arrays
    labels, instances = dataSet.getLabelsAndInstances()
    
    labels2 = []
    instances2 = []
    for i in range(len(labels)):
        instT = instances[i]
#         instT = windowData(instances[i], 10)
        instances2.append(instT)       
        labels2.extend([labels[i]]*instT.shape[0])
    instances2 = np.vstack(instances)  
    instances2 = np.vstack(instances2)  
    
    scaledInstances = normalizeData(instances2)
     
    # Separate training from test
    labelsTrain, labelsTest, instanceTrain, instanceTest = train_test_split(
            labels2, scaledInstances, test_size=percentTest)
     
    # Train and test
    clf = SVC()
    clf.fit(instanceTrain, labelsTrain)
     
    result = clf.score(instanceTest, labelsTest)
    
    print "Test and train results was:"
    print result
    
def trainSVM(instanceAndLabels=None, dataSet=None, windowSize=10):
    """ Put in either instanceAndLabels or a dataSet 
    """
    if dataSet is not None:
        # Format into two continious lists/arrays
        labels, instanceGroupsList = dataSet.getLabelsAndInstances()
        
        labels2 = []
        instancesGroupsList2 = []
        for i in range(len(labels)):
            instGroupT = windowData(instanceGroupsList[i], windowSize)
            instancesGroupsList2.append(instGroupT)       
            labels2.extend([labels[i]]*instGroupT.shape[0])
        
        labels = labels2
        instances = np.vstack(instancesGroupsList2) 
    elif instanceAndLabels is not None:
        instances, labels = instanceAndLabels
        
    # Normalizer
    scaler = StandardScaler()
    scaler.fit(instances)
    
    # Train
    clf = SVC()
    scaledInstances = scaler.transform(instances)
    clf.fit(scaledInstances, labels)
    
    return clf, scaler

def testSVM(clf, dataSet):
    labels, instances = dataSet.getLabelsAndInstances()
    predictionsLoL = map(clf.predict, instances)
    
    numWrong = 0
    numPred = 0
    for i in range(len(labels)):
        l = labels[i]
        predList = predictionsLoL[i]
        wrongList = it.ifilter(lambda x: x!=l, predList)
        
        numWrong += len(wrongList)
        numPred += len(predList)
        
    print "Wrong: ", numWrong
    print "Predicted: ", numPred
    print "Percent: ", float(numWrong)/numPred

def gestureToSpeech(port, dataSet):
    buffSize = 10
    countThresh = 9
    
    speechEngine = pyttsx.init()
    clf, scaler = trainSVM(dataSet=dataSet, windowSize=1)
       
    glove = SerialWrapper(port)
    glove.start()
    
    pBuffer = deque(maxlen=buffSize)
    prevOutput = None
    while True:
        # Read and predict
        packet = glove.getPacket()
        if packet is None:
            time.sleep(.001)
            continue
        else:
            instance = scaler.transform(np.array(packet))
            [prediction] = clf.predict(instance)
            pBuffer.append(prediction)
            
#             ##############################
#             print packet, prediction
        
        # Filter output and debounce
        [(mostCommon, count)] = Counter(pBuffer).most_common(1)
        
#         #########################
#         print mostCommon, count
        
        if count > countThresh:
            if mostCommon != prevOutput:
#                 print "======"
#                 print mostCommon
                prevOutput = mostCommon
                
                if mostCommon == "relaxed":
                    print " "
                elif mostCommon != "nothing":
                    print mostCommon
                
                # Speech
                if mostCommon != "nothing" and mostCommon != "relaxed":
                    speechEngine.say(mostCommon)
                    speechEngine.runAndWait()
                   
pass
# =============================================================================
# GRAPH FUNCTIONS
# =============================================================================

def plotClasses(dataSet):
    labels, instanceGroupList = dataSet.getLabelsAndInstances()
    instanceGroupListW = [windowData(instanceArray, 10)
                          for instanceArray in instanceGroupList]
    
    allInstances = np.vstack(instanceGroupList)
#     allInstances = np.vstack(instanceGroupListW)
    scaledInstances = normalizeData(allInstances)
    
    print labels
    
    imageplot = plt.imshow(scaledInstances.T, aspect="auto")
        
    plt.colorbar()
    plt.show()
    
def plotSensor(dataSet):
    
    _, instanceGroupList = dataSet.getLabelsAndInstances()
    allInstances = np.vstack(instanceGroupList)
    
    for i in range(allInstances.shape[1]):
        print "Sensor ", i
        sensor = allInstances[:,i]
        plt.plot(range(sensor.size), sensor)
        plt.show()
    
pass
# =============================================================================
# TEST FUNCTIONS
# =============================================================================

def windowData(instanceArray, windowSize):
    window = np.ones((windowSize, 1))
    windowed = convolve2d(instanceArray, window, mode="valid")
    return windowed[::windowSize,:]

def normalizeData(instanceArray):   
    scaler = StandardScaler()
    scaler.fit(instanceArray)
    return scaler.transform(instanceArray)

def test(port):
    s = serial.Serial(port)
    print s.read(30).encode('hex')
    
def test2(port):
    serWrap = SerialWrapper(port)
    serWrap.start()
    print "aaaaa"
    time.sleep(50)
    serWrap.close()
 
def _printFreq(serialWObj):
    while True:
        time.sleep(1)
        print "Freq: {}".format(serialWObj.freq["freq"])

pass
# =============================================================================
# REPORT FUNCTIONS
# =============================================================================

def signGroups(dataSet, saveFolder=None):
    _, allInstanceGroupList = dataSet.getLabelsAndInstances()
    allInstances = np.vstack(allInstanceGroupList)
    scaler = StandardScaler()
    scaler.fit(allInstances)
    
    # A, B, C, D
    _, instanceGroupList = dataSet.getLabelsAndInstances(["a", "b", "c", "d"])
    scaledInstances = scaler.transform(np.vstack(instanceGroupList))
    
    plt.imshow(scaledInstances.T, aspect="auto")
    plt.title("Sensor Readings for A, B, C, and D")
    plt.xlabel("Instances")
    plt.ylabel("Sensors")
    plt.colorbar()
    if saveFolder is not None:
        plt.savefig(saveFolder+"/abcd.png", bbox_inches="tight")
    plt.show()
    
    # K, P
    _, instanceGroupList = dataSet.getLabelsAndInstances(["k", "p"])
    scaledInstances = scaler.transform(np.vstack(instanceGroupList))
    
    plt.imshow(scaledInstances.T, aspect="auto")
    plt.title("Sensor Readings for K and P")
    plt.xlabel("Instances")
    plt.ylabel("Sensors")
    plt.colorbar()
    if saveFolder is not None:
        plt.savefig(saveFolder+"/kp.png", bbox_inches="tight")
    plt.show()
    
    # I, J, Nothing
    _, instanceGroupList = dataSet.getLabelsAndInstances(["i", "j", "nothing"])
    scaledInstances = scaler.transform(np.vstack(instanceGroupList))
    
    plt.imshow(scaledInstances.T, aspect="auto")
    plt.title("Sensor Readings for I, J, and Nothing")
    plt.xlabel("Instances")
    plt.ylabel("Sensors")
    plt.colorbar()
    if saveFolder is not None:
        plt.savefig(saveFolder+"/ijnothing.png", bbox_inches="tight")
    plt.show()
    
def individualSensors(dataSet, saveFolder=None):
    names = ["X-Accel", "Y-Accel", "Z-Accel", 
             "X-Gyro", "Y-Gyro", "Z-Gyro", 
             "Thumb", "Index", "Middle", "Ring", "Index",
             "Side", "Top"]
    
    _, instanceGroupList = dataSet.getLabelsAndInstances()
    allInstances = np.vstack(instanceGroupList)
    
    for i in range(allInstances.shape[1]):
        sensor = allInstances[:,i]
        plt.plot(range(sensor.size), sensor)
        plt.title("Sensor Readings for " + names[i])
        plt.xlabel("Instances")
        plt.ylabel("Sensor Readings")
        if saveFolder is not None:
            plt.savefig(saveFolder+"/{}.png".format(names[i]), bbox_inches="tight")
        plt.show()

def accuracyOverTime(dataSet, saveFolder=None):
    instanceCountList = [1, 2, 5, 10, 25, 50, 100, 200, 400]
    testOn = 100
    testsPer = 10
    
    labels, instanceGroupList = dataSet.getLabelsAndInstances() 
    numInstancesPer = instanceGroupList[0].shape[0]
        
    results = []
    for count in instanceCountList:
        
        # Multiple times to use average
        scoreList = []
        for i in range(testsPer):
            print "Debug: ", (count, i)
            
            # Get training set and testing set
            indices = np.random.choice(np.arange(numInstancesPer), 
                                       size=testOn+count, replace=False)
            testIndices = np.sort(indices[:testOn])
            trainIndices = np.sort(indices[testOn:])
            
            testLabelGroupList = [[l]*testOn for l in labels]
            trainLabelGroupList = [[l]*count for l in labels]
            testInstanceGroupList = [instanceG[testIndices] 
                                     for instanceG in instanceGroupList]
            trainInstanceGroupList = [instanceG[trainIndices] 
                                     for instanceG in instanceGroupList]
            
            testLabels = np.hstack(testLabelGroupList)
            trainLabels = np.hstack(trainLabelGroupList)
            testInstances = np.vstack(testInstanceGroupList)
            trainInstances = np.vstack(trainInstanceGroupList) 
            
            # Train and predict
            scaler = StandardScaler()
            clf = SVC()
            
            scaler.fit(np.vstack((testInstances, trainInstances)))
            scaledTestInstances = scaler.transform(testInstances)
            scaledTrainInstances = scaler.transform(trainInstances)
            
            clf.fit(scaledTrainInstances, trainLabels)
            scoreList.append(clf.score(scaledTestInstances, testLabels))
        
        results.append(scoreList)
    
    # Average and plot results
    averages = np.average(np.array(results), axis=1)
    plt.plot(instanceCountList, averages)
    plt.title("Accuracy of Variable Size Datasets")
    plt.ylabel("Mean accuracy for all labels")
    plt.xlabel("Instances per label")
    if saveFolder is not None:
        plt.savefig(saveFolder+"/accuracy.png", bbox_inches="tight")
    plt.show()

def confusionMatrix(dataSet, saveFolder=None):
    testPercent = .2
    labels, instances = dataSet.getLabelsAndInstances2()
    scaledInstances = normalizeData(instances)
    
    # Separate training from test
    yTrain, yTest, xTrain, xTest = train_test_split(
            labels, scaledInstances, test_size=testPercent)
     
    # Train and predict
    clf = SVC()
    clf.fit(xTrain, yTrain)
    yPred = clf.predict(xTest)
    cm = confusion_matrix(yTest, yPred)

    labels, _ = dataSet.getLabelsAndInstances()
    
    plt.matshow(cm, aspect="auto")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.yticks(range(28), labels)
    plt.xticks(range(28), labels, rotation=90)
    plt.colorbar()
    plt.grid(True)
    if saveFolder is not None:
        plt.savefig(saveFolder+"/confussion.png", bbox_inches="tight")
    plt.show()

def voltageDividerPlots(saveFolder):
    vcc = 5
    resitances = [10, 15, 20, 25, 30]
    flexValues = np.linspace(11, 27, 10)
    for r in resitances:
        voltages = map(lambda x: float(x)/(x + r)*vcc, flexValues)
        plt.plot(flexValues, voltages)
        
    labels = ["Resistance: {}".format(r) for r in resitances]
    plt.ylim((0,5))
    plt.legend(labels, loc=4)
    plt.title("Flex Sensor Output Ranges")
    plt.xlabel("Flex Sensor Resistance")
    plt.ylabel("Virtual Ground Output")
    if saveFolder is not None:
        plt.savefig(saveFolder+"/resitances.png", bbox_inches="tight")
    plt.show()

def getAllReportPlots(dataSet, saveFolder=None):
    signGroups(dataSet, saveFolder)
    individualSensors(dataSet, saveFolder)
    accuracyOverTime(dataSet, saveFolder)
    confusionMatrix(dataSet, saveFolder)
    voltageDividerPlots(saveFolder)

pass
# =============================================================================
# MAIN
# =============================================================================
 
# Tests
# test()
# test2("/dev/ttyUSB0")
# while True: pass
 
if __name__ == "__main__":
    print "Starting..."
     
    port = "/dev/ttyUSB0"
#     dataSetPath = "alphabet.pickle"
#     dataSetPath = "alphabet_mon.pickle"
    dataSetPath = "alphabet_rob.pickle"
#     dataSetPath = "small_set.pickle"
#     dataSetPath = "test.pickle"
    
    # Make or load data set
    if os.path.exists(dataSetPath):
        print "Load data set"
        dataSet = DataSet.load(dataSetPath)
    else:
        print "New data set"
        dataSet = DataSet(dataSetPath)
        dataSet.save()
        
    # ---------------------------------
    
    # Gather data
#     gatherData(port, dataSet)
#     gatherAlphabet(port, dataSet)

    # ---------------------------------
    
    # Print data set
    print "\nData Set Size:"
    for label, instanceList in sorted(dataSet.data.items()):
        print label, len(instanceList)
        
    # Plotting
#     plotClasses(dataSet)
#     plotSensor(dataSet)
    getAllReportPlots(dataSet, "/home/robert/temp")

    # ---------------------------------

    # Learning
#     simpleTrainAndTest(dataSet)
    
#     clf = trainSVM(dataSet)
#     testSVM(clf)

    # ---------------------------------
    
    # Continious Prediction
#     gestureToSpeech(port, dataSet)
    
    # ---------------------------------
    
    print "Done"
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
