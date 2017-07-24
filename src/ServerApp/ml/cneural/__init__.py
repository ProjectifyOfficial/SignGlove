import time
import ctypes
import serial
EPSILON = -1
SENSOR_COUNT = 6
"""
	This file contains a wrapper for a simple neural net that uses the backpropagation algorithm
"""


def Parse(ser):
    try:
        line = ser.readline()
        line = line.split(',')
        line = line[:len(line) - 1]
        if len(line) == SENSOR_COUNT:
            return map(lambda x: int(x), line)
        else:
            return SENSOR_COUNT * [0]
    except BaseException:
        return SENSOR_COUNT * [0]


def Connect(max_tries=10, baudrate=9600):  # serial connect
    ser = None
    for i in range(max_tries):
        try:
            ser = serial.Serial('/dev/ttyACM{0}'.format(i), baudrate)
            return ser
        except BaseException:
            print 'Cannot find serial at /dev/ttyACM{0}'.format(i)
    if ser is None:
        raise Exception('Serial connection failed')
    return ser


class CTypesArray:
    """Wrapper class for ctypes-flavoured arrays"""

    def __init__(self, arr, datatype=ctypes.c_double):
        self.n = len(arr)
        self.arr = (datatype * self.n)()
        for i in range(self.n):
            self.arr[i] = arr[i]
        self.arr_as_list = arr
        self.ptr = ctypes.cast(self.arr, ctypes.POINTER(datatype))

    def __setitem__(self, i, v):
        self.arr[i] = v

    def tolist(self):
        for i in range(self.n):
            self.arr_as_list[i] = self.arr[i]
        return self.arr_as_list

    def __getitem__(self, i): return self.arr[i]

    def __call__(self): return self.ptr

    def __len__(self): return self.n


class Network:
    """Wrapper class for our neural net"""

    def __init__(self):
        self.DLL = ctypes.cdll.LoadLibrary("./libNeuralNetwork.so")
        # initialization methods
        self.DLL.Initialize.restype = ctypes.c_void_p
        self.DLL.Initialize.argtype = (
            ctypes.c_int, ctypes.POINTER(
                ctypes.c_int))
        self.DLL.InitializeFromFile.argtype = (
            ctypes.c_char_p, ctypes.POINTER(ctypes.c_int))
        self.DLL.InitializeFromFile.restype = ctypes.c_void_p
        self.DLL.InitializeFromFileWithSymbols.argtype = (
            ctypes.c_char_p, ctypes.POINTER(ctypes.c_int))
        self.DLL.InitializeFromFileWithSymbols.restype = ctypes.c_void_p
        self.DLL.InitializeFromFileWithSymbolsAndArch.argtype = (
            ctypes.c_char_p, ctypes.POINTER(ctypes.c_int), ctypes.c_char_p, ctypes.c_int)
        self.DLL.InitializeFromFileWithSymbolsAndArch.restype = ctypes.c_void_p

        # functions
        self.DLL.Feed.argtypes = (
            ctypes.c_void_p, ctypes.POINTER(
                ctypes.c_double), ctypes.POINTER(
                ctypes.c_double))
        self.DLL.Feed.restype = ctypes.POINTER(ctypes.c_double)
        self.DLL.Train.argtypes = (ctypes.c_void_p, ctypes.c_double)
        self.DLL.Train.restype = ctypes.c_void_p
        self.DLL.SaveWeights.argtype = (ctypes.c_void_p, ctypes.c_char_p)
        self.DLL.LoadWeights.argtype = (ctypes.c_void_p, ctypes.c_char_p)
        self.DLL.SaveWeights.restype = ctypes.c_int
        self.DLL.LoadWeights.restype = ctypes.c_int
        self.neural_instance = None

    def Initialize(self, data):
        self.n_outputs = data[-1]
        data = CTypesArray(data, datatype=ctypes.c_int)
        self.neural_instance = self.DLL.Initialize(len(data), data.ptr)
        return self.neural_instance

    def InitializeFromFile(self, filename):
        filename = ctypes.c_char_p(filename)
        self.n_outputs = CTypesArray([0], datatype=ctypes.c_int)
        self.neural_instance = self.DLL.InitializeFromFile(
            filename, self.n_outputs.ptr)
        self.n_outputs = self.n_outputs[0]
        return self.neural_instance

    def InitializeFromFileWithSymbols(self, filename):
        filename = ctypes.c_char_p(filename)
        self.n_outputs = CTypesArray([0], datatype=ctypes.c_int)
        self.neural_instance = self.DLL.InitializeFromFileWithSymbols(
            filename, self.n_outputs.ptr)
        self.n_outputs = self.n_outputs[0]
        return self.neural_instance

    def InitializeFromFileWithSymbolsAndArch(self, filename, arch_file, n):
        filename = ctypes.c_char_p(filename)
        arch_file = ctypes.c_char_p(arch_file)
        n = ctypes.c_int(n)
        self.n_outputs = CTypesArray([0], datatype=ctypes.c_int)
        self.neural_instance = self.DLL.InitializeFromFileWithSymbolsAndArch(
            filename, self.n_outputs.ptr, arch_file, n)
        self.n_outputs = self.n_outputs[0]
        return self.neural_instance

    def Feed(self, params):
        params = CTypesArray(params)
        outputs = CTypesArray(self.n_outputs * [0])
        feed_ptr = self.DLL.Feed(self.neural_instance, params.ptr, outputs.ptr)
        return outputs.tolist()

    def Train(self, min_error=0.01):
        self.DLL.Train(self.neural_instance, ctypes.c_double(min_error))

    def SaveWeights(self, output_dir):
        output_dir = ctypes.c_char_p(output_dir)
        return self.DLL.SaveWeights(self.neural_instance, output_dir) == 1

    def LoadWeights(self, input_dir):
        input_dir = ctypes.c_char_p(input_dir)
        return self.DLL.LoadWeights(self.neural_instance, input_dir) == 1

    def Vote(self, params):
        outputs = self.Feed(params)
        # print 'sin: {0}, cos: {1}'.format(outputs[0]*2 - 1, outputs[1]*2 - 1)
        print outputs
        j = 0
        for i in range(len(outputs)):
            if outputs[i] > outputs[j]:
                j = i

        results = [j]

        for i in range(len(outputs)):
            if outputs[i] == outputs[j] and i != j:
                return None
            elif i != j and abs(outputs[i] - outputs[j]) < EPSILON:
                results.append(i)

        return results

    @staticmethod
    def Default(min_error=0.1):
        network = Network()
        network.Initialize([2, 5, 1])
        network.Train(min_error)
        network.SaveWeights('./weights.txt')
        return network


class SymbolState:
    IDLE, VALID_GESTURE = range(2)

    def __init__(self):
        self.CurrentState = SymbolState.IDLE
        self.TimeInside = 0                      # time inside current state

    def SetState(self, newState, dt):

        if self.CurrentState != newState:        # enter new state => reset time
            self.TimeInside = 0
        else:                                    # already in new state => accumulate time
            self.TimeInside += dt

        self.CurrentState = newState


class SymbolManager:

    def __init__(
            self,
            symbols,
            weights_dir=None,
            filename=None,
            with_symbols=True):
        self.State = SymbolState()
        self.ActivationData = None
        self.StartTime = time.time()
        self.Subscribers = []
        self.Symbols = symbols
        self.neural = Network()
        if filename is not None:
            if not with_symbols:
                self.neural.InitializeFromFile(filename)
            else:
                self.neural.InitializeFromFileWithSymbols(filename)
        else:
            self.neural.Initialize([5, 10, 20, 5])
        if weights_dir is not None:
            self.neural.LoadWeights(weights_dir)
        else:
            self.neural.Train(min_error=0.001)
            self.neural.SaveWeights('./weights.txt')

    def Update(self, values):
        dt = time.time() - self.StartTime

        candidate_class = self.neural.Vote(values)
        # TODO Sync
        if candidate_class is not None and len(candidate_class) == 1:
            self.OnActivationChanged(self.Symbols[candidate_class[0]])

        self.StartTime = time.time()

    def OnActivationChanged(self, data):
        for f in self.Subscribers:
            f(data)

    # replaced by self.Vote
    def GetActivated(self):
        print 'obsolete'

    def __str__(self):
        return self.knn.clusters.__str__()

    def FeedFromFile(sel, filename):
        with open(filename, 'r') as f:
            while 1 == 1:
                line = f.readline()
                if not line:
                    break
                line = line.strip('\n').split(' ')
                line = map(lambda x: int(x), line)
                self.Update(line)

    @staticmethod
    def DummySub(data):
        print 'shake that! Everyday I am shuffling:  ' + str(data)

    @staticmethod
    def populate_all(filename, symbols_file, weights_dir=None):
        with open(symbols_file, 'r') as f:
            symbols = f.readline().strip('\n').split(' ')
        print symbols[:len(symbols) - 1]
        return SymbolManager(symbols[:len(symbols) - 1],
                             weights_dir=weights_dir,
                             filename=filename,
                             with_symbols=True)


if __name__ == '__main__':
    #import os; os.chdir('/storage/emulated/0/com.hipipal.qpyplus/projects/ServerApp2')
    ser = Connect()

    #symbol_manager = SymbolManager(['sin','cos','tan'], weights_dir=None, filename='/home/marios/xor.txt', with_symbols=True)

    symbol_manager = SymbolManager.populate_all(
        None, '../../symbols.txt', weights_dir='./weights.txt')
    symbol_manager.Subscribers.append(SymbolManager.DummySub)

    while True:
        y = Parse(ser)
        print y
        symbol_manager.Update(y)
        time.sleep(0.5)
