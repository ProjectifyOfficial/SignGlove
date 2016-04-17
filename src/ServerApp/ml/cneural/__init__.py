import time,ctypes
EPSILON = -1

"""
	This file contains a wrapper for a simple neural net that uses the backpropagation algorithm
"""

class CTypesArray:
	
	def __init__(self, arr, datatype=ctypes.c_double):
		self.n = len(arr)
		self.arr = (datatype * self.n)()
		for i in range(self.n):
			self.arr[i] = arr[i]	
		self.arr_as_list = arr
		self.ptr = ctypes.cast(self.arr , ctypes.POINTER(datatype))	
		
	def __setitem__(self, i, v):
		self.arr[i] = v
		
	def tolist(self):
		for i in range(self.n):
			self.arr_as_list[i] = self.arr[i]
		return self.arr_as_list
			
	__getitem__ = lambda self, i: self.arr[i]
	__call__ = lambda self: self.ptr	
	__len__ = lambda self: self.n	

class Network:
	
	def __init__(self):
		self.DLL = ctypes.cdll.LoadLibrary("./libNeuralNetwork.so")
		#initialization methods
		self.DLL.Initialize.restype = ctypes.c_void_p
		self.DLL.Initialize.argtype = (ctypes.c_int, ctypes.POINTER (ctypes.c_int) )
		self.DLL.InitializeFromFile.argtype = (ctypes.c_char_p, ctypes.POINTER(ctypes.c_int))
		self.DLL.InitializeFromFile.restype = ctypes.c_void_p
		
		#functions
		self.DLL.Feed.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double))
		self.DLL.Feed.restype = ctypes.POINTER(ctypes.c_double)
		self.DLL.Train.argtypes = (ctypes.c_void_p, ctypes.c_double)
		self.DLL.Train.restype = ctypes.c_void_p
		self.DLL.SaveWeights.argtype = (ctypes.c_void_p,ctypes.c_char_p)
		self.DLL.LoadWeights.argtype = (ctypes.c_void_p,ctypes.c_char_p)
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
		self.neural_instance = self.DLL.InitializeFromFile(filename, self.n_outputs.ptr)
		self.n_outputs = self.n_outputs[0]
		return self.neural_instance
		
	def Feed(self, params):
		params = CTypesArray(params)
		outputs = CTypesArray(self.n_outputs*[0])
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
		for x in outputs:
			x = 2*(x - 0.5)
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
	def Default():
		network = Network()
		network.Initialize([2,5,1])
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
	
	def __init__(self, symbols, weights_dir=None, filename=None):
		self.State = SymbolState()
		self.ActivationData = None
		self.StartTime = time.time()
		self.Subscribers = []
		self.Symbols = symbols
		self.neural = Network()
		if filename is not None:
			self.neural.InitializeFromFile(filename)
		else:
			self.neural.Initialize([2,3,4])
		if weights_dir is not None:
			self.neural.LoadWeights(weights_dir)
		else:
			self.neural.Train(min_error=0.01)
			self.neural.SaveWeights('./weights.txt')
	
	def Update(self, values):
		dt = time.time() - self.StartTime
				
		candidate_class = self.neural.Vote(values)
		
		if candidate_class != None and len(candidate_class) == 1:
			self.OnActivationChanged(self.Symbols[candidate_class[0]])

		self.StartTime = time.time()
				
	def OnActivationChanged(self, data):  
		for f in self.Subscribers:
			f(data)
	
	#replaced by vote	 
	def GetActivated(self):
		print 'obsolete'
		
	def __str__(self):
		return self.knn.clusters.__str__()	
		
	@staticmethod
	def DummySub(data):
		print 'shake that ' + str(data)
		
	@staticmethod
			
				
if __name__ == '__main__':

	symbol_manager = SymbolManager(['sin','cos'], weights_dir=None, filename='/home/marios/sin.txt')
	symbol_manager.Subscribers.append(SymbolManager.DummySub)
	print 'end start'
	
	x = 0.0
	while x <= 6.28:
		print x
		symbol_manager.Update([x])
		x += 0.01
