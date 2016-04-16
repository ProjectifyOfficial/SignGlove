import ctypes

"""
	This file contains a wrapper for a simple neural net that uses thw backpropagation algorithm
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

class NeuralWrapper:
	
	def __init__(self):
		self.DLL = ctypes.cdll.LoadLibrary("./libNeuralNetwork.so")
		self.DLL.Initialize.restype = ctypes.c_void_p
		self.DLL.Initialize.argtype = (ctypes.c_int, ctypes.POINTER (ctypes.c_int))
		
		self.DLL.Feed.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double))
		self.DLL.Feed.restype = ctypes.POINTER(ctypes.c_double)
		self.DLL.SaveWeights.argtype = (ctypes.c_void_p,ctypes.c_char_p)
		self.DLL.LoadWeights.argtype = (ctypes.c_void_p,ctypes.c_char_p)
		self.DLL.SaveWeights.restype = ctypes.c_int
		self.DLL.LoadWeights.restype = ctypes.c_int
		self.neural_instance = None
		
	def Initialize(self, data):
		data = CTypesArray(data, datatype=ctypes.c_int)
		self.neural_instance = self.DLL.Initialize(len(data), data.ptr)
		return self.neural_instance
		
	def Feed(self, params, n_ouputs=1):
		params = CTypesArray(params)
		outputs = CTypesArray(n_ouputs*[0])
		feed_ptr = self.DLL.Feed(self.neural_instance, params.ptr, outputs.ptr)
		return outputs.tolist(), feed_ptr
			
	def SaveWeights(self, output_dir):
		output_dir = ctypes.c_char_p(output_dir)
		return self.DLL.SaveWeights(self.neural_instance, output_dir) == 1
		
	def LoadWeights(self, input_dir):
		input_dir = ctypes.c_char_p(input_dir)
		return self.DLL.LoadWeights(self.neural_instance, input_dir) == 1	
		
if __name__ == '__main__':
	
	wrapper = NeuralWrapper()
	foo = wrapper.Initialize([2,5,1])		
	print wrapper.Feed([0,0], 2)	
	wrapper.SaveWeights("./foo.txt")
	wrapper.LoadWeights("./foo.txt")
