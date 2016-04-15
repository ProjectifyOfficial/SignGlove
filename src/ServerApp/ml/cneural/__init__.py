import ctypes


class CTypesArray:
	
	def __init__(self, arr, datatype=ctypes.c_double):
		self.n = len(arr)
		self.arr = (datatype * self.n)()
		for i in range(self.n):
			self.arr[i] = arr[i]	
		del arr
		self.ptr = ctypes.cast(self.arr , ctypes.POINTER(datatype))	
		
	def __setitem__(self, i, v):
		self.arr[i] = v
	
	__getitem__ = lambda self, i: self.arr[i]
	__call__ = lambda self: self.ptr	
	__len__ = lambda self: self.n	


class NeuralWrapper:
	
	def __init__(self):
		self.DLL = ctypes.cdll.LoadLibrary("./libNeuralNetwork.so")
		self.DLL.Initialize.restype = ctypes.c_void_p
		self.DLL.Feed.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_double))
	
	def Initialize(self):
		return self.DLL.Initialize()
		
	def Feed(self, network_ptr, params):
		params = CTypesArray(params)
		return self.DLL.Feed(network_ptr, len(params), params.ptr)
				
		
if __name__ == '__main__':
	
	wrapper = NeuralWrapper()
	foo = wrapper.Initialize()		
	wrapper.Feed(foo, [2,4])	
	
