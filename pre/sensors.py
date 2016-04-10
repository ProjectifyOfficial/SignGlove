import math 

class Sensor:
	ALL, ACCELEROMETER, MAGNETOMETER, LIGHT = range(1,5)
	
	def read_sensor(self):
		pass

class AccelerometerSensor(Sensor):
	
	def __init__(self, droid, delayTime = 100): #delayTime in ms
		self.droid = droid #use global droid? 
		self.x, self.y, self.z = 0,0,0
		self.nx, self.ny, self.nz = 0,0,0
		self.data = [0,0,0]; self.inclination, self.norm = 0, 0
		self._dx, self._dy, self._dz = 0,0,0
		self._delayTime = delayTime
		try:
			droid.startSensingTimed(Sensor.ACCELEROMETER, delayTime)
		except:
			print 'Sensor may be on'
	
	@property	
	def delayTime(self):
		return self._delayTime
		
	@delayTime.setter
	def delayTime(self, dt):
		try:
			droid.stopSensing()
		except:
			return
		try:
			droid.startSensingTimed(Sensor.ACCELEROMETER, dt)
			self._delayTime = dt
		except:
			return
	
	def read_sensor(self):
		data = self.droid.sensorsReadAccelerometer()
		data = data.result
		self.x, self.y, self.z = data
		self.data = data
		self.norm = math.sqrt(self.x**2 + self.y**2 + self.z**2)
		try:
			self.nx, self.ny, self.nz = self.x / self.norm, self.y / self.norm, self.z / self.norm
			self.inclination = math.acos(self.z)
			self.rotation = math.atan2(self.x, self.y)
		except:
			pass
		return data
	
	def get_change(self):
		if self.data == [None,None,None]:
			tmp = [0,0,0]
		else:
			tmp = self.data
		self.read_sensor()
		self._dx, self._dy, self._dz = self.x - tmp[0], self.y - tmp[1], self.z - tmp[2]
		return [self._dx, self._dy, self._dz]
		
	@property
	def dx(self):
		return self._dx
			
	@property
	def dy(self):
		return self._dy
	
	@property
	def dz(self):
		return self._dz
		
	@dx.getter
	def dx(self):
		self.get_change()
		return self._dx
	
	@dy.getter
	def dx(self):
		self.get_change()
		return self._dy	
	
	@dz.getter
	def dx(self):
		self.get_change()
		return self._dz	
		
	def __del__(self):
		self.droid.stopSensing()

map_range = lambda x, l, h, L, H: float(H - L)/float(h-l) * float(x)  + float(L)

#------- NEEDED (?) ------------------------------------------

#TODO 
class FlexSensor(Sensor):
	
	def __init__(self, id, duino, delayTime, length, pullup_resistor):
		pass
		
	def convert_data(self, data):
		pass #TODO take measures
		
class Hand:
	pass
	
#-----------------------------------------------------------	
		
		 
#test
if __name__ == '__main__':
	import time
	try:
		import androidhelper as android
	except ImportError:
		import android
	except:
		print 'Problem'; exit()
	
	droid = android.Android()
	
	accel = AccelerometerSensor(droid)
	
	print accel.read_sensor()
	
	time.sleep(3)
	
	print accel.get_change()
	
	time.sleep(3)
	
	print accel.dx
