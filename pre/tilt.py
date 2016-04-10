try:
	import androidhelper as android
except:
	try:
		import android
	except:
		print 'problem'
		exit()
	
droid = android.Android()

import time, math, thread

global UPDATE_TIME, EPSILON
UPDATE_TIME = 0.75
EPSILON = [4.5,3,10]


droid.startSensingTimed(1, 500)

norm = lambda x,y,z: math.sqrt(x**2 + y**2 + z**2)

def tilt_finder():
	while True:
		tmp[0] = data[0]
		tmp[1] = data[1]
		tmp[2] = data[2]
	
		time.sleep(UPDATE_TIME)
		
		for i in range(len(flags)):
			flags[i] = (data[i] - tmp[i]) >= EPSILON[i] 
			if flags[i] == True:
				print 'axial tilt: {0}'.format(i)
				#print math.acos(data[2]), math.atan2(data[0], data[1]))

if __name__ == '__main__':
	
	
	global data, tmp
	global flags
	
	
	data = droid.sensorsReadAccelerometer().result
	flags = [False, False, False]
	tmp = [0,0,0]
	
	t = thread.start_new_thread(tilt_finder, ()) 
	
	for i in range(5000):
		data = droid.sensorsReadAccelerometer().result
		try:
			data.append(math.acos(data[2]))
			data.append(math.atan2(data[0], data[1]))
		except ValueError:
			pass
		



	
