import math, serial, time, thread

SENSOR_COUNT = 6
UPDATE_TIME = 10.0

def connect():
    ser = None
    for i in range(10):
        try:
            ser = serial.Serial('/dev/ttyACM{0}'.format(i), 9600)
            for i in range(10):
                ser.readline()
				
            return ser
        except:
            print 'try to open next port'
            continue
    return ser

def parse(ser):
    try:
        a = ser.readline()
        a = a.split(',')
        a = a[:len(a) - 1]
        
        return map(lambda x: int(x), a)
    except:
        print 'mapa'
        return SENSOR_COUNT*[0]

def read():
	global flag
	global dlag
	while True:
		w = raw_input('')
		if w == 'S':
			flag = not(flag)
			if flag == True:
				_min, _max = SENSOR_COUNT*[10000], SENSOR_COUNT*[-10000]
		
		if w == 'C':
			dlag = False
			break
	
        
if __name__ == '__main__':
	global _min,_max
	_min, _max = SENSOR_COUNT*[10000], SENSOR_COUNT*[-10000]
	t = thread.start_new_thread(read, ())
	
	ser = connect()
	global flag,dlag
	flag, dlag = False, True
	
	while dlag == True:
		global y
		y = parse(ser)
		print y
		if flag == True:
			for i in range(SENSOR_COUNT):
				if y[i] < _min[i]:
					_min[i] = y[i]
				if y[i] > _max[i]:
					_max[i] = y[i]
		
		
		time.sleep(UPDATE_TIME/1000)

	print _min
	print _max
