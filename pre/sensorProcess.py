import time, math, serial, threading
import Queue as queue

SENSOR_COUNT = 5
SYMBOL_COUNT = 4
UPDATE_TIME = 10            # 10 ms
VALID_GESTURE_TIME = 10     # 100 ms (~ 150)

class Range:                        # defines mathematical range [Min, Max]
    
    def __init__(self, _min, _max, care=True):
        self.Min = float(_min)
        self.Max = float(_max)
        self.Care = care            # is that needed?

    #------------------------DETERMINES RANGE DATA----------------------
    def Belongs(self, value):
        fvalue = float(value)
        return (fvalue >= self.Min and fvalue <= self.Max) and self.Care
    #-------------------------------------------------------------------

    def Print(self):
        print "(" + str(self.Min) + ", " + str(self.Max) + ")"

    @staticmethod
    def Default():
        return Range(5, 10, False)

#--------------------------------------------------------------------------------------------------------------------------------------------------------
class Symbol:       # defines a symbol with range for sensor data to lie inside                                           

    def __init__(self, ranges, data):
        self.LetterRanges = []
        for i in range(0, SENSOR_COUNT):
            self.LetterRanges.append(ranges[i])

        self.Activated = [False for i in range(0, SENSOR_COUNT)]
        self.ActivationPercentage = float(0)                            # should never actually be a float value for the given SENSOR_COUNT = 5
        self.Data = data                                                # symbol print character

    #-----------------DETERMINES ACTIVATION DATA--------------------
    def ActivatedCount(self):
        count = 0
        for i in range(0, SENSOR_COUNT):
            if self.Activated[i] == True:
                count = count + 1
        return count
    #---------------------------------------------------------------
        

    def Update(self, values, dt):           # values = SENSOR_COUNT-element array of sensor feedback, dt = time between this and last update
        
        for i in range(0, SENSOR_COUNT):
            self.Activated[i] = self.LetterRanges[i].Belongs(values[i])
            
        actives = self.ActivatedCount()
        self.ActivationPercentage = float(actives) / float(SENSOR_COUNT) * 100

    def Print(self):
        for i in range(0, SENSOR_COUNT):
            self.LetterRanges[i].Print()

    @staticmethod
    def Default():
        return Symbol([Range.Default() for i in range(0, SENSOR_COUNT)], 'A')

#--------------------------------------------------------------------------------------------------------------------------------------------------------
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
        
#--------------------------------------------------------------------------------------------------------------------------------------------------------
class SymbolManager:

    def __init__(self, symbols=[Symbol.Default() for i in range(0, SYMBOL_COUNT)]):         # TODO: make ctor for custom symbols
        self.Symbols = symbols
        self.State = SymbolState()
        self.ActivationData = None
        self.Subscribers = []                   # subscribers for event OnActivationCahnged

    def Update(self, values, dt):               # values = SENSOR_COUNT-element array of sensor feedback, dt = time between this and last update
        
        for i in range(0, SYMBOL_COUNT):
            self.Symbols[i].Update(values, dt)

        actives = self.GetActivated()
        length = len(actives)

        if length > 1:                                              # ERROR. One gesture at a time
            print 'Found more than one active gestures. Error in ranges.'

        elif length == 0:                                           # IDLE state
            self.State.SetState(SymbolState.IDLE, dt)
            self.ActivationData = None

        else:                                                       # length == 1 => One gesture spotted
            self.State.SetState(SymbolState.VALID_GESTURE, dt)
            
            if self.State.TimeInside >= VALID_GESTURE_TIME:         # gesture identified:
                identifiedGesture = actives[0]                      # the only element of actives array

                if self.ActivationData is None:                     # first time data change after IDLE state
                    self.OnActivationChanged(identifiedGesture.Data)                    
                
                self.ActivationData = identifiedGesture.Data
                

    # event like function
    def OnActivationChanged(self, data):
        for f in self.Subscribers:
            f(data)        
        
    def GetActivated(self):
        actives = []
        
        for i in range(0, SYMBOL_COUNT):
            if self.Symbols[i].ActivationPercentage == 100:
                actives.append(self.Symbols[i])
                
        return actives
            

    def Print(self):
        for i in range(0, SYMBOL_COUNT):
            self.Symbols[i].Print()
            
#--------------------------------------------------------------------------------------------------------------------------------------------------------



def permit():
    script = '''
    setenforce 0 &&
    chmod 777 -R /dev/ttyACM* &&
    chown u0_a109 /dev/ttyACM*
    '''
       
class GestureRecogniserException(Exception):
    pass
			
class GestureRecogniser(threading.Thread):
    
    @staticmethod
    def connect():
        ser = None
        for i in range(10):
            try:
                ser = serial.Serial('/dev/ttyACM{0}'.format(i))
                return ser
            except:
                continue
        return ser
    
    @staticmethod
    def Sub(data): #change to fire event 
        print 'found {0} with activation data {1}'.format(data, str(y))
    
    @staticmethod
    def parse(ser):
        try:
            a = ser.readline()
            a = a.split(','); a = a[:len(a)-2]
            #if self.android_use == True:
            #    a += list(droid.sensorReadAccelerometer().result)
            return map(lambda x: int(x), a)
        except:
            return SENSOR_COUNT*[0]

    def __init__(self, android):
        super(GestureRecogniser, self).__init__()
        if android == True:
            try:
                import androidhelper as android
            except:
                import android
                
            global droid
            droid = android.Android()
            droid.startSensingTimed(1, 100)    
            self.android_use = True
        else:
            self.android_use = False        
                
    @staticmethod
    def Default(testdata = True):
        NOTHING = Symbol([Range(937,938), Range(949,951), Range(952,953), Range(957,958), Range(953,956)],'NOTHING')
        A = Symbol([Range(934,939), Range(926,929), Range(925,927), Range(930,932), Range(936,939)], 'A')
        L = Symbol([Range(941,947), Range(954,958), Range(924,927), Range(922,933), Range(939,944)], 'L')	
        U = Symbol([Range(911,915), Range(956,958), Range(953,955), Range(929,933), Range(937,941)], 'U')	        
            
        s = SymbolManager([NOTHING,A,L,U])
        
        if testdata == True: #use a textfie    
            f = open('test.txt', 'r')
        else:
            f = GestureRecogniser.connect()
        
        s.Subscribers.append(GestureRecogniser.Sub)
        global y
        k = 0
        
        while True:
            y = GestureRecogniser.parse(f)
            if(y == [0, 0, 0, 0, 0] or len(y) == 0):
                break
            print str(k) + ' ' + str(y)
            k+=1
            s.Update(y, UPDATE_TIME)
            time.sleep(UPDATE_TIME/1000)
                
        f.close()
    
    def run(self):
        GestureRecogniser.Default()

if __name__ == '__main__':
    #add queue?
    global q; q = queue.Queue()
    
    
    GestureRecogniser(android=False).start()
