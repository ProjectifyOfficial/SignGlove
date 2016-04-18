import time
#global constants
global SENSOR_COUNT, SYMBOL_COUNT, UPDATE_TIME, VALID_GESTURE_TIME, BAUDRATE, THRESHOLD
SENSOR_COUNT = 6
#SYMBOL_COUNT = 4
UPDATE_TIME = 10
VALID_GESTURE_TIME = 0.01
BAUDRATE = 9600
THRESHOLD = 98.0
from __init__ import Connect, Parse

class Range:
        
    def __init__(self, _min, _max, care=True):
        self.Min = float(_min)
        self.Max = float(_max)
        self.Care = care
        self._Center = 0.5*(_min + _max)
        
    @property
    def Center(self):
        return self._Center
    
    @Center.getter
    def Center(self):
        0.5*(self.Min + self.Max)
		
    def __str__(self):
        return '[{0},{1}]'.format(self.Min, self.Max)

    #------------------------DETERMINES RANGE DATA----------------------
    def Belongs(self, value):
        fvalue = float(value)
        return (fvalue >= self.Min and fvalue <= self.Max) or (not(self.Care))
    #-------------------------------------------------------------------
    
class Symbol:                                       

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
    
    def __str__(self):
        return self.Data    

    def Update(self, values):           # values = SENSOR_COUNT-element array of sensor feedback
        
        for i in range(0, SENSOR_COUNT):
            self.Activated[i] = self.LetterRanges[i].Belongs(values[i])
            
        actives = self.ActivatedCount()
        self.ActivationPercentage = float(actives) / float(SENSOR_COUNT) * 100


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

    def __init__(self, symbols):
        
        self.Symbols = symbols
        self.State = SymbolState()
        self.ActivationData = None
        self.StartTime = time.time()
        self.Subscribers = []                   # subscribers for event OnActivationCahnged

    def Update(self, values):         # values = SENSOR_COUNT-element array of sensor feedback
        
        #SYMBOL_COUNT = len(self.Symbols)
        for i in range(0, len(self.Symbols)):
            self.Symbols[i].Update(values)

        actives = self.GetActivated()
        length = len(actives)
        
        
        dt = time.time() - self.StartTime

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
                
        self.StartTime = time.time()
                

    def OnActivationChanged(self, data):
        for f in self.Subscribers:
            f(data)        
        
    def GetActivated(self):
        actives = []
        
        for i in range(0, len(self.Symbols)):
            if self.Symbols[i].ActivationPercentage >= THRESHOLD:
                actives.append(self.Symbols[i])
                
        return actives
        
    #Subscribers
    @staticmethod
    def DummySub(data):
        print 'found {0} with activation data {1}'.format(data, str(y))        


def populate_all(filename):
    symbols = []
    _min, _max, name = None, None, None
    
    with open(filename) as f:
        k = 0
        while True:
            line = f.readline()
            if not (line) :
                break
            line = ((line.strip('[')).strip(']')).strip('\n')
            line = line.strip(']')
            line = line.split(',')
            if k % 3 == 0:
                _min = map(lambda x: int(x), line)
            if k % 3 == 1:     
                _max = map(lambda x: int(x), line) 
            if k % 3 == 2:
                name = line[0]
                symbols.append(populate(_min, _max, name))                
            k += 1
            
    return SymbolManager(symbols)
 
#def Connect(max_tries = 10):  #serial connect
    #ser = None
    #for i in range(max_tries):
        #try:
            #ser = serial.Serial('/dev/ttyACM{0}'.format(i), BAUDRATE)
            #try:
                #ser.open()
            #except:
                #pass
            #return ser
        #except:
            #print 'Cannot find serial at /dev/ttyACM{0}'.format(i)
    #if ser is None:
        #raise Exception('Serial connection failed')
    #return ser
               
#def Parse(ser):
    #try:
        #line = ser.readline()
        #line = line.split(',')
        #line = line[:len(line) - 1]
        #if len(line) == SENSOR_COUNT:
            #return map(lambda x: int(x), line)
        #else:
            #return SENSOR_COUNT*[0]
    #except:
        #print 'Data could not be parsed. Serial error'
        #return SENSOR_COUNT*[0]
    
def populate(_min, _max, name, care = True):
    ranges = []
    for i in range(len(_min)):
        ranges.append(Range(_min[i], _max[i], care))
    return Symbol(ranges, name)

def RangeOverlap(A,B):
    if (A.Max >= B.Min and A.Max <= B.Max) or (A.Min >= B.Min and A.Min <= B.Max):
        return True
    return False

def Overlap(a, b):
    flag = True
    for i in range(SENSOR_COUNT):
        flag = flag and RangeOverlap(a.LetterRanges[i], b.LetterRanges[i])
        if flag == False:
            return False 
    return True

def OverlapAll(symbol_manager):
    overlaps = []
    for i in range(SENSOR_COUNT):
        for j in range(i+1, SENSOR_COUNT):
            if Overlap(SymbolManager.Symbols[i], SymbolManager.Symbols[j]):
                overlaps.append((SymbolManager.Symbols[i], SymbolManager.Symbols[j]))
    return overlaps
    
if __name__ == '__main__':
    import time, math, serial
    try:                
        symbol_manager = populate_all('data.txt')
        SYMBOL_COUNT = len(symbol_manager.Symbols)
        symbol_manager.Subscribers.append(SymbolManager.DummySub)
    except:
        print 'Data could not be populated'
        
    global y
    
    ser = Connect()
    
    while True:
        try:
            y = Parse(ser)
        except:
            print 'Parsing error'
            break
            
        if len(y) == 0 or y == SENSOR_COUNT*[0]:
            continue
            
        print 'Data: {0}, AP: {1}'.format(y, [symbol_manager.Symbols[i].ActivationPercentage for i in range(SYMBOL_COUNT)])    
        
        symbol_manager.Update(y)
        time.sleep(UPDATE_TIME*1.0 / 1000)
        
    try:
        ser.close()
    except:
        print 'Serial could not be closed'    
