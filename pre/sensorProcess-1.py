import time
import math
import serial

SENSOR_COUNT = 6
SYMBOL_COUNT = 5
UPDATE_TIME = 10            # 10 ms
VALID_GESTURE_TIME = 0.75    # 100 ms (~ 150)


class Range:                        # defines mathematical range [Min, Max]

    def __init__(self, _min, _max, care=True):
        self.Min = float(_min)
        self.Max = float(_max)
        self.Care = care            # is that needed?

    #------------------------DETERMINES RANGE DATA----------------------
    def Belongs(self, value):
        fvalue = float(value)
        return (fvalue >= self.Min and fvalue <= self.Max) or (not(self.Care))
    #-------------------------------------------------------------------

    def Print(self):
        print "(" + str(self.Min) + ", " + str(self.Max) + ")"

    @staticmethod
    def Default():
        return Range(5, 10, False)

#-------------------------------------------------------------------------


class Symbol:       # defines a symbol with range for sensor data to lie inside

    def __init__(self, ranges, data):
        self.LetterRanges = []
        for i in range(0, SENSOR_COUNT):
            self.LetterRanges.append(ranges[i])

        self.Activated = [False for i in range(0, SENSOR_COUNT)]
        # should never actually be a float value for the given SENSOR_COUNT = 5
        self.ActivationPercentage = float(0)
        # symbol print character
        self.Data = data

    #-----------------DETERMINES ACTIVATION DATA--------------------
    def ActivatedCount(self):
        count = 0
        for i in range(0, SENSOR_COUNT):
            if self.Activated[i]:
                count = count + 1
        return count
    #---------------------------------------------------------------

    # values = SENSOR_COUNT-element array of sensor feedback, dt = time
    # between this and last update
    def Update(self, values, dt):

        for i in range(0, SENSOR_COUNT):
            # print len(values)
            self.Activated[i] = self.LetterRanges[i].Belongs(values[i])

        actives = self.ActivatedCount()
        self.ActivationPercentage = float(actives) / float(SENSOR_COUNT) * 100

    def Print(self):
        for i in range(0, SENSOR_COUNT):
            self.LetterRanges[i].Print()

    @staticmethod
    def Default():
        return Symbol([Range.Default() for i in range(0, SENSOR_COUNT)], 'A')

#-------------------------------------------------------------------------


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

#-------------------------------------------------------------------------


class SymbolManager:

    def __init__(self, symbols):         # TODO: make ctor for custom symbols
        #self.Symbols = [Symbol.Default() for i in range(0, SYMBOL_COUNT)]
        self.Symbols = symbols
        self.State = SymbolState()
        self.ActivationData = None
        self.Subscribers = []                   # subscribers for event OnActivationCahnged

    # values = SENSOR_COUNT-element array of sensor feedback, dt = time
    # between this and last update
    def Update(self, values, dt):

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
                # the only element of actives array
                identifiedGesture = actives[0]

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
            if self.Symbols[i].ActivationPercentage >= 80:
                actives.append(self.Symbols[i])

        return actives

    def Print(self):
        for i in range(0, SYMBOL_COUNT):
            self.Symbols[i].Print()

#-------------------------------------------------------------------------


def Sub(data):
    print 'found {0} with activation data {1}'.format(data, str(y))


def connect():
    ser = None
    for i in range(10):
        try:
            ser = serial.Serial('/dev/ttyACM{0}'.format(i), 9600)
            for i in range(10):
                ser.readline()

            return ser
        except BaseException:
            print 'try to open next port'
            continue
    return ser


def parse(ser):
    try:
        a = ser.readline()
        a = a.split(',')
        a = a[:len(a) - 1]

        return map(lambda x: int(x), a)
    except BaseException:
        print 'mapa'
        return SENSOR_COUNT * [0]


def populate(_min, _max, name):
    ranges = []
    for i in range(len(_min) - 1):
        ranges.append(Range(_min[i], _max[i]))
    ranges.append(Range(0, -1, False))
    return Symbol(ranges, name)


foo = Range(0, -1, False)

NOTHING = Symbol([Range(937, 938), Range(949, 951), Range(
    952, 953), Range(957, 958), Range(953, 956), foo], 'NOTHING')
A = Symbol([Range(925, 932), Range(917, 924), Range(
    914, 926), Range(918, 928), Range(925, 936), foo], 'A')
L = Symbol([Range(941, 947), Range(954, 958), Range(
    924, 927), Range(922, 933), Range(939, 944), foo], 'L')
U = Symbol([Range(911, 915), Range(956, 958), Range(
    953, 955), Range(929, 933), Range(937, 941), foo], 'U')
A_star = populate([936, 921, 918, 919, 945, 881], [
                  943, 934, 928, 928, 955, 899], 'A*')


s = SymbolManager([NOTHING, A, L, U, A_star])

ser = connect()

#f = open('/home/marios/test.txt', 'r')
f = open('log.txt', 'w')


s.Subscribers.append(Sub)
global y
while True:
    try:
        y = parse(ser)
    except BaseException:
        print 'Problem!'
        break
        f.write(str(y) + '\n')

    if (y == SENSOR_COUNT * [0] or len(y) == 0):
        break
    print 'Data {0}. AP: {1}'.format(y, [int(s.Symbols[i].ActivationPercentage) for i in range(SYMBOL_COUNT)])
    s.Update(y, UPDATE_TIME)
    time.sleep(UPDATE_TIME / 1000)

f.close()
try:
    ser.close()
except BaseException:
    print 'ser not found'
