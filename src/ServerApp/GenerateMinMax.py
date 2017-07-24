#-*-coding:utf8;-*-
# qpy:2
# qpy:kivy

# GenerateMinMax.py : Used to generate boundary values for data


from __init__ import *

BAUDRATE = 9600
SENSOR_COUNT = 5


def UpdateSymbols(s):
    for k in symbols.keys():
        symbols[k] = '0 ' + symbols[k]
    symbols[s] = '1 ' + len(symbols) * '0 '


start = False
Letter = None


def Input(fout):                # get inputs.. blocking function
    global start, running, Letter, pos
    pos = 0
    while True:
        if c == 's':
            if not start:
                inp = raw_input('give letter to start: \n')
                Letter = str(inp)
 #               ResetMinMax()
            else:
                fout.write(str(Min) + '\n')
                fout.write(str(Max) + '\n')
                fout.write(str(Letter) + '\n')
                print Min, Max, Letter
            start = not(start)
        if c == 'p':
            running = False


def Process(ser, fout):          # process inputs from serial
    global running, pos
    pos = -1
    while True:
        if running:
            array = Parse(ser)
            print array
            s = ''
            if array != SENSOR_COUNT * [0]:
                for i in range(SENSOR_COUNT):
                    # if array[i] > 1024 or array[i] < 500: #TODO find new
                    # limits
                    if 1 == 1:
                        s = s + str(array[i]) + ' '
                fout.write('{0}{1}'.format(s, pos * '0 ' + '1\n'))
                fout.flush()


class InterfaceMinMax(BoxLayout):
    def __init__(self, as_popup=False, popup_obj=None):
        super(InterfaceMinMax, self).__init__()
        global running
        self.fout = open('data.txt', 'w')
        self.symbols_fout = open('symbols.txt', 'w')
        self.ser = Connect()
        running = False
        inp_thread = thread.start_new_thread(Process, (self.ser, self.fout))
        self.as_popup = as_popup
        self.popup_obj = False

    def start(self):
        global pos
        if self.ids.input_box == '':
            return

        global Letter, running

        if running:
            return
        pos += 1
        Letter = self.ids.input_box.text
        running = True

    def stop(self):

        global running, Letter, pos

        if not running:
            return
        running = False
        time.sleep(0.02)

        self.symbols_fout.write(Letter + ' ')
        self.symbols_fout.flush()
        self.ids.input_box.text = ''

    def quit(self):
        self.ser.close()
        self.fout.close()
        self.symbols_fout.close()
        global running
        running = False
        exit()


class MinMaxApp(App):
    def build(self): return InterfaceMinMax()

# main


if __name__ == '__main__':

    MinMaxApp().run()
