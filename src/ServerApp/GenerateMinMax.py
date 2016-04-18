#-*-coding:utf8;-*-
#qpy:2
#qpy:kivy

#FOR TESTING PURPOSES
from __init__ import *

BAUDRATE = 9600
SENSOR_COUNT = 6
  # TODO: fill
  # TODO: fill what? 
  
global symbols
symbols = {'a': '0 0 0 0 1', 'c': '0 0 0 1 0', 'h': '0 0 1 0 0', 'i': '0 1 0 0 0', 'n': '1 0 0 0 0'}

def UpdateSymbols(s):
	for k in symbols.keys():
		symbols[k] = '0 ' + symbols[k]
	symbols[s] = '1 ' +  len(symbols) * '0 '   	  


start = False
Letter = None

def Input(fout):                # get inputs.. blocking function
	global start, running, Letter
	while True:
		if c == 's':
			if start == False:
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


def Process(ser,fout):          # process inputs from serial
	#ResetMinMax()
	
	#thread.start_new_thread(Input, (fout,))    
	#global ser; ser = Connect()

	#global Min, Max
	global running
	while True:
		if running == True:
			array = Parse(ser)
			#print array
			s = ''
			if array != SENSOR_COUNT*[0]:
				for i in range(SENSOR_COUNT):
					if array[i] > 1024 or array[i] < 500: #TODO find new limits
						s = s + str(array[i]) + ' '			
				fout.write('{0}{1}'.format(s, symbols[Letter]))
				fout.flush()
#-----------------------------------------------------------------------------------------------------------------------------------------

#gui

class InterfaceMinMax(BoxLayout):
	def __init__(self, as_popup=False, popup_obj = None):
		super(InterfaceMinMax, self).__init__()
		global running
		self.fout = open('data.txt', 'w')
		self.ser = Connect()
		running = False
		inp_thread = thread.start_new_thread(Process, (self.ser,self.fout))
		self.as_popup = as_popup
		self.popup_obj = False
	
	def boo(self):
		print 'hello'
		
	def start(self):
		if self.ids.input_box == '':
			return
		
		global Letter, running
		
		if running == True:
			return
		Letter = self.ids.input_box.text
		#ResetMinMax()
		running = True
		
	def stop(self):
	   
		global running, Min, Max, Letter
		
		if running == False:
			return
		running = False
		#self.ids.min_max_lbl.text = str(Min) + ' ' + str(Max) + ' ' + str(Letter)
		#self.fout.write('{0}\n{1}\n{2}\n'.format(str(Min), str(Max), str(Letter)))
		#self.fout.flush()
		self.ids.input_box.text = ''
		
	def quit(self):
		self.ser.close()
		self.fout.close()
		global running; running = False
		exit()
		
class MinMaxApp(App):
	build = lambda self: InterfaceMinMax()

#main

if __name__ == '__main__':    
	MinMaxApp().run()
		

