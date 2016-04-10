import kivy,serial,time,thread
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.button import Button

global serialObject

ANDROID = True

try:
	import androidhelper as android
except ImportError:
	try:
		import android
	except:
		ANDROID = False

def connect():
	ser = None
	x = 0
	while x < 10:
		try:
			ser = serial.Serial('/dev/ttyACM{0}'.format(x), 9600)
			return ser
		except:
			x += 1	
	return None

class Interface(BoxLayout):
	
	def display_serial(self, ser):
		while True:
			try:
				s = str(ser.readline())
				self.ids.hulk.text = s; print s
			except:
				print 'Problem'
			finally:
				time.sleep(2)
	
	def start(self):
		print 'foo'
		serialObject = connect()
		assert (serialObject != None)
		thread.start_new_thread(self.display_serial, (serialObject,))
	
		
class SerialViewerApp(App):
	def build(self):
		return Interface()
		
if __name__ == '__main__':
	SerialViewerApp().run()
	
