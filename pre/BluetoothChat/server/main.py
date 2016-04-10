#-*-coding:utf8;-*-
#qpy:2
#qpy:kivy

from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput

ANDROID = True

try:
	import androidhelper as android
except ImportError:
	try:
		import android
	except:
		ANDROID = False
		try:
			import bluetooth
		except:
			exit()
		
if ANDROID == True:
	global droid; droid = android.Android()
	
class Interface(BoxLayout):
	
	def connect(self):
		try:
			droid.toggleBluetoothState(True)
			droid.bluetoothMakeDiscoverable(4000)
			self.conn = droid.bluetoothAccept()
			droid.makeToast('Connection made with {0}'.format(self.conn.result))
		except:
			pass
				
	def send(self):
		try:
			droid.bluetoothWrite(str(self.ids.superman.text))
			self.ids.superman.text = ''	
		except:
			pass
			
	quit = lambda self: exit()		
	
class ServerApp(App):
	build = lambda self: Interface()
	
if __name__ == '__main__':
	ServerApp().run()
