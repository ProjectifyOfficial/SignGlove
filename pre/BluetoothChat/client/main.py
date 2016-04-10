from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label

#XT1068
class Phones:
	class XT1068:
		UUID = '457807C0-4897-11DF-9879-0800200C9A66'
		ADDR = 'E4:90:7E:E3:50:6B'
	class DUOS:
		UUID = '457807C0-4897-11DF-9879-0800200C9A66'
		ADDR = '00:73:E0:54:41:60'

import bluetooth, thread, time, threading

class Interface(BoxLayout):
	
	def display_data(self, socketObj):
		
		while True:
			try:
				data = socketObj.recv(4096)
			except:
				return
			
			if data != '':
				self.ids.hulk.text = data
			
			time.sleep(0.5)
	
	def pair(self):
		self.socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM) #initialize a socket object
		try:
			self.socket.connect((Phones.XT1068.ADDR, 5))
		finally:
			time.sleep(1)
			
		thr = thread.start_new_thread(self.display_data, (self.socket,))
			
	def quit(self):
		try:
			self.socket.close()
		finally:
			exit()
	
class ClientApp(App):
	def build(self):
		return Interface()
	
if __name__ == '__main__':
	ClientApp().run()


