from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.checkbox import CheckBox

import thread, time

class Interface(BoxLayout):
	def goo(self):
		for i in range(10):
			self.ids.text_input.text += '{0}\n'.format(i)
			time.sleep(2)
	
	def moo(self):
		for i in range(10,20):
			self.ids.text_input_data.text += '{0}\n'.format(i)
			time.sleep(1)
	
	def foo(self):
		p = thread.start_new_thread(self.goo, ())			
		q = thread.start_new_thread(self.moo, ())			

class ServerApp(App):
	def build(self):
		return Interface()
		
if __name__ == '__main__':
	ServerApp().run()
