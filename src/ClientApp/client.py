#-*-coding:utf8;-*-
#qpy:2
#qpy:kivy

# ClientApp for PreSFHMMY Hackathon
# Goal: Interpret language for the deaf using TTS

#kivy imports 
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.checkbox import CheckBox

#python imports
import os, thread, threading, time
import Queue as queue

global UPDATE_TIME, MAX_BYTES
UPDATE_TIME = 0.5
MAX_BYTES = 4096

try:
	import bluetooth
except:
	pass

global ANDROID_DEVICE; ANDROID_DEVICE = True
global q; q = queue.Queue()


#imports for android scripting layer
try:
    import androidhelper as android
except ImportError:
    try:
        import android
    except ImportError:
        ANDROID_DEVICE = False
        import bluetooth, pyttsx
        global engine; engine = pyttsx.init()
        #engine.runAndWait()
        print 'Not an android device'
        
class MyPhone:
    NAME = 'XT1068'
    USER_ID = 'u0_a109'
    SERIAL_PORTS = ['/dev/ttyACM0']
    SL4A_PORT = 5
    BT_UUID = '457807C0-4897-11DF-9879-0800200C9A66'
    BT_ADDR = 'E4:90:7E:E3:50:6B'
    
def _connect_threaded():
	if ANDROID_DEVICE == False:
		try:
			nearby = bluetooth.discover_devices()
			if MyPhone.BT_ADDR in nearby:
				try:
					socket.connect((MyPhone.BT_ADDR, MyPhone.SL4A_PORT))
					return True
				except:
					self.ids.superman.text = 'Problem!'
					return False    
	else:
        try:
            droid.bluetoothConnect()
            return True	
        except:
            self.ids.superman.text = 'Problem!'
            return False   
	
	
def get_data():
    global data
    while flag == True:
        if ANDROID_DEVICE == True:
            try:
                data = droid.bluetoothReadLine()
            except:
            pass
        else:
            try:
                data = socket.recv(MAX_BYTES)
            except:
                pass
				
        if data != '':
            if ANDROID_DEVICE == True:
                if not(droid.ttsIsSpeaking()):
                    droid.ttsSpeak(data)
            else:
                engine.say(data)
				
		time.sleep(UPDATE_TIME)		
		q.put(data)
						

class Interface(BoxLayout):
    def quit(self):
        exit()
    
    def connect(self):
		global conn_thread; conn_thread = thread.start_new_thread(_connect_threaded, ())				
		
    def start(self):
        global flag
        flag = True
        global get_data_thread; get_data_thread = thread.start_new_thread(get_data, ())
        global update_labels_thread; update_labels_thread = thread.start_new_thread(self.update_values, ())
        
    def stop(self):
        flag = not(flag)
        
    def update_values(self):
		
		while flag == True:
			if not (q.empty()):
				try:
					self.ids.superman.text = q.get()
				except:
					pass
				finally:
					time.sleep(UPDATE_TIME)
			    
class ClientApp(App):
	
    def build(self):
        return Interface()
		
if __name__ == '__main__':
    if ANDROID_DEVICE == True:
        try:
            global droid; droid = android.Android()
        except:
            pass
    else:
		global socket; socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)	
		
    
    client = ClientApp()
    client.run()
