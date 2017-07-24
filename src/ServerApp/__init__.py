# __init__.py
# Usage call imports and instansiate global objects

# kivy imports
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.checkbox import CheckBox
from kivy.uix.popup import Popup

# python imports
import os
import thread
import threading
import serial
import time

# libraries
import TiltState
import SensorProcessor
import GenerateMinMax

# global variable is set to True if Android
global ANDROID_DEVICE
ANDROID_DEVICE = True
global EXEC_PERMISSIONS
EXEC_PERMISSIONS = True

global SENSOR_COUNT, SYMBOL_COUNT, UPDATE_TIME, BAUDRATE, TOTAL_COUNT
SENSOR_COUNT = 5
TOTAL_COUNT = 6


# Serial Communication
def Connect(max_tries=10, baudrate=9600):  # serial connect
    ser = None
    for i in range(max_tries):
        try:
            ser = serial.Serial('/dev/ttyACM{0}'.format(i), baudrate)
            return ser
        except BaseException:
            print 'Cannot find serial at /dev/ttyACM{0}'.format(i)
    if ser is None:
        raise Exception('Serial connection failed')
    return ser


def Parse(ser):
    try:
        line = ser.readline()
        line = line.split(',')
        line = line[:len(line) - 1]
        if len(line) == TOTAL_COUNT:
            line = map(lambda x: int(x), line)
            return line[:SENSOR_COUNT]
        else:
            return SENSOR_COUNT * [0]
    except BaseException:
        return SENSOR_COUNT * [0]

# Set permissions to access serial port


def get_permissions(serial_port_name, user_id):
    def su_exec(cmd): return os.system("su -c '{0}'".format(cmd))
    try:
        su_exec('setenfore 0')
        su_exec('chmod 777 {0}'.format(serial_port_name))
        su_exec('chown {0} {1}'.format(user_id, serial_port_name))
        return True
    except BaseException:
        print 'Cannot get permissions'
        return False


def notify(text):
    if ANDROID_DEVICE:
        droid.makeToast(text)
    else:
        print text


# Android Scripting Layer Imports
try:
    import androidhelper as android
except ImportError:
    try:
        import android
    except ImportError:
        ANDROID_DEVICE = False
        print 'Not an android device'

if ANDROID_DEVICE:
    # droid object and permissions
    global droid
    droid = android.Android()


# Phone details
class MyPhone:
    NAME = 'XT1068'
    USER_ID = 'u0_a109'
    SERIAL_PORTS = ['/dev/ttyACM0']
    SL4A_PORT = 5
    BT_UUID = '457807C0-4897-11DF-9879-0800200C9A66'
    BT_ADDR = 'E4:90:7E:E3:50:6B'
