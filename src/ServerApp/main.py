#-*-coding:utf8;-*-
# qpy:2
# qpy:kivy

# ServerApp for ECESCON 9 Hackathon
# Goal: Interpret language for the deaf using TTS

# Imports are located at __init__
# This file hosts the main UI and functionality of the application
from __init__ import *


class Interface(BoxLayout):

    def UpdateValues(self):
        """Constantly updating values being read from sensors"""
        while True:
            if self.running:
                try:
                    self.tilt_state.Update(
                        droid.sensorsReadOrientation().result)
                    if self.ids.use_log is True:
                        self.log_file.write(
                            'rot ' + str(self.tilt_state.Rotation) + '\n')
                except BaseException:
                    print 'Error in reading orientation data'

                try:
                    y = SensorProcessor.Parse(serialObject)

                    print y, [self.symbol_manager.Symbols[i].ActivationPercentage for i in range(len(self.symbol_manager.Symbols))]
                    self.symbol_manager.Update(y)
                    if self.ids.use_log is True:
                        self.log_file.write('sensor data ' + str(y) + '\n')
                except BaseException:
                    print 'Problem in reading sensor values'

    def TiltEvent(self):
        """ Recognise when device is tilted """
        notify('Tilt event!')
        try:
            if self.ids.use_tts.active is True:
                droid.ttsSpeak(self.word)
                notify('tts speak done')
        except BaseException:
            pass

        # cleanup
        self.word = ''
        self.ids.hulk.text += ' '

        if self.bt_activated:  # Send data to client
            droid.bluetoothWrite(self.word)

    def SensorEvent(self, data):
        """ Event called when input is read by sensors """
        self.ids.hulk.text += str(data)
        self.word += str(data)

    def __init__(self):
        """ Initialization """
        super(Interface, self).__init__()
        self.symbol_manager = None
        print 'Data were populated successfully'
        self.tilt_state = TiltState.TiltState()
        self.running = False
        self.update_values_thread = thread.start_new_thread(
            self.UpdateValues, ())
        #self.sensor_data = SensorProcessor.SENSOR_COUNT*[0]
        #self.tilt_data = [0,0,0]

        # subscribers
        self.tilt_state.Subscribers.append(self.TiltEvent)

        self.word = ''
        self.bt_activated = False

        self.log_file = open('log.txt', 'w')

    def wait_bt_accept(self):
        droid.bluetoothAccept()
        self.bt_activated = droid.bluetoothActiveConnections()

    def quit(self):
        try:
            serialObject.close()
            droid.stopSensing()
        except BaseException:
            pass
        try:
            self.log_file.close()
        except BaseException:
            pass
        exit()

    def connect(self):
        if ANDROID_DEVICE and EXEC_PERMISSIONS:
            get_permissions(MyPhone.SERIAL_PORTS[0], MyPhone.USER_ID)
        try:
            global serialObject
            serialObject = Connect()
            notify('Serial Connection Established')
        except BaseException:
            notify('Serial Connection Failed')

    def start(self):
        if self.running:
            notify('already running')
            return

        self.running = True
        self.symbol_manager = SensorProcessor.populate_all('data.txt')
        if len(self.symbol_manager.Subscribers) == 0:
            self.symbol_manager.Subscribers.append(self.SensorEvent)

        try:
            droid.startSensingTimed(1, 200)
        except BaseException:
            notify('Cannot start sensing')

    def stop(self):
        if not self.running:
            notify('already stopped')
            return
        try:
            self.running = False
            droid.stopSensing()
        except BaseException:
            notify('could not stop sensing')

    def pair(self):
        """ Pair with client """
        if ANDROID_DEVICE:
            droid.toggleBluetoothState(True)
            droid.bluetoothMakeDiscoverable(3000)
            bluetooth_conn_thread = thread.start_new_thread(
                self.wait_bt_accept, ())

    def about(self):
        """ Pops-up about dialog """
        popup = Popup(
            title='About',
            content=Label(
                text='Glowing Enigma Team!\nSign Language Interpreter\nfor ECESCON Hackathon 2.0 2016!'),
            size_hint=(
                None,
                None),
            size=(
                470,
                470))
        popup.open()

    def train(self):
        """ Pops-up train dialog """
        train_popup = Popup(title='Train')
        train_popup.content = GenerateMinMax.InterfaceMinMax(True, None)
        train_popup.size = (500, 500)
        train_popup.title_align = 'center'
        train_popup.content.popup_obj = train_popup
        train_popup.open()


class ServerApp(App):
    def build(self): return Interface()


if __name__ == '__main__':
    app = ServerApp()
    app.run()
