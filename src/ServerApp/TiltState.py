import time
from SensorProcessor import Range

global UPDATE_TIME

UPDATE_TIME = 250.0


class TiltState:
    HORIZONTAL, VERTICAL_LEFT, VERTICAL_RIGHT = Range(
        -0.25, 0.25), Range(1.4, 1.8), Range(-1.8, -1.4)
    VALID_TILT_TIME = 0.65

    def __init__(self):
        self.State = TiltState.HORIZONTAL
        self.Subscribers = []
        self.Azimuth, self.Yaw, self.Rotation = 0, 0, 0
        self.StartTime = time.time()

    def Update(self, values):
        if values == 3 * [None]:
            return

        self.Azimuth, self.Yaw, self.Rotation = values

        # print self.Rotation

        if TiltState.HORIZONTAL.Belongs(self.Rotation):
            if (self.State == TiltState.VERTICAL_LEFT or self.State == TiltState.VERTICAL_RIGHT) and (
                    time.time() - self.StartTime) >= TiltState.VALID_TILT_TIME:
                print 'all found'
                for i in range(len(self.Subscribers)):
                    self.Subscribers[i]()
                self.State = TiltState.HORIZONTAL
                return

        if (TiltState.VERTICAL_LEFT.Belongs(self.Rotation) or TiltState.VERTICAL_RIGHT.Belongs(
                self.Rotation)) and self.State == TiltState.HORIZONTAL:
            # change state
            self.StartTime = time.time()
            print 'vertical'
            if TiltState.VERTICAL_LEFT.Belongs(self.Rotation):
                self.State = TiltState.VERTICAL_LEFT
            else:
                self.State = TiltState.VERTICAL_RIGHT

    @staticmethod
    def DummySub():
        print 'Hello'


if __name__ == '__main__':
    import time
    try:
        import androidhelper as android
    except BaseException:
        import android

    global droid
    droid = android.Android()
    droid.startSensingTimed(1, 200)
    tilt = TiltState()
    tilt.Subscribers.append(TiltState.DummySub)

    while True:
        tilt.Update()
        #time.sleep(UPDATE_TIME / 1000)
