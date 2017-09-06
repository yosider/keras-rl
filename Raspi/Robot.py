import picamera
import picamera.array
import cv2

import Body
import numpy as np
import time

class Robot(object):
    """Robot hardware Control."""
    def __init__(self, cam_resolution=(80,80), action_interval=1):
        self.body = Body.Body()
        self.camera = picamera.PiCamera()
        self.camera.resolution = cam_resolution
        time.sleep(2) # camera initialization
        self.stream = picamera.array.PiRGBArray(self.camera)

        self.time = 0
        self.action_interval = action_interval

    def get_view(self):
        # reset the stream
        self.stream.seek(0)
        self.stream.truncate()
        # capture (video port is faster than camera port)
        self.camera.capture(self.stream, 'bgr', use_video_port=True)
        # (縦, 横, channel) : confirmed.
        view = self._normalize(np.array(self.stream.array))
        return view

    def act(self, action):
        if self.time % self.action_interval == 0:
            # Body().NEUTRAL = 276
            necks = action[:2] * (300-250) + 250
            necks = np.clip(necks,250, 300)
            necks = list(map(int, necks))
            legs = action[2:] * (376-176) + 176
            legs = np.clip(legs, 176, 376)
            legs = list(map(int, legs))
            self.body.setPCA9685Duty2(6, *necks)
            self.body.setPCA9685Duty6(0, *legs)
        self.time += 1

    def _normalize(self, arr):
        # normalize mean to 0, std to 1
        return (arr - arr.mean()) / arr.std()

    def __del__(self):
        print("*********del worked!**********")
        self.act(np.array([276]*8))
        self.stream.close()
        self.camera.close()
