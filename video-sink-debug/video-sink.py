##
## Copyright (c) 2017, 2021 ADLINK Technology Inc.
##
## This program and the accompanying materials are made available under the
## terms of the Eclipse Public License 2.0 which is available at
## http://www.eclipse.org/legal/epl-2.0, or the Apache License, Version 2.0
## which is available at https://www.apache.org/licenses/LICENSE-2.0.
##
## SPDX-License-Identifier: EPL-2.0 OR Apache-2.0
##
## Contributors:
##   ADLINK zenoh team, <zenoh@adlink-labs.tech>
##

from zenoh_flow import Sink
import cv2
import numpy as np
import io

class WindowState:
    def __init__(self):
        self.window_name = "DebugOutput"
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

    def finalize(self):
        cv2.destroyWindow(self.window_name)

class VideoSink(Sink):
    def initialize(self, _configuration):
        return WindowState()

    def finalize(self, state):
        state.finalize()
        return None

    def run(self, _ctx, state, data):
        img = data
        img = np.load(io.BytesIO(img), allow_pickle=True)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        cv2.imshow(state.window_name, img)
        cv2.waitKey(10)



def register():
    return VideoSink


def main():
    import time
    vs = VideoSink()
    state = vs.initialize({})
    camera = cv2.VideoCapture(0)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY),100]
    while True:
        _, img = camera.read()
        time.sleep(0.40)
        jpeg = cv2.imencode('.jpg', img, encode_params)[1]
        buf = io.BytesIO()
        np.save(buf, jpeg, allow_pickle=True)
        vs.run(None, state, buf.getvalue())

if __name__=='__main__':
    main()
