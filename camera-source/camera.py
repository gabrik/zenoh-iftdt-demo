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

from zenoh_flow import Inputs, Outputs, Source
import time
import cv2
import io
import numpy as np

class CameraState:
    def __init__(self, configuration):
        self.sleep = 0
        if configuration['fps'] is not None:
            self.sleep = 1//int(configuration['fps'])
        self.camera = cv2.VideoCapture(0)
        self.encode_params = [int(cv2.IMWRITE_JPEG_QUALITY),100]

    def finalize(self):
        self.camera.release()

class CameraSource(Source):
    def initialize(self, configuration):
        return CameraState(configuration)

    def finalize(self, state):
        state.finalize()
        return None

    def run(self, _ctx, state):
        _, frame = state.camera.read()
        time.sleep(state.sleep)
        jpeg = cv2.imencode('.jpg', frame, state.encode_params)[1]
        buf = io.BytesIO()
        np.save(buf, jpeg, allow_pickle=True)
        return buf.getvalue()


def register():
    return CameraSource