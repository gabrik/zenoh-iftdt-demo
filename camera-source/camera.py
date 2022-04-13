#
# Copyright (c) 2022 ZettaScale Technology
#
# This program and the accompanying materials are made available under the
# terms of the Eclipse Public License 2.0 which is available at
# http://www.eclipse.org/legal/epl-2.0, or the Apache License, Version 2.0
# which is available at https://www.apache.org/licenses/LICENSE-2.0.
#
# SPDX-License-Identifier: EPL-2.0 OR Apache-2.0
#
# Contributors:
#   ZettaScale Zenoh Team, <zenoh@zettascale.tech>
#

from zenoh_flow.interfaces import  Source
import time
import cv2
import io
import numpy as np

class CameraState:
    def __init__(self, configuration):
        self.sleep = 0
        self.outfile = "/tmp/camera-source.csv"
        if configuration['fps'] is not None:
            self.sleep = 1//int(configuration['fps'])
        if configuration['outfile'] is not None:
            self.outfile = configuration['outfile']
        self.camera = cv2.VideoCapture(0)
        self.encode_params = [int(cv2.IMWRITE_JPEG_QUALITY),100]
        self.file = open(self.outfile, "w+")
        self.file.write("node,time_in,time_out,kind")
        self.file.flush()

    def finalize(self):
        self.file.close()
        self.camera.release()

class CameraSource(Source):
    def initialize(self, configuration):
        return CameraState(configuration)

    def finalize(self, state):
        state.finalize()

        return None

    def run(self, _ctx, state):
        intime = time.time_ns()
        _, frame = state.camera.read()
        time.sleep(state.sleep)
        jpeg = cv2.imencode('.jpg', frame, state.encode_params)[1]
        buf = io.BytesIO()
        np.save(buf, jpeg, allow_pickle=True)
        value = buf.getvalue()
        outtime = time.time_ns()

        state.file.write(f'camera-source,{intime},{outtime},source')
        state.file.flush()

        return value


def register():
    return CameraSource