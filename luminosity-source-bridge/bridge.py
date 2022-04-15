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

import zenoh
from zenoh import Reliability, SubMode
from zenoh_flow.interfaces import Source
import time
import struct

value = 0
has_value = False
zenoh.init_logger()

def zlistener(sample):
    global value, has_value
    value = int(sample.payload.decode("utf-8"))
    has_value = True

class LuminosityBridgeState:
    def __init__(self, configuration={}):
        self.key_expr = '/paris/office/gb-jl/luminosity'
        if configuration is not None and configuration.get('key-expr') is not None:
             self.key_expr = configuration['key-expr']

        self.zenoh = zenoh.open(None)
        self.sub = self.zenoh.subscribe(self.key_expr, zlistener)

        self.outfile = "/tmp/luminosity-source.csv"
        if configuration['outfile'] is not None:
            self.outfile = configuration['outfile']
        self.file = open(self.outfile, "w+")
        self.file.write("node,time_in,time_out,kind")
        self.file.flush()

    def close(self):
        self.sub.close()
        self.zenoh.close()

class LuminosityBridge(Source):
    def initialize(self, configuration):
        return LuminosityBridgeState(configuration)

    def finalize(self, state):
        state.close()
        return None

    def run(self, _ctx, state):
        intime = time.time_ns()

        global value, has_value
        while (has_value == False):
            pass
        has_value = False
        ba = bytearray(struct.pack("I", value))

        outtime = time.time_ns()
        state.file.write(f'luminosity-source,{intime},{outtime},source')
        state.file.flush()

        return ba

def register():
    return LuminosityBridge
