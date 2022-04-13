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

from zenoh_flow.interfaces import Operator
import struct
import time

MAX = 500.0

class State:
    def __init__(self, configuration):
        self.outfile = "/tmp/normalization.csv"
        if configuration['outfile'] is not None:
            self.outfile = configuration['outfile']
        self.file = open(self.outfile, "w+")
        self.file.write("node,time_in,time_out,kind")
        self.file.flush()


class LuminosityNormalization(Operator):
    def initialize(self, configuration):
        return State(configuration)

    def finalize(self, state):
        state.file.close()
        return None

    def input_rule(self, _ctx, _state, _tokens):
        return True

    def output_rule(self, _ctx, _state, outputs, _deadline_miss = None):
        return outputs

    def run(self, _ctx, _state, inputs):
        intime = time.time_ns()

        # Getting the inputs
        data = inputs.get('LuminosityRaw').get_data()
        value = struct.unpack('f', data)[0]
        outputs = {}

        output = min(value, MAX)
        output = output / MAX

        outputs['LuminosityNorm'] = bytearray(struct.pack("f", output))

        outtime = time.time_ns()
        state.file.write(f'normalization,{intime},{outtime},operator')
        state.file.flush()

        return outputs

def register():
    return LuminosityNormalization