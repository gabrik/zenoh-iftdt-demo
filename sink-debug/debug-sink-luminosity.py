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

from zenoh_flow.interfaces import Sink
import struct

class OutputData:
    def __init__(self, filename="/tmp/zf-iftdt.out"):
        self.file = open(filename, "w+")
    def close(self):
        self.file.close()


class LuminositySink(Sink):
    def initialize(self, configuration):
        return OutputData()

    def finalize(self, state):
        return state.close()

    def run(self, _ctx, state, data):
        monitoring = struct.unpack('f', data.get_data())[0]
        result = f'Monitoring received {monitoring}\n'

        state.file.write(result)
        state.file.flush()


def register():
    return LuminositySink