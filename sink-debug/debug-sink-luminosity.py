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

import struct

class LuminositySink(Sink):
    def initialize(self, configuration):
        return None

    def finalize(self, _state):
        return None

    def run(self, _ctx, state, data):
        monitoring = struct.unpack('f', data.data)[0]
        print(f'Monitoring received {monitoring}')

def register():
    return LuminositySink